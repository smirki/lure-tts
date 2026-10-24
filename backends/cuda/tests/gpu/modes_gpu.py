"""Optional real-weights check of every prompt mode and layout on the CUDA backend (not collected by pytest; needs
CUDA). For each model folder and each supported (mode, layout): greedy engine codes (CUDA graphs) vs qwen-tts 0.1.1
`generate()` eager on the same weights, the same requests again as one mixed batch, and one seeded sampled take
through the core API saved as a wav.

    python backends/cuda/tests/gpu/modes_gpu.py --out results/modes --models /models/Qwen3-TTS-12Hz-1.7B-Base ...

Stock checkpoints: Qwen/Qwen3-TTS-12Hz-0.6B-Base, Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice, Qwen/Qwen3-TTS-12Hz-1.7B-Base,
Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign. Use --dtype float32 for a long greedy
match; bf16 drifts after a few frames, so the bf16 prefix is reported only. Exits 1 on an error or an empty take, and in
float32 on a parity prefix shorter than min(48, frames) or a mixed batch that differs from the solo runs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ENGINE = os.path.abspath(os.path.join(ROOT, "..", ".."))
for p in (ROOT, os.path.join(ENGINE, "core"), os.path.join(ENGINE, "conformance")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cases as C  # noqa: E402  (conformance/cases.py: reference voice)

TEXT = "Okay, the engine just made a noise I have never heard before, so I am pulling over right now."
DESIGN = "A warm, calm male voice in his forties, speaking slowly and clearly."
CV_INSTRUCT = "Speak in a cheerful, upbeat tone."
LAYOUTS = ("non_streaming", "streaming")
SEED = 1234


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def _wav(path, pcm, sr=24000):
    import soundfile as sf

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.asarray(pcm, dtype=np.float32), sr, subtype="PCM_16")


def _free():
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------- what to run
def plan(spec):
    """(mode, layout, kwargs for Lure.speak / the prompt) for every supported pair."""
    speaker = "vivian" if "vivian" in spec.speakers else (spec.speakers[0] if spec.speakers else None)
    out = []
    for mode in spec.modes:
        for layout in LAYOUTS:
            kw = {"instruct": None, "speaker": None}
            if mode == "custom_voice":
                kw["speaker"] = speaker
                if spec.instruct == "optional":
                    kw["instruct"] = CV_INSTRUCT
            elif mode == "voice_design":
                kw["instruct"] = DESIGN
            out.append((mode, layout, kw))
    return out


def voices(tts):
    if not tts.spec.is_clone:
        return {}
    return {"icl": tts.make_voice(C.REF_WAV, C.REF_TEXT, name="modes_icl"),
            "xvec": tts.make_voice(C.REF_WAV, None, name="modes_xvec")}


# --------------------------------------------------------------------------------------------- parity
def _prompt(tts, mode, kw):
    from lure_tts.prompt import build_prompt_ids

    ref_text = C.REF_TEXT if mode == "icl" else None
    return build_prompt_ids(tts.tokenizer, TEXT, ref_text, kw["instruct"])


def _engine_request(be, p, voice, mode, layout, kw, sp, seed, max_new):
    from lure_cuda.engine import Request
    from lure_cuda.sampling import make_generator

    with torch.inference_mode():
        emb, trailing, pad = be.prefill_rows(p, voice, "english", mode, layout, kw["speaker"])
    r = Request(prefill=emb, tts_pad=pad, trailing=trailing, sampling=sp, max_new_tokens=max_new,
                generator=make_generator(seed, be.device))
    return be.engine.submit(r)


def _qwen_greedy(be, p, voice, mode, layout, kw, max_new):
    """qwen-tts generate() eager, called the way its wrappers call it."""
    dev = be.device
    t = lambda ids: torch.tensor([ids], device=dev)  # noqa: E731
    args = dict(input_ids=[t(p.input_ids)], instruct_ids=[t(p.instruct_ids) if p.instruct_ids else None],
                languages=["english"], non_streaming_mode=layout == "non_streaming", max_new_tokens=max_new,
                do_sample=False, subtalker_dosample=False, repetition_penalty=1.05)
    if mode in ("icl", "xvec"):
        rc, spk = be._voice_tensors(voice)
        icl = mode == "icl"
        args.update(ref_ids=[t(p.ref_ids) if icl else None], voice_clone_prompt=dict(
            ref_code=[rc if icl else None], ref_spk_embedding=[spk], x_vector_only_mode=[not icl], icl_mode=[icl]))
    elif mode == "custom_voice":
        args["speakers"] = [kw["speaker"]]
    with torch.inference_mode():
        codes, _ = be.m.generate(**args)
    return codes[0].tolist()


def _prefix(a, b, key=lambda f: f):
    n = 0
    while n < min(len(a), len(b)) and key(a[n]) == key(b[n]):
        n += 1
    return n


def step_parity(tts, vs, args, strict):
    from lure_cuda.engine import drain
    from lure_cuda.sampling import SamplingParams

    be = tts.backend
    sp = SamplingParams(do_sample=False, subtalker_do_sample=False)
    res, ok, solo = {}, True, {}
    work = plan(tts.spec)
    for mode, layout, kw in work:
        key = f"{mode}/{layout}"
        p = _prompt(tts, mode, kw)
        voice = vs.get(mode)
        got = list(drain(_engine_request(be, p, voice, mode, layout, kw, sp, 0, args.max_new), timeout=600))
        want = _qwen_greedy(be, p, voice, mode, layout, kw, args.max_new)
        n, n0 = _prefix(got, want), _prefix(got, want, key=lambda f: f[0])
        need = min(48, len(want)) if strict else 0
        res[key] = dict(engine_frames=len(got), qwen_frames=len(want), identical=got == want, match_prefix=n,
                        cb0_match_prefix=n0, required=need, ok=n >= need and len(got) > 0)
        ok &= res[key]["ok"]
        solo[key] = got
    # every request of this model at once: one batch with mixed modes and layouts
    reqs = {}
    for mode, layout, kw in work:
        reqs[f"{mode}/{layout}"] = _engine_request(be, _prompt(tts, mode, kw), vs.get(mode), mode, layout, kw, sp, 0,
                                                   args.max_new)
    mixed = {k: list(drain(r, timeout=600)) == solo[k] for k, r in reqs.items()}
    res["mixed_batch"] = dict(requests=len(reqs), same_as_solo=mixed)
    if strict:
        ok &= all(mixed.values())
    return res, ok


# --------------------------------------------------------------------------------------------- sampled takes
def step_takes(tts, vs, args, wdir):
    res, ok = {}, True
    for mode, layout, kw in plan(tts.spec):
        key = f"{mode}/{layout}"
        infos = []
        t0 = time.perf_counter()
        pcm = tts.speak(TEXT, voice=vs.get(mode), instruct=kw["instruct"], speaker=kw["speaker"], seed=SEED,
                        language="english", layout=layout, infos=infos, x_vector_only=mode == "xvec")
        wall = time.perf_counter() - t0
        _wav(os.path.join(wdir, f"{mode}_{layout}.wav"), pcm)
        r = dict(seconds=pcm.size / 24000, wall_s=wall, frames=sum(i.frames for i in infos),
                 eos=all(i.ended_by_eos for i in infos), ttfa_s=infos[0].ttfa_s if infos else None,
                 extra=infos[0].extra if infos else None)
        r["ok"] = pcm.size > 0
        ok &= r["ok"]
        res[key] = r
    res["_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    return res, ok


# --------------------------------------------------------------------------------------------- driver
def _load(model_dir, dtype, args, cache):
    from lure_tts import Lure

    return Lure.load(model_dir, backend="cuda", cache_dir=cache, dtype=dtype, max_batch=args.max_batch,
                       max_seq_len=args.max_seq_len)


def run_model(model_dir, args):
    name = os.path.basename(os.path.normpath(model_dir))
    out = os.path.join(args.out, name)
    res, ok, tts = {"model_dir": model_dir, "dtype": args.dtype}, False, None
    try:
        tts = _load(model_dir, args.dtype, args, os.path.join(out, "cache"))
        vs = voices(tts)
        res["parity"], ok = step_parity(tts, vs, args, strict=args.dtype == "float32")
        res["takes"], good = step_takes(tts, vs, args, os.path.join(out, "wav"))
        ok &= good
    except Exception:
        res["error"] = traceback.format_exc()
    finally:
        if tts is not None:
            tts.close()
        _free()
    res["ok"] = ok
    _save(os.path.join(out, "modes.json"), res)
    print(json.dumps({name: res}, indent=1, default=float))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", required=True, help="model folders (HF layout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--max-new", type=int, default=200, help="greedy frame cap")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("CUDA is required")
    summary = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "models": {}}
    for d in args.models:
        summary["models"][d] = run_model(d, args)
    _save(os.path.join(args.out, "summary.json"), summary)
    print(json.dumps(summary, indent=1))
    sys.exit(0 if all(summary["models"].values()) else 1)


if __name__ == "__main__":
    main()
