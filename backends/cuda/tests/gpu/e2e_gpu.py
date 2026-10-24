"""End-to-end GPU checks for the Lure CUDA backend (not collected by pytest; needs CUDA and a real checkpoint).

    python tests/gpu/e2e_gpu.py --model /path/to/lure --out results/<gpu> all
    python tests/gpu/e2e_gpu.py --model /path/to/lure --out results/<gpu> parity render stream batch cancel ttfa

Writes <out>/<step>.json, plus wavs for render. Exits 1 if a hard check fails: parity prefix too short, batch invariance
broken, or cancel leaking a slot. Quality metrics (WER, speaker similarity, tag-read) are computed from the saved wavs
by the conformance scorer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ENGINE = os.path.abspath(os.path.join(ROOT, "..", ".."))
for p in (ROOT, os.path.join(ENGINE, "core"), os.path.join(ENGINE, "conformance")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cases as C  # noqa: E402  (conformance/cases.py)

SHORT = [k for k in C.CASES if C.CASES[k][2] in ("short", "tag")]


def _save(out, name, obj):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(json.dumps({name: obj if len(json.dumps(obj, default=float)) < 4000 else "(see file)"}, indent=1, default=float))


def _wav(path, pcm):
    import soundfile as sf

    sf.write(path, pcm, 24000, subtype="PCM_16")


def _run(be, text, voice, instruct, sampling=None, seed=None, stream=True, **kw):
    chunks = list(be.generate_segment(text, voice, instruct=instruct, sampling=sampling, seed=seed, stream=stream, **kw))
    pcm = np.concatenate([c.pcm for c in chunks]) if chunks else np.zeros(0, np.float32)
    return pcm, chunks[-1].info, chunks


def _frames_of(be, text, voice, instruct, sampling, seed, max_new):
    """Raw codes from the engine (no decode) for parity checks."""
    from lure_cuda.engine import Request, drain
    from lure_cuda.sampling import make_generator, resolve_sampling

    p = be.prompt_ids(text, voice.ref_text, instruct)
    with torch.inference_mode():
        emb, pad = be.prefill(p, voice, "english")
    r = Request(prefill=emb, tts_pad=pad, sampling=resolve_sampling(sampling), max_new_tokens=max_new,
                generator=make_generator(seed, be.device))
    be.engine.submit(r)
    return list(drain(r, timeout=600)), p


def step_parity(be, voice, args):
    """Greedy (talker + sub-talker) engine/graph codes vs qwen-tts generate(non_streaming_mode=True) eager, same weights."""
    from lure_cuda.sampling import SamplingParams

    m = be.m
    rc, spk = be._voice_tensors(voice)
    sp = SamplingParams(do_sample=False, subtalker_do_sample=False)
    res, ok = {}, True
    for k in SHORT:
        text, direction, _ = C.CASES[k]
        max_new = 256
        got, p = _frames_of(be, text, voice, direction, sp, 0, max_new)
        dev = rc.device
        vcp = dict(ref_code=[rc], ref_spk_embedding=[spk], x_vector_only_mode=[False], icl_mode=[True])
        with torch.inference_mode():
            codes, _ = m.generate(input_ids=[torch.tensor([p.input_ids], device=dev)],
                                  instruct_ids=[None if p.instruct_ids is None else torch.tensor([p.instruct_ids], device=dev)],
                                  ref_ids=[torch.tensor([p.ref_ids], device=dev)], voice_clone_prompt=vcp,
                                  languages=["english"], non_streaming_mode=True, max_new_tokens=max_new,
                                  do_sample=False, subtalker_dosample=False, repetition_penalty=1.05)
        want = codes[0].tolist()
        n = 0
        while n < min(len(got), len(want)) and got[n] == want[n]:
            n += 1
        cb0 = 0
        while cb0 < min(len(got), len(want)) and got[cb0][0] == want[cb0][0]:
            cb0 += 1
        res[k] = dict(engine_frames=len(got), hf_frames=len(want), identical=got == want, match_prefix=n,
                      cb0_match_prefix=cb0)
        # bf16: graph vs eager kernels may flip a near-tie argmax late; demand a long exact prefix, report the rest
        ok &= n >= min(24, len(want))
    _save(args.out, "parity", res)
    return ok


def step_render(be, voice, args):
    """Seeded sampled renders of every conformance case (defaults), wavs for WER / ECAPA / tag-read / listening."""
    wdir = os.path.join(args.out, "wav")
    os.makedirs(wdir, exist_ok=True)
    res = {}
    for k, (text, direction, kind) in C.CASES.items():
        for j in range(args.takes):
            seed = C.seed(k, j)
            if kind == "long":
                try:
                    from lure_tts.text import segment_text
                    segs = [s.text for s in segment_text(text)]
                except Exception as exc:  # core not importable: crude sentence split, documented in the json
                    import re
                    segs = [s for s in re.split(r"(?<=[.!?\"])\s+", text) if s.strip()]
                    res.setdefault("_warnings", []).append(f"core segmenter unavailable: {exc!r}")
                parts, infos = [], []
                for i, s in enumerate(segs):
                    pcm, info, _ = _run(be, s, voice, direction, seed=seed + i)
                    parts.append(pcm)
                    infos.append(dict(frames=info.frames, eos=info.ended_by_eos))
                pcm = np.concatenate(parts)
                res[f"{k}_{j}"] = dict(seed=seed, segments=len(segs), seg=infos, seconds=pcm.size / 24000)
            else:
                pcm, info, _ = _run(be, text, voice, direction, seed=seed)
                res[f"{k}_{j}"] = dict(seed=seed, frames=info.frames, eos=info.ended_by_eos, ttfa_s=info.ttfa_s,
                                       wall_s=info.wall_s, rtf=info.wall_s / max(pcm.size / 24000, 1e-6),
                                       stop=info.extra.get("stop_reason"))
            _wav(os.path.join(wdir, f"{k}_{j}.wav"), pcm)
    res["_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    _save(args.out, "render", res)
    return True


def step_stream(be, voice, args):
    """Same seed: streamed chunks vs one full decode ([ref; gen] + proportional cut, the reference path)."""
    res = {}
    for k in ("a", "b", "c"):
        text, direction, _ = C.CASES[k]
        full, i1, _ = _run(be, text, voice, direction, seed=11, stream=False)
        st, i2, ch = _run(be, text, voice, direction, seed=11, stream=True, chunk_frames=4, first_chunk_frames=4)
        same = i1.frames == i2.frames
        n = min(full.size, st.size)
        d = full[:n] - st[:n]
        snr = 10 * np.log10((full[:n] ** 2).sum() / max((d ** 2).sum(), 1e-20)) if n else None
        res[k] = dict(same_frames=same, frames=i2.frames, chunks=len(ch), max_abs=float(np.abs(d).max()) if n else None,
                      snr_db=snr, first_chunk_s=ch[0].pcm.size / 24000 if ch else None)
    _save(args.out, "stream", res)
    return all(v["same_frames"] for v in res.values())


def step_ttfa(be, voice, args):
    text, direction, _ = C.CASES["a"]
    t = []
    for i in range(args.reps):
        _, info, _ = _run(be, text, voice, direction, seed=i, stream=True, chunk_frames=4, first_chunk_frames=4)
        t.append(info.ttfa_s)
    _save(args.out, "ttfa", dict(n=len(t), median_s=float(np.median(t)), p90_s=float(np.percentile(t, 90)),
                                 min_s=float(min(t))))
    return True


def step_batch(be, voice, args):
    """Aggregate throughput at concurrency 1..max_batch, fixed 200 frames (EOS blocked), plus seeded batch invariance."""
    from lure_cuda.sampling import resolve_sampling

    text, direction, _ = C.CASES["a"]
    sp = resolve_sampling(None, min_new_tokens=200)
    res = {}
    for n in [x for x in (1, 2, 4, 8, 16) if x <= be.engine.B]:
        out = {}

        def work(i):
            out[i] = _run(be, text, voice, direction, sampling=sp, seed=100 + i, stream=True, max_new_tokens=201)

        t0 = time.perf_counter()
        ts = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for x in ts:
            x.start()
        for x in ts:
            x.join()
        wall = time.perf_counter() - t0
        frames = sum(o[1].frames for o in out.values())
        res[n] = dict(wall_s=wall, frames=frames, frames_per_s=frames / wall, ms_per_step=1000 * wall / 200,
                      rtf_each=wall / (200 / 12.5))
    # invariance: request 0 alone vs inside the last batch (same seed -> same frames, any batch composition)
    alone, _p = _frames_of(be, text, voice, direction, sp, 100, 201)
    got = {}

    def w2(i):
        got[i] = _frames_of(be, text, voice, direction, sp, 100 + i, 201)[0]

    ts = [threading.Thread(target=w2, args=(i,)) for i in range(be.engine.B)]
    for x in ts:
        x.start()
    for x in ts:
        x.join()
    inv = got[0] == alone
    res["invariant"] = inv
    res["rebases"] = be.engine.rebases
    res["_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    res["_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    _save(args.out, "batch", res)
    return inv


def step_cancel(be, voice, args):
    """8 streams; close half of them after their first chunk; the rest must finish and every slot must free."""
    text = C.PASSAGE[:300]
    done = {}

    def work(i):
        g = be.generate_segment(text, voice, seed=i, stream=True)
        n = 0
        for ch in g:
            n += 1
            if i % 2 == 0 and n == 1:
                g.close()
                break
        done[i] = n

    ts = [threading.Thread(target=work, args=(i,)) for i in range(be.engine.B)]
    for x in ts:
        x.start()
    for x in ts:
        x.join(600)
    t0 = time.time()
    while be.engine.active and time.time() - t0 < 30:
        time.sleep(0.1)
    res = dict(chunks=done, active_after=be.engine.active, all_returned=len(done) == be.engine.B)
    _save(args.out, "cancel", res)
    return res["active_after"] == 0 and res["all_returned"]


STEPS = dict(parity=step_parity, stream=step_stream, ttfa=step_ttfa, render=step_render, batch=step_batch,
             cancel=step_cancel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--decode-context", type=int, default=72)
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("steps", nargs="+", choices=list(STEPS) + ["all"])
    args = ap.parse_args()

    import lure_cuda

    t0 = time.perf_counter()
    be = lure_cuda.load(args.model, max_batch=args.max_batch, use_graphs=not args.no_graphs, dtype=args.dtype,
                          decode_context_frames=args.decode_context)
    load_s = time.perf_counter() - t0
    voice = be.make_voice(C.REF_WAV, C.REF_TEXT, voice_id=C.VOICE_ID)
    _save(args.out, "env", dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                                load_s=load_s, ref_frames=int(voice.ref_codes.shape[0]), graphs=not args.no_graphs,
                                max_batch=args.max_batch, dtype=args.dtype, platform=sys.platform,
                                turbo=bool(getattr(be, "turbo", False))))
    steps = list(STEPS) if "all" in args.steps else args.steps
    failed = []
    for s in steps:
        try:
            if not STEPS[s](be, voice, args):
                failed.append(s)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            failed.append(f"{s}: {exc!r}")
    be.close()
    _save(args.out, "summary", dict(steps=steps, failed=failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
