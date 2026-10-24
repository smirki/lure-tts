"""Conformance runner: every backend plugs in through the core API (lure_tts.Lure.load(backend=NAME)), so segmentation,
prompt ids, seeds and sampling are identical by construction and only the backend differs.

    python conformance/runner.py --backend mlx --model /path/to/lure --out runs/mlx [--cases a,b,t1] [--takes 2]
           [--voice-npz voice.npz] [--opt compile_cp=1 --opt pipeline=1] [--no-stream]

Writes into --out:
    ref.wav                      reference voice audio (ECAPA anchor for scoring)
    voice_<backend>.npz          ref_codes [T,16] int32 + spk_emb [D] float32 this run used (cross-backend comparison)
    <case>_k<take>.wav           24 kHz float wav per case and take
    manifest.jsonl               one row per wav: file, case, take, kind, text, direction, seed, segments, frames,
                                 ended_by_eos (per segment), ttfa_s, wall_s, audio_s, rtf, backend, opts
    env.json                     backend, opts, platform, versions
    prompt_check.json            only when the backend exposes dump_layout(): its rows vs reference/prompt_ids.json and the
                                 prompt_modes.json entries that apply to the loaded model

Plug-in contract for a backend: be loadable via lure_tts.backends.load_backend(name, ...) (entry point "lure.backends"),
i.e. load / encode_voice / generate / close as in core/lure_tts/backends/base.py. Optional: dump_layout(text, ref_text,
direction, n_ref_frames, language, mode="icl", layout="non_streaming", speaker=None) -> prefill rows, or
{"layout": rows, "trailing": rows}, used for the prompt check.

Score the output with score.py (Qwen3-ASR WER, ECAPA cosine, tag-read detection).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cases as C  # noqa: E402

SR = 24000


def load_voice_npz(path: str, voice_id: str = C.VOICE_ID):
    from lure_tts.backends.base import Voice
    z = np.load(path)
    return Voice(voice_id, C.REF_TEXT, z["ref_codes"].astype(np.int32), z["spk_emb"].astype(np.float32).reshape(-1),
                 {"source": os.path.abspath(path), "wav_sha256": f"npz:{os.path.basename(path)}"})


def save_voice_npz(path: str, voice):
    np.savez(path, ref_codes=np.asarray(voice.ref_codes, np.int32), spk_emb=np.asarray(voice.spk_emb, np.float32))


def parse_opts(items):
    from lure_tts.cli import parse_opts as p
    return p(items or [])


def run(backend: str, model_dir: str, out: str, case_ids=None, takes: int = 2, voice_npz=None, opts=None,
        stream: bool = True, cache_dir=None, quant=None) -> list[dict]:
    from lure_tts import Lure
    from lure_tts.audio import load_audio, write_wav
    from lure_tts.text import segment_text

    opts = dict(opts or {})
    os.makedirs(out, exist_ok=True)
    case_ids = case_ids or list(C.CASES)
    tts = Lure.load(model_dir, backend=backend, quant=quant, cache_dir=cache_dir or os.path.join(out, ".voices"), **opts)
    rows = []
    bname = str(getattr(tts.backend, "name", None) or tts.backend_name).replace(":", "_").replace("/", "_")
    try:
        write_wav(os.path.join(out, "ref.wav"), load_audio(C.REF_WAV))
        t0 = time.time()
        voice = load_voice_npz(voice_npz) if voice_npz else tts.make_voice(C.REF_WAV, C.REF_TEXT, name=C.VOICE_ID)
        voice_s = time.time() - t0
        save_voice_npz(os.path.join(out, f"voice_{bname}.npz"), voice)
        own = None
        if voice_npz:  # also encode with this backend: how close is its own voice prompt to the shared one?
            codes, spk = tts.worker.call(tts.backend.encode_voice, load_audio(C.REF_WAV), C.REF_TEXT)
            np.savez(os.path.join(out, f"voice_own_{bname}.npz"), ref_codes=codes, spk_emb=spk)
            own = voice_agreement(voice, codes, spk)
        env = {"backend": bname, "backend_target": tts.backend_name, "opts": opts, "quant": quant, "stream": stream, "voice": voice_npz or "own",
               "voice_s": round(voice_s, 3), "own_voice_vs_shared": own, "n_ref_frames": int(voice.n_ref_frames), "platform": platform.platform(),
               "python": sys.version.split()[0], "build_info": getattr(tts.backend, "build_info", None)}
        json.dump(env, open(os.path.join(out, "env.json"), "w"), indent=1)
        if hasattr(tts.backend, "dump_layout"):
            json.dump(prompt_check(tts, voice), open(os.path.join(out, "prompt_check.json"), "w"), indent=1)
        mf = open(os.path.join(out, "manifest.jsonl"), "w")
        for k in range(takes):
            for cid in case_ids:
                text, direction, kind = C.CASES[cid]
                sd = C.seed(cid, k)
                segs = segment_text(text)
                t0 = time.time()
                audio = tts.speak(text, voice, instruct=direction, seed=sd, stream=stream)
                wall = time.time() - t0
                infos = list(tts.last_infos)
                fn = f"{cid}_k{k}.wav"
                write_wav(os.path.join(out, fn), audio)
                audio_s = len(audio) / SR
                row = {"file": fn, "case": cid, "take": k, "kind": kind, "text": text, "direction": direction, "seed": sd,
                       "segments": [s.text for s in segs], "frames": [i.frames for i in infos],
                       "ended_by_eos": [bool(i.ended_by_eos) for i in infos],
                       "ttfa_s": round(infos[0].ttfa_s, 4) if infos and infos[0].ttfa_s is not None else None,
                       "seg_wall_s": [round(i.wall_s or 0, 4) for i in infos], "wall_s": round(wall, 4),
                       "audio_s": round(audio_s, 3), "rtf": round(wall / audio_s, 4) if audio_s else None,
                       "extra": [i.extra for i in infos][:1], "backend": bname, "opts": opts, "stream": stream}
                mf.write(json.dumps(row) + "\n")
                mf.flush()
                rows.append(row)
                print(f"[{bname}] {fn} segs={len(segs)} frames={sum(row['frames'])} eos={all(row['ended_by_eos'])} "
                      f"audio={audio_s:.2f}s rtf={row['rtf']} ttfa={row['ttfa_s']}", flush=True)
        mf.close()
    finally:
        tts.close()
    return rows


def voice_agreement(voice, codes, spk) -> dict:
    """Shared voice vs this backend's own encode: frame count, per-codebook code match, x-vector cosine."""
    a, b = np.asarray(voice.ref_codes), np.asarray(codes)
    n = min(len(a), len(b))
    s1, s2 = np.asarray(voice.spk_emb, np.float64).reshape(-1), np.asarray(spk, np.float64).reshape(-1)
    return {"frames": [int(len(a)), int(len(b))], "cb0_match": round(float((a[:n, 0] == b[:n, 0]).mean()), 4) if n else None,
            "all_match": round(float((a[:n] == b[:n]).mean()), 4) if n else None,
            "spk_cos": round(float(s1 @ s2 / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-12)), 5) if len(s1) == len(s2) else None}


def applicable_mode_entries(spec) -> list[dict]:
    """prompt_modes.json entries whose mode the loaded model runs and whose config ids equal the model's."""
    from lure_tts.prompt import ModelIds
    ref = json.load(open(os.path.join(HERE, "reference", "prompt_modes.json")))
    cfgs: dict = {}
    out = []
    for e in ref["entries"]:
        if e["config"] not in cfgs:
            cfgs[e["config"]] = ModelIds.from_config(os.path.join(HERE, "reference", "configs", e["config"]))
        if e["mode"] in spec.modes and cfgs[e["config"]] == spec.ids:
            out.append(e)
    return out


def _dump_entry(backend, e: dict, n_ref: int) -> dict:
    args = (e["text"], e["ref_text"], e.get("direction"), n_ref, e.get("language", "english"))
    mode, layout = e.get("mode", "icl"), e.get("prompt_layout", "non_streaming")
    if mode == "icl" and layout == "non_streaming":
        got = backend.dump_layout(*args)
    else:
        got = backend.dump_layout(*args, mode=mode, layout=layout, speaker=e.get("speaker"))
    got = got if isinstance(got, dict) else {"layout": got}
    return {"id": e["id"], "n_ref_frames": n_ref, **got}


def prompt_check(tts, voice) -> dict:
    """Ask the backend for its own layout for every applicable reference entry and compare (check_prompt.py)."""
    from check_prompt import check
    result = {}
    ref = json.load(open(os.path.join(HERE, "reference", "prompt_ids.json")))
    sets = [("prompt_ids", ref["entries"] if "icl" in tts.spec.modes else [])]
    sets.append(("prompt_modes", applicable_mode_entries(tts.spec)))
    for name, ref_entries in sets:
        entries, unsupported = [], []
        for e in ref_entries:
            n_ref = int(voice.n_ref_frames) if name == "prompt_ids" and voice is not None else e["n_ref_frames"]
            try:
                entries.append(_dump_entry(tts.backend, e, n_ref))
            except (TypeError, NotImplementedError, ValueError) as err:
                unsupported.append(f"{e['id']}: {err}")
        n_ok, _n, report = check({"backend": tts.backend_name, "entries": entries}, {"entries": ref_entries})
        result[name] = {"ok": n_ok, "total": len(ref_entries), "report": report, "unsupported": unsupported}
    return result


def summarize(rows: list[dict]) -> dict:
    """Local summary (no ASR): EOS rate, frames, RTF, TTFA per case kind."""
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        eos = [e for r in rs for e in r["ended_by_eos"]]
        out[kind] = {"n": len(rs), "eos_rate": round(float(np.mean(eos)), 3) if eos else None,
                     "rtf_mean": round(float(np.mean([r["rtf"] for r in rs if r["rtf"]])), 4),
                     "ttfa_med": round(float(np.median([r["ttfa_s"] for r in rs if r["ttfa_s"] is not None])), 4),
                     "audio_s": round(sum(r["audio_s"] for r in rs), 2)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", default=os.environ.get("LURE_MODEL"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=None, help="comma list or short|tags|long|all (default all)")
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--voice-npz", default=None, help="shared voice (ref_codes, spk_emb) so every backend clones the same prompt")
    ap.add_argument("--opt", action="append", default=[], help="backend option KEY=VALUE (repeatable)")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--no-stream", action="store_true")
    a = ap.parse_args(argv)
    groups = {"short": C.SHORT, "tags": C.TAGS, "long": C.LONG, "all": list(C.CASES)}
    ids = None
    if a.cases:
        ids = groups.get(a.cases) or [c.strip() for c in a.cases.split(",") if c.strip()]
        bad = [c for c in ids if c not in C.CASES]
        if bad:
            ap.error(f"unknown cases {bad}")
    rows = run(a.backend, a.model, a.out, ids, a.takes, a.voice_npz, parse_opts(a.opt), not a.no_stream, quant=a.quant)
    s = summarize(rows)
    json.dump(s, open(os.path.join(a.out, "summary_local.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
