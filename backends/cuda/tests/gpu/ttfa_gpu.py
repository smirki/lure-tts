"""Warmed time-to-first-audio with a breakdown (not collected by pytest). Needs CUDA and a real model dir.

    python tests/gpu/ttfa_gpu.py --model /path/to/lure --max-batch 1 --out OUT

For every (decode context, first_chunk_frames) pair: 3 warmup calls, then --reps streamed calls of conformance case "a"
(with its direction). Reported per pair:
  ttfa_s               request in -> first audio chunk out
  first_frame_s        engine submit -> first frame (admission, talker prefill forward, first step)
  ms_per_frame_after   mean ms per frame after the first chunk
  dec_first_window_ms  one decode_window of (context + first chunk) frames, timed alone
and once: pre_s, the prompt ids + prefill embedding build before the engine sees the request.
Writes OUT/ttfa_breakdown.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ENGINE = os.path.abspath(os.path.join(ROOT, "..", ".."))
for p in (ROOT, os.path.join(ENGINE, "core"), os.path.join(ENGINE, "conformance")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cases as C  # noqa: E402


def stats(x):
    x = [v for v in x if v is not None]
    return dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)), min=float(min(x))) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--first", default="1,2,4")
    ap.add_argument("--contexts", default="72,25")
    a = ap.parse_args()

    import lure_cuda

    t0 = time.perf_counter()
    be = lure_cuda.load(a.model, max_batch=a.max_batch)
    load_s = time.perf_counter() - t0
    voice = be.make_voice(C.REF_WAV, C.REF_TEXT, voice_id=C.VOICE_ID)
    text, direction, _ = C.CASES["a"]
    rc, _ = be._voice_tensors(voice)
    res = dict(gpu=torch.cuda.get_device_name(), model=a.model, turbo=bool(getattr(be, "turbo", False)), max_batch=a.max_batch,
               load_s=load_s, ref_frames=int(voice.ref_codes.shape[0]), text=text, direction=direction, runs={})

    # pre: prompt + prefill embeds, synced
    pre = []
    for _ in range(a.reps):
        torch.cuda.synchronize()
        ts = time.perf_counter()
        p = be.prompt_ids(text, voice.ref_text, direction)
        with torch.inference_mode():
            emb, _pad = be.prefill(p, voice, "english")
        torch.cuda.synchronize()
        pre.append(time.perf_counter() - ts)
    res["pre_s"] = stats(pre)
    res["prefill_len"] = int(emb.shape[1])

    for ctx in [int(c) for c in a.contexts.split(",")]:
        be.opts["decode_context_frames"] = ctx
        for fc in [int(f) for f in a.first.split(",")]:
            # decode cost of the first window alone
            dts = []
            codes = torch.cat([rc[-ctx:].long().cpu(), rc[:fc].long().cpu()], 0)
            for _ in range(5):
                torch.cuda.synchronize()
                ts = time.perf_counter()
                be.codec.decode_window(codes)
                torch.cuda.synchronize()
                dts.append(time.perf_counter() - ts)
            tt, ff, sm = [], [], []
            for i in range(3 + a.reps):
                ts = time.perf_counter()
                first_t = None
                info = None
                for ch in be.generate_segment(text, voice, instruct=direction, seed=i, stream=True, chunk_frames=4,
                                              first_chunk_frames=fc):
                    if first_t is None and ch.pcm.size:
                        first_t = time.perf_counter() - ts
                    info = ch.info or info
                tot = time.perf_counter() - ts
                if i < 3:
                    continue
                tt.append(info.ttfa_s if info.ttfa_s is not None else first_t)
                ff.append((getattr(info, "extra", None) or {}).get("first_frame_s"))
                if info.frames > 1:
                    sm.append(1000 * (tot - tt[-1]) / max(1, info.frames - fc))
            res["runs"][f"ctx{ctx}_first{fc}"] = dict(ttfa_s=stats(tt), first_frame_s=stats(ff), ms_per_frame_after=stats(sm),
                                                      dec_first_window_ms=1000 * float(np.median(dts)))
            print(f"ctx{ctx} first{fc}", json.dumps(res["runs"][f"ctx{ctx}_first{fc}"]), flush=True)
    be.close()
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "ttfa_breakdown.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
