"""Aggregate throughput with N concurrent streams through the public API.

    python benchmarks/batch.py --model /path/to/lure [--backend cuda] [--concurrency 1,2,4,8] [--frames 200]
                               [--out results/batch.json]

Every stream renders conformance case "a" with EOS blocked for --frames frames, so each request does the same work.
Reports aggregate codec frames per second (12.5 frames/s = one real-time stream) and ms per decode step. Only backends that
set `concurrent = True` (cuda) batch requests together; on the others the streams run one after another and the
aggregate stays flat.
"""
from __future__ import annotations

import threading
import time

from common import C, emit, load, parser

FRAME_RATE = 12.5


def main():
    ap = parser(__doc__.split("\n\n")[0])
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    tts, voice, env = load(args)
    text, direction, _ = C.CASES["a"]
    sampling = {"min_new_tokens": args.frames, "max_new_tokens": args.frames + 1}
    runs = {}
    try:
        list(tts.stream(text, voice, instruct=direction, seed=0))
        for n in [int(x) for x in args.concurrency.split(",")]:
            frames = [0] * n
            errors: list = []

            def work(i):
                infos: list = []
                try:
                    for _ in tts.stream(text, voice, instruct=direction, sampling=sampling, seed=100 + i, infos=infos):
                        pass
                    frames[i] = sum(x.frames for x in infos)
                except Exception as e:
                    errors.append(repr(e))

            threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
            t0 = time.perf_counter()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0
            total = sum(frames)
            runs[n] = {"wall_s": round(wall, 3), "frames": total, "frames_per_s": round(total / wall, 1),
                       "ms_per_step": round(1000 * wall / args.frames, 1),
                       "realtime_streams": round(total / wall / FRAME_RATE, 2), "errors": errors}
            print(f"concurrency {n}: {runs[n]}", flush=True)
    finally:
        tts.close()
    emit(args, {"env": env, "frames_per_request": args.frames, "runs": runs})


if __name__ == "__main__":
    main()
