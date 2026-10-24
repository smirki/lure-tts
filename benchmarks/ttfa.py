"""Time to first audio through the public API: request in, first PCM chunk out, after warmup.

    python benchmarks/ttfa.py --model /path/to/lure [--backend cuda] [--reps 20] [--first-chunk-frames 1,2,4]
                              [--opt turbo=0] [--out results/ttfa.json]

Measures `Lure.stream()` on conformance case "a" (one segment, so segmenting adds nothing). Reports the wall-clock time
to the first chunk seen by the caller and the backend's own per-segment ttfa_s.
"""
from __future__ import annotations

import time

from common import C, emit, load, parser, stats


def main():
    ap = parser(__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--first-chunk-frames", default="4", help="comma list; 1 frame = 80 ms of audio")
    ap.add_argument("--case", default="a", choices=sorted(C.CASES))
    args = ap.parse_args()

    tts, voice, env = load(args)
    text, direction, _ = C.CASES[args.case]
    runs = {}
    try:
        for fc in [int(x) for x in args.first_chunk_frames.split(",")]:
            wall, backend = [], []
            for i in range(args.warmup + args.reps):
                infos: list = []
                t0 = time.perf_counter()
                first = None
                for pcm in tts.stream(text, voice, instruct=direction, seed=i, infos=infos, first_chunk_frames=fc):
                    if first is None and len(pcm):
                        first = time.perf_counter() - t0
                if i >= args.warmup:
                    wall.append(first)
                    backend.append(infos[0].ttfa_s if infos else None)
            runs[f"first_chunk_frames={fc}"] = {"ttfa_s": stats(wall), "backend_ttfa_s": stats(backend)}
            print(f"first_chunk_frames={fc}: {runs[f'first_chunk_frames={fc}']}", flush=True)
    finally:
        tts.close()
    emit(args, {"env": env, "case": args.case, "runs": runs})


if __name__ == "__main__":
    main()
