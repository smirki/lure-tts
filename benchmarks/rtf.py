"""Real-time factor (wall time / audio time) through the public API, streamed and non-streamed.

    python benchmarks/rtf.py --model /path/to/lure [--backend mlx] [--takes 2] [--no-stream] [--opt turbo=0]
                             [--out results/rtf.json]

Short = conformance cases a-d and t1-t6; long = the ~90 s passage (segmented by the core). RTF below 1.0 is faster than
real time. The first render is a warmup and is not counted.
"""
from __future__ import annotations

import time

from common import C, emit, load, parser, stats

SR = 24000


def render(tts, voice, case, seed, stream):
    text, direction, _ = C.CASES[case]
    infos: list = []
    t0 = time.perf_counter()
    audio = tts.speak(text, voice, instruct=direction, seed=seed, infos=infos, stream=stream)
    wall = time.perf_counter() - t0
    return wall, len(audio) / SR, infos


def main():
    ap = parser(__doc__.split("\n\n")[0])
    ap.add_argument("--takes", type=int, default=2)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--skip-long", action="store_true")
    args = ap.parse_args()

    stream = not args.no_stream
    groups = {"short": C.SHORT + C.TAGS, "long": [] if args.skip_long else C.LONG}
    tts, voice, env = load(args)
    result = {"env": env, "stream": stream}
    try:
        render(tts, voice, "a", 0, stream)
        for name, case_ids in groups.items():
            rtf, audio_s, eos = [], 0.0, []
            for k in range(args.takes):
                for case in case_ids:
                    wall, secs, infos = render(tts, voice, case, C.seed(case, k), stream)
                    rtf.append(wall / secs if secs else None)
                    audio_s += secs
                    eos += [i.ended_by_eos for i in infos]
            if rtf:
                result[name] = {"rtf": stats(rtf), "audio_s": round(audio_s, 1),
                                "eos_rate": round(sum(eos) / len(eos), 3) if eos else None}
                print(f"{name}: {result[name]}", flush=True)
    finally:
        tts.close()
    emit(args, result)


if __name__ == "__main__":
    main()
