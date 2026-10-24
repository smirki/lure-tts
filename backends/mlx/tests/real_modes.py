"""Run every mode a checkpoint supports once with the mlx backend and save the wavs (optional manual check).

Not a pytest test: it loads a full checkpoint. Run it by hand on Apple silicon, one model at a time:

    python backends/mlx/tests/real_modes.py /path/to/model --out runs/real_modes

Per variant:
  Base (stock and Lure family)  icl and xvec, each in the non_streaming and streaming layouts; the Lure family also
                                  runs icl with an instruct
  CustomVoice                     custom_voice in both layouts; on 1.7B also with an instruct
  VoiceDesign                     voice_design in both layouts (the instruct is the voice)
Every run streams (chunked decode); the model's default mode and layout also run once with stream=False (whole decode).

Output: <out>/<model name>/<run>.wav plus manifest.json (mode, layout, frames, EOS, audio seconds, wall time, peak
memory). The script refuses to start when less than --min-free percent of memory is free (macOS memory_pressure).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
REF_WAV = os.path.join(ROOT, "conformance", "assets", "reference_voice.wav")
REF_TEXT = ("Yes, the weekend order can stay exactly as it is. "
            "Everything's fine on our end, truly it is, thank you for checking.")
TEXT = "The kettle clicked off just as the rain started, so I stayed in and finally read the letter."
CLONE_INSTRUCT = "Calm and warm, with a small smile."
SPEAKER_INSTRUCT = "Speak slowly, in a gentle and reassuring tone."
DESIGN_INSTRUCT = "A warm, low female voice in her forties, relaxed and clear, close to the microphone."


def free_percent() -> float | None:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return float(m.group(1)) if m else None


def plan(spec) -> list[dict]:
    """One dict per run: name plus speak() keyword arguments (voice is filled in later)."""
    layouts = ("non_streaming", "streaming")
    runs = []
    if spec.variant == "base":
        for layout in layouts:
            runs.append({"name": f"icl_{layout}", "mode": "icl", "layout": layout})
            runs.append({"name": f"xvec_{layout}", "mode": "xvec", "layout": layout, "x_vector_only": True})
        if spec.instruct != "none":
            runs.append({"name": "icl_non_streaming_instruct", "mode": "icl", "layout": "non_streaming",
                         "instruct": CLONE_INSTRUCT})
    elif spec.variant == "custom_voice":
        speaker = spec.speakers[0]
        for layout in layouts:
            runs.append({"name": f"custom_voice_{layout}", "mode": "custom_voice", "layout": layout, "speaker": speaker})
        if spec.instruct != "none":
            runs.append({"name": "custom_voice_non_streaming_instruct", "mode": "custom_voice", "layout": "non_streaming",
                         "speaker": speaker, "instruct": SPEAKER_INSTRUCT})
    else:
        for layout in layouts:
            runs.append({"name": f"voice_design_{layout}", "mode": "voice_design", "layout": layout,
                         "instruct": DESIGN_INSTRUCT})
    default = next(r for r in runs if r["mode"] == spec.modes[0] and r["layout"] == spec.layout)
    runs.append(dict(default, name=default["name"] + "_whole", stream=False))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "real_modes"))
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--ref-wav", default=REF_WAV, help="reference clip for Base checkpoints")
    ap.add_argument("--ref-text", default=REF_TEXT, help="transcript of --ref-wav")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--memory-limit-gb", type=float, default=6.0)
    ap.add_argument("--cache-limit-mb", type=int, default=128)
    ap.add_argument("--min-free", type=float, default=35.0, help="refuse to start below this free memory percentage")
    args = ap.parse_args()

    from lure_tts import Lure
    from lure_tts.audio import write_wav
    from lure_tts.model import inspect

    free = free_percent()
    if free is not None and free < args.min_free:
        sys.exit(f"only {free:.0f}% of memory is free (need {args.min_free:.0f}%): close apps or reboot, then retry")
    spec = inspect(args.model_dir)
    runs = plan(spec)
    out = os.path.join(args.out, re.sub(r"[^A-Za-z0-9._-]+", "_", spec.name))
    os.makedirs(out, exist_ok=True)
    print(f"{spec.name}: {spec.variant} {spec.size_label}, modes {spec.modes}, default layout {spec.layout}; "
          f"{len(runs)} runs -> {out}")

    t0 = time.time()
    tts = Lure.load(args.model_dir, backend="mlx", cache_dir=os.path.join(out, "voices"), dtype=args.dtype,
                      memory_limit_gb=args.memory_limit_gb, cache_limit_mb=args.cache_limit_mb)
    manifest = {"model": spec.to_dict(), "load_s": round(time.time() - t0, 2), "text": args.text, "runs": []}
    try:
        voice = tts.make_voice(args.ref_wav, args.ref_text) if spec.is_clone else None
        for r in runs:
            kw = {k: v for k, v in r.items() if k not in ("name", "mode")}
            infos: list = []
            t1 = time.time()
            audio = tts.speak(args.text, voice if spec.is_clone else None, seed=args.seed, infos=infos, **kw)
            wall = time.time() - t1
            path = os.path.join(out, r["name"] + ".wav")
            write_wav(path, audio)
            row = {"name": r["name"], "mode": r["mode"], "layout": r["layout"], "stream": r.get("stream", True),
                   "speaker": r.get("speaker"), "instruct": r.get("instruct"), "wav": os.path.basename(path),
                   "audio_s": round(len(audio) / 24000, 3), "wall_s": round(wall, 3),
                   "frames": [i.frames for i in infos], "ended_by_eos": [i.ended_by_eos for i in infos],
                   "backend_mode": [i.extra.get("mode") for i in infos],
                   "peak_gb": max((i.extra.get("peak_gb", 0) for i in infos), default=None)}
            manifest["runs"].append(row)
            print(f"  {r['name']:<40} {row['audio_s']:6.2f} s audio  {wall:6.2f} s wall  frames {row['frames']}  "
                  f"eos {all(row['ended_by_eos'])}  peak {row['peak_gb']} GB")
    finally:
        tts.close()
        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
    print(f"wrote {len(manifest['runs'])} wavs and manifest.json to {out}")


if __name__ == "__main__":
    main()
