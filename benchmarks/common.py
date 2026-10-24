"""Shared setup for the benchmark scripts: CLI flags, model load, voice, result output."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "conformance"))

import cases as C  # noqa: E402


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", default=os.environ.get("LURE_MODEL"), help="model dir; default $LURE_MODEL")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE", help="backend option, e.g. turbo=0")
    ap.add_argument("--ref-wav", default=C.REF_WAV, help="voice prompt wav (default: the conformance voice)")
    ap.add_argument("--ref-text", default=C.REF_TEXT)
    ap.add_argument("--out", default=None, help="write the result JSON here")
    return ap


def load(args):
    from lure_tts import Lure
    from lure_tts.cli import parse_opts

    t0 = time.perf_counter()
    tts = Lure.load(args.model, backend=args.backend, **parse_opts(args.opt))
    load_s = time.perf_counter() - t0
    voice = tts.make_voice(args.ref_wav, args.ref_text)
    env = {"backend": tts.backend_name, "opts": args.opt, "turbo": tts.turbo, "concurrent": tts.concurrent,
           "load_s": round(load_s, 2), "platform": platform.platform(), "machine": platform.machine(), "device": device_name()}
    return tts, voice, env


def device_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
    except ImportError:
        pass
    return None


def stats(xs) -> dict | None:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"median": round(float(np.median(xs)), 4), "p90": round(float(np.percentile(xs, 90)), 4),
            "min": round(float(min(xs)), 4), "n": len(xs)}


def emit(args, result: dict):
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
