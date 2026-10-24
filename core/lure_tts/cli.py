"""`lure` command line.

  lure say "Hello there. (laugh) Really?" --voice willow -o out.wav [--instruct "Warm."] [--seed 1] [--play]
  lure say - --ref-wav ref.wav --ref-text "..." --save-voice willow -o out.wav     (text from stdin)
  lure say "Hi." --speaker vivian -o out.wav            (CustomVoice)
  lure say "Hi." --instruct "A deep, calm voice." -o out.wav     (VoiceDesign)
  lure info [MODEL] [--json]                             (what the checkpoint is: variant, speakers, layout, ...)
  lure voice add willow ref.wav "reference transcript" ; lure voice list
  lure serve --port 8880
  lure prompt "text" --direction "..." [--mode M] [--layout L] [--speaker S] [--json]   (prompt rows, no weights)
  lure backends
Common: --model DIR (or LURE_MODEL), --backend auto|mlx|cuda|cpp|fake, --quant, --cache DIR, --layout, --inline-tags.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from .defaults import Sampling
from .prompt import LAYOUTS, MODES

SAMPLING_FLAGS = ("temperature", "top_k", "top_p", "repetition_penalty", "subtalker_temperature", "subtalker_top_k",
                  "subtalker_top_p", "max_new_tokens")


class CliError(ValueError):
    pass


def _common(p):
    p.add_argument("--model", default=os.environ.get("LURE_MODEL"), help="model dir (HF layout); default $LURE_MODEL")
    p.add_argument("--backend", default="auto")
    p.add_argument("--quant", default=None)
    p.add_argument("--cache", default=None, help="cache dir; default $LURE_CACHE or ~/.cache/lure")
    p.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE", help="backend option, e.g. compile_cp=1")
    p.add_argument("--layout", default=None, choices=LAYOUTS, help="override the model's default prompt layout")
    p.add_argument("--inline-tags", dest="inline_tags", default=None, action=argparse.BooleanOptionalAction,
                   help="override whether parentheses are event tags")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lure", description="Lure TTS: runs Qwen3-TTS 12Hz and Lure-family checkpoints")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("say", help="synthesize text")
    s.add_argument("text", help="text, or - for stdin")
    s.add_argument("-o", "--out", default=None, help="output .wav (or - for raw PCM16 on stdout)")
    s.add_argument("--voice", default=None, help="cached voice id (Base models)")
    s.add_argument("--speaker", default=None, help="preset speaker (CustomVoice models)")
    s.add_argument("--ref-wav", default=None)
    s.add_argument("--ref-text", default=None)
    s.add_argument("--x-vector-only", action="store_true", help="clone from --ref-wav without a transcript")
    s.add_argument("--language", default=None, help="language name or auto; default from the model")
    s.add_argument("--save-voice", default=None, help="cache the ref voice under this id")
    s.add_argument("--instruct", "--direction", dest="instruct", default=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--chunk-s", type=float, default=None)
    s.add_argument("--play", action="store_true", help="play while streaming (needs sounddevice)")
    s.add_argument("--stats", action="store_true", help="print per-segment frames / eos / ttfa to stderr")
    s.add_argument("--temperature", type=float)
    s.add_argument("--top-k", type=int)
    s.add_argument("--top-p", type=float)
    s.add_argument("--repetition-penalty", type=float)
    s.add_argument("--subtalker-temperature", type=float)
    s.add_argument("--subtalker-top-k", type=int)
    s.add_argument("--subtalker-top-p", type=float)
    s.add_argument("--max-new-tokens", type=int)
    _common(s)

    v = sub.add_parser("voice", help="manage cached voices")
    vs = v.add_subparsers(dest="vcmd", required=True)
    va = vs.add_parser("add")
    va.add_argument("name")
    va.add_argument("wav")
    va.add_argument("ref_text")
    _common(va)
    vl = vs.add_parser("list")
    vl.add_argument("--cache", default=None)

    sv = sub.add_parser("serve", help="OpenAI-compatible server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8880)
    _common(sv)

    pr = sub.add_parser("prompt", help="print prompt ids / prefill layout (no weights)")
    pr.add_argument("text")
    pr.add_argument("--direction", "--instruct", dest="direction", default=None)
    pr.add_argument("--ref-text", default="Reference transcript.")
    pr.add_argument("--n-ref-frames", type=int, default=6)
    pr.add_argument("--language", default=None, help="default: the model's default language")
    pr.add_argument("--mode", default=None, choices=MODES, help="default: the model's first mode")
    pr.add_argument("--layout", default=None, choices=LAYOUTS, help="default: the model's layout")
    pr.add_argument("--speaker", default=None, help="preset speaker (custom_voice)")
    pr.add_argument("--json", action="store_true")
    pr.add_argument("--segment", action="store_true", help="run the segmenter first, one prompt per segment")
    pr.add_argument("--model", default=os.environ.get("LURE_MODEL"))

    inf = sub.add_parser("info", help="inspect a model folder (no weights loaded)")
    inf.add_argument("model", nargs="?", default=os.environ.get("LURE_MODEL"))
    inf.add_argument("--json", action="store_true")
    inf.add_argument("--layout", default=None, choices=LAYOUTS)
    inf.add_argument("--inline-tags", dest="inline_tags", default=None, action=argparse.BooleanOptionalAction)

    sub.add_parser("backends", help="list backends and what auto picks")
    return ap


@dataclass
class SayRequest:
    text: str
    out: Optional[str]
    voice: Optional[str]
    speaker: Optional[str]
    ref_wav: Optional[str]
    ref_text: Optional[str]
    save_voice: Optional[str]
    instruct: Optional[str]
    seed: Optional[int]
    sampling: Sampling
    chunk_s: Optional[float]
    play: bool
    stats: bool
    model: Optional[str]
    backend: str
    quant: Optional[str]
    cache: Optional[str]
    opts: dict = field(default_factory=dict)
    language: Optional[str] = None
    x_vector_only: bool = False
    layout: Optional[str] = None
    inline_tags: Optional[bool] = None
    sampling_overrides: dict = field(default_factory=dict)  # only the flags given; applied over the model's defaults


def parse_opts(items) -> dict:
    out = {}
    for it in items:
        if "=" not in it:
            raise CliError(f"--opt expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        low = v.lower()
        out[k] = True if low in ("1", "true", "yes", "on") else False if low in ("0", "false", "no", "off") else _num(v)
    return out


def _num(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def parse_say(ns, stdin=None) -> SayRequest:
    text = ns.text
    if text == "-":
        text = (stdin or sys.stdin).read()
    if not text or not text.strip():
        raise CliError("empty text")
    instruct = ns.instruct.strip() if ns.instruct and ns.instruct.strip() else None
    ref = bool(ns.ref_wav or ns.ref_text)
    if sum(map(bool, (ns.voice, ns.speaker, ref))) > 1:
        raise CliError("use one of --voice, --speaker or --ref-wav/--ref-text")
    if ref and not ns.ref_wav:
        raise CliError("--ref-text needs --ref-wav")
    if ns.ref_wav and not ns.ref_text and not ns.x_vector_only:
        raise CliError("--ref-wav needs --ref-text (or --x-vector-only to clone without a transcript)")
    if not (ns.voice or ns.speaker or ref or instruct):
        raise CliError("need --voice ID, --ref-wav WAV --ref-text TEXT, --speaker NAME (CustomVoice) "
                       "or --instruct TEXT (VoiceDesign)")
    if ns.save_voice and not ns.ref_wav:
        raise CliError("--save-voice needs --ref-wav/--ref-text")
    if not ns.out and not ns.play:
        raise CliError("need -o OUT.wav (or - for PCM16 stdout) or --play")
    samp = {k: getattr(ns, k) for k in SAMPLING_FLAGS if getattr(ns, k, None) is not None}
    return SayRequest(text, ns.out, ns.voice, ns.speaker, ns.ref_wav, ns.ref_text, ns.save_voice, instruct, ns.seed,
                      Sampling.from_dict(samp), ns.chunk_s, ns.play, ns.stats, ns.model, ns.backend, ns.quant, ns.cache,
                      parse_opts(ns.opt), ns.language, ns.x_vector_only, ns.layout, ns.inline_tags, samp)


def _load(ns_model, backend, quant, cache, opts, layout=None, inline_tags=None):
    from .api import Lure
    return Lure.load(ns_model, backend=backend, quant=quant, cache_dir=cache, layout=layout, inline_tags=inline_tags,
                       **opts)


def cmd_say(r: SayRequest) -> int:
    import numpy as np

    from .audio import to_pcm16, write_wav
    tts = _load(r.model, r.backend, r.quant, r.cache, r.opts, r.layout, r.inline_tags)
    try:
        voice = r.voice
        if r.ref_wav:
            voice = tts.make_voice(r.ref_wav, None if r.x_vector_only else r.ref_text, name=r.save_voice)
        player = None
        if r.play:
            import sounddevice as sd
            player = sd.OutputStream(samplerate=tts.sample_rate, channels=1, dtype="float32")
            player.start()
        parts = []
        sampling = tts.spec.sampling.with_(**r.sampling_overrides)
        for pcm in tts.stream(r.text, voice, instruct=r.instruct, sampling=sampling, seed=r.seed, chunk_s=r.chunk_s,
                              speaker=r.speaker, language=r.language, x_vector_only=r.x_vector_only):
            if player:
                player.write(pcm.reshape(-1, 1))
            if r.out == "-":
                sys.stdout.buffer.write(to_pcm16(pcm))
                sys.stdout.buffer.flush()
            else:
                parts.append(pcm)
        if player:
            player.stop()
        if r.out and r.out != "-":
            write_wav(r.out, np.concatenate(parts) if parts else np.zeros(0, np.float32))
        if r.stats:
            for i in tts.last_infos:
                print(json.dumps({k: v for k, v in i.__dict__.items() if k != "extra"}), file=sys.stderr)
    finally:
        tts.close()
    return 0


def cmd_prompt(ns) -> int:
    from .model import inspect
    from .prompt import build_prompt, build_prompt_ids
    from .text import segment_text
    from .tokenizer import QwenTokenizer
    if not ns.model:
        raise CliError("need --model DIR or LURE_MODEL (only tokenizer files and config.json are read)")
    spec = inspect(ns.model, layout=ns.layout)
    tok = QwenTokenizer.from_dir(ns.model)
    mode = ns.mode or spec.modes[0]
    language = ns.language or spec.default_language
    texts = [s.text for s in segment_text(ns.text, tags=spec.inline_tags)] if ns.segment else [ns.text]
    out = []
    for t in texts:
        p = build_prompt_ids(tok, t, ns.ref_text if mode == "icl" else None, ns.direction)
        try:
            pr = build_prompt(p, spec.ids, mode=mode, layout=spec.layout, language=language,
                              n_ref_frames=ns.n_ref_frames if mode == "icl" else 0, speaker=ns.speaker)
        except ValueError as e:
            raise CliError(str(e))
        out.append({"text": t, "mode": mode, "layout_mode": spec.layout, "language": language, **p.to_dict(),
                    "layout": pr.prefill, "trailing": pr.trailing, "spans": pr.spans})
    if ns.json:
        print(json.dumps(out if ns.segment else out[0], ensure_ascii=False))
    else:
        for o in out:
            print(f"text: {o['text']!r}  ({o['mode']}, {o['layout_mode']}, {o['language']})\n"
                  f"  instruct_ids: {o['instruct_ids']}\n  input_ids: {o['input_ids']}\n  ref_ids: {o['ref_ids']}\n"
                  f"  spans: {o['spans']}\n  trailing: {o['trailing']}")
    return 0


def cmd_info(ns) -> int:
    from .model import describe, inspect
    if not ns.model:
        raise CliError("need MODEL or LURE_MODEL")
    try:
        spec = inspect(ns.model, layout=ns.layout, inline_tags=ns.inline_tags)
    except (OSError, ValueError) as e:
        raise CliError(str(e))
    print(json.dumps(spec.to_dict(), indent=1) if ns.json else describe(spec))
    return 0


def main(argv=None) -> int:
    ap = build_parser()
    ns = ap.parse_args(argv)
    try:
        if ns.cmd == "say":
            return cmd_say(parse_say(ns))
        if ns.cmd == "prompt":
            return cmd_prompt(ns)
        if ns.cmd == "info":
            return cmd_info(ns)
        if ns.cmd == "backends":
            from .backends import available, select
            for k, v in available().items():
                print(f"{k:6s} {v}")
            try:
                print("auto ->", select("auto"))
            except RuntimeError as e:
                print("auto -> none:", e)
            return 0
        if ns.cmd == "voice":
            if ns.vcmd == "list":
                from .voice import VoiceCache
                print("\n".join(VoiceCache(ns.cache).list()))
                return 0
            tts = _load(ns.model, ns.backend, ns.quant, ns.cache, parse_opts(ns.opt), ns.layout, ns.inline_tags)
            try:
                v = tts.make_voice(ns.wav, ns.ref_text, name=ns.name)
                print(f"{v.voice_id}: {v.n_ref_frames} ref frames -> {tts.voices.path(v.voice_id)}")
            finally:
                tts.close()
            return 0
        if ns.cmd == "serve":
            from .server import serve
            tts = _load(ns.model, ns.backend, ns.quant, ns.cache, parse_opts(ns.opt), ns.layout, ns.inline_tags)
            serve(tts, ns.host, ns.port)
            return 0
    except CliError as e:
        ap.error(str(e))
    return 2


if __name__ == "__main__":
    sys.exit(main())
