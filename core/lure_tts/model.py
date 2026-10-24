"""Model inspection: what a Qwen3-TTS-12Hz checkpoint folder is and how to prompt it.

    from lure_tts.model import inspect
    spec = inspect("/path/to/model")        # reads config.json, generation_config.json, lure.json, turbo.json

Families:
  qwen3_tts  stock Qwen3-TTS 12Hz (0.6B / 1.7B; Base, CustomVoice, VoiceDesign). Defaults match qwen-tts 0.1.1:
             Base clones with the streaming layout (generate_voice_clone non_streaming_mode=False), CustomVoice and
             VoiceDesign use non_streaming (non_streaming_mode=True). Parentheses are plain text.
  lure       Lure, Lure Turbo. Marked by lure.json, or by turbo.json alone. Base layout, expects the
             non_streaming layout, takes an instruct (direction) and performs inline (tag) events.

lure.json: {"format": "lure-model-v1", "family": "lure", "prompt_layout": "non_streaming", "inline_tags": true}
(optional "name", e.g. "Lure").
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from . import turbo as turbo_mod
from .defaults import LANGUAGE, Sampling
from .prompt import CLONE_MODES, ModelIds, normalize_layout

LURE_JSON = "lure.json"
LURE_FORMAT = "lure-model-v1"
VARIANTS = ("base", "custom_voice", "voice_design")
TOKENIZER_12HZ = "qwen3_tts_tokenizer_12hz"
STOCK_LAYOUT = {"base": "streaming", "custom_voice": "non_streaming", "voice_design": "non_streaming"}
VARIANT_MODES = {"base": CLONE_MODES, "custom_voice": ("custom_voice",), "voice_design": ("voice_design",)}
VARIANT_NAMES = {"base": "Base", "custom_voice": "CustomVoice", "voice_design": "VoiceDesign"}
SIZE_NAMES = {"0b6": "0.6B", "1b7": "1.7B"}
# generation_config.json key -> Sampling field (qwen-tts _merge_generate_kwargs)
GEN_KEYS = {"do_sample": "do_sample", "top_k": "top_k", "top_p": "top_p", "temperature": "temperature",
            "repetition_penalty": "repetition_penalty", "subtalker_dosample": "subtalker_do_sample",
            "subtalker_top_k": "subtalker_top_k", "subtalker_top_p": "subtalker_top_p",
            "subtalker_temperature": "subtalker_temperature"}


@dataclass(frozen=True)
class ModelSpec:
    model_dir: str
    name: str  # display name, e.g. "Qwen3-TTS-12Hz-1.7B-CustomVoice" or "Lure Turbo"
    family: str  # "qwen3_tts" | "lure"
    variant: str  # "base" | "custom_voice" | "voice_design" (config tts_model_type)
    size: str  # config tts_model_size verbatim: "0b6" | "1b7"
    talker_hidden_size: int  # 1024 (0.6B) | 2048 (1.7B)
    code_predictor_hidden_size: int  # 1024 for both sizes
    speaker_encoder_dim: Optional[int]  # Base only: x-vector size (= talker_hidden_size); None without a speaker encoder
    num_code_groups: int
    speakers: tuple  # preset speaker names, lowercase, sorted (CustomVoice)
    speaker_dialects: dict  # name -> dialect language key, for dialect speakers only
    languages: tuple  # what `language=` accepts: "auto" plus codec_language_id keys without "dialect"
    default_language: str  # "auto" for stock (qwen-tts default), "english" for the Lure family
    modes: tuple  # prompt modes this variant runs (lure_tts.prompt.MODES)
    layout: str  # default prompt layout: "non_streaming" | "streaming"
    inline_tags: bool  # parentheses are event tags (segmenter keeps them whole) vs plain text
    instruct: str  # "none" | "optional" | "required"
    sampling: Sampling
    max_new_tokens: int  # generation_config.json cap (the core still passes a per-segment cap below it)
    turbo: Optional[turbo_mod.TurboConfig] = None
    ids: ModelIds = field(default_factory=ModelIds, repr=False)
    lure_json: dict = field(default_factory=dict, repr=False)

    @property
    def size_label(self) -> str:
        return SIZE_NAMES.get(self.size, self.size)

    @property
    def is_clone(self) -> bool:
        return self.variant == "base"

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("name", "family", "variant", "size", "talker_hidden_size",
                                           "code_predictor_hidden_size", "speaker_encoder_dim", "num_code_groups",
                                           "default_language", "layout", "inline_tags", "instruct", "max_new_tokens")}
        d.update(model_dir=self.model_dir, size_label=self.size_label, speakers=list(self.speakers),
                 speaker_dialects=dict(self.speaker_dialects), languages=list(self.languages), modes=list(self.modes),
                 sampling=self.sampling.to_dict(), turbo=self._turbo_dict())
        return d

    def _turbo_dict(self) -> Optional[dict]:
        if self.turbo is None:
            return None
        t = asdict(self.turbo)
        t.pop("raw", None)
        return t


def _load_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_lure_json(model_dir: str) -> Optional[dict]:
    raw = _load_json(os.path.join(model_dir, LURE_JSON))
    if raw is None:
        return None
    where = os.path.join(model_dir, LURE_JSON)
    if raw.get("format") != LURE_FORMAT:
        raise ValueError(f"{where}: format must be {LURE_FORMAT!r}, got {raw.get('format')!r}")
    if raw.get("family", "lure") != "lure":
        raise ValueError(f"{where}: family must be 'lure', got {raw.get('family')!r}")
    if "inline_tags" in raw and not isinstance(raw["inline_tags"], bool):
        raise ValueError(f"{where}: inline_tags must be true or false")
    normalize_layout(raw.get("prompt_layout", "non_streaming"))
    return raw


def sampling_from_generation_config(gen: Optional[dict]) -> Sampling:
    """generation_config.json over the qwen-tts hard defaults (which equal defaults.Sampling)."""
    gen = gen or {}
    return Sampling().with_(**{f: gen[k] for k, f in GEN_KEYS.items() if k in gen})


def inspect(model_dir: str, layout: Optional[str] = None, inline_tags: Optional[bool] = None, turbo=None,
            turbo_head: Optional[str] = None) -> ModelSpec:
    """Read a model folder. layout / inline_tags override the family defaults; turbo / turbo_head follow
    lure_tts.turbo.resolve."""
    cfg = _load_json(os.path.join(model_dir, "config.json"))
    if cfg is None:
        raise FileNotFoundError(f"no config.json in {model_dir!r}")
    tc = cfg.get("talker_config") or {}
    variant = cfg.get("tts_model_type")
    if variant not in VARIANTS:
        raise ValueError(f"{model_dir}: tts_model_type must be one of {VARIANTS}, got {variant!r}")
    if cfg.get("tokenizer_type", TOKENIZER_12HZ) != TOKENIZER_12HZ:
        raise ValueError(f"{model_dir}: only the 12Hz speech tokenizer is supported, got {cfg.get('tokenizer_type')!r}")
    ids = ModelIds.from_config(cfg)
    if ids.num_code_groups != 16:
        raise ValueError(f"{model_dir}: expected 16 code groups, got {ids.num_code_groups}")
    cp = tc.get("code_predictor_config") or {}
    enc = cfg.get("speaker_encoder_config") or {}

    aj = read_lure_json(model_dir)
    tcfg = turbo_mod.resolve(model_dir, turbo=turbo, turbo_head=turbo_head)
    family = "lure" if aj is not None or turbo_mod.is_turbo_dir(model_dir) else "qwen3_tts"
    aj = aj or {}
    size = str(cfg.get("tts_model_size", ""))
    if family == "lure":
        if variant != "base":
            raise ValueError(f"{model_dir}: Lure-family models use the Base layout, config says {variant!r}")
        default_layout, default_tags = normalize_layout(aj.get("prompt_layout", "non_streaming")), aj.get("inline_tags", True)
        name = aj.get("name") or ("Lure Turbo" if tcfg is not None else "Lure")
        default_language, instruct = LANGUAGE, "optional"
    else:
        default_layout, default_tags = STOCK_LAYOUT[variant], False
        name = f"Qwen3-TTS-12Hz-{SIZE_NAMES.get(size, size)}-{VARIANT_NAMES[variant]}"
        default_language = "auto"
        # qwen-tts: generate_voice_clone takes no instruct, generate_custom_voice drops it on 0.6B
        # (`tts_model_size in "0b6"`), generate_voice_design is driven by it
        instruct = {"base": "none", "voice_design": "required"}.get(variant)
        if instruct is None:
            instruct = "none" if size == "0b6" else "optional"

    gen = _load_json(os.path.join(model_dir, "generation_config.json"))
    langs = ("auto",) + tuple(k for k in ids.language_ids if "dialect" not in k)
    return ModelSpec(
        model_dir=model_dir, name=name, family=family, variant=variant, size=size,
        talker_hidden_size=int(tc.get("hidden_size", 1024)),
        code_predictor_hidden_size=int(cp.get("hidden_size", tc.get("hidden_size", 1024))),
        speaker_encoder_dim=int(enc.get("enc_dim", 1024)) if variant == "base" else None,
        num_code_groups=ids.num_code_groups,
        speakers=tuple(sorted(ids.speaker_ids)),
        speaker_dialects={k: v for k, v in ids.speaker_dialects.items() if v},
        languages=langs, default_language=default_language, modes=VARIANT_MODES[variant],
        layout=normalize_layout(layout) if layout else default_layout,
        inline_tags=default_tags if inline_tags is None else bool(inline_tags),
        instruct=instruct, sampling=sampling_from_generation_config(gen),
        max_new_tokens=int((gen or {}).get("max_new_tokens", 2048)),
        turbo=tcfg, ids=ids, lure_json=aj,
    )


def describe(spec: ModelSpec) -> str:
    """Human-readable summary for `lure info`."""
    s = spec
    lines = [
        f"{s.name}  ({s.model_dir})",
        f"  family       {s.family}",
        f"  variant      {s.variant} ({s.size_label})",
        f"  hidden       talker {s.talker_hidden_size}, code predictor {s.code_predictor_hidden_size}"
        + (f", speaker encoder {s.speaker_encoder_dim}" if s.speaker_encoder_dim else ""),
        f"  modes        {', '.join(s.modes)}",
        f"  layout       {s.layout}",
        f"  inline tags  {'yes' if s.inline_tags else 'no (parentheses are plain text)'}",
        f"  instruct     {s.instruct}",
        f"  languages    {', '.join(s.languages)} (default {s.default_language})",
    ]
    if s.speakers:
        dial = {k: v for k, v in s.speaker_dialects.items()}
        lines.append("  speakers     " + ", ".join(f"{k} ({dial[k]})" if k in dial else k for k in s.speakers))
    sm = s.sampling
    lines.append(f"  sampling     temperature {sm.temperature}, top_k {sm.top_k}, top_p {sm.top_p}, "
                 f"repetition_penalty {sm.repetition_penalty}; subtalker {sm.subtalker_temperature}/"
                 f"{sm.subtalker_top_k}/{sm.subtalker_top_p}")
    lines.append(f"  turbo        {'on, ' + str(s.turbo.n_passes) + ' passes (' + s.turbo.head_path + ')' if s.turbo else 'off'}")
    return "\n".join(lines)
