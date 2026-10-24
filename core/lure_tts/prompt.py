"""Prompt spec: the exact talker prefill of qwen-tts 0.1.1 for every mode and layout, in pure Python.

Source of truth: qwen_tts/core/models/modeling_qwen3_tts.py `Qwen3TTSForConditionalGeneration.generate` and
`generate_icl_prompt`, reached through the qwen-tts wrappers `generate_voice_clone`, `generate_custom_voice` and
`generate_voice_design`. conformance/reference/ holds dumps produced by running that code with symbolic embeddings; the
tests check this module against them.

Every prefill position is the SUM of a text-side embedding and a codec-side embedding (either may be absent):

  text side   T(id)   = talker.text_projection(talker.text_embedding(id))          (text vocab, 151936)
  codec side  C(id)   = talker.codec_embedding(id)                                  (codec vocab, 3072)
              SPK     = speaker-encoder x-vector of the reference wav (enc_dim = talker hidden size, cast to the talker dtype)
              REF(t)  = C(ref_codes[t,0]) + sum_{g=1..15} code_predictor.codec_embedding[g-1](ref_codes[t,g])

A row is [text, codec] with text = token id or None and codec = int id | "spk" | "ref:<t>" | None. A preset speaker
(custom_voice) is the plain codec row C(spk_id), so it needs no special marker.

Modes:
  icl           clone from reference codes + reference transcript + x-vector (Base)
  xvec          clone from the x-vector only, no reference codes or transcript (Base)
  custom_voice  preset speaker C(spk_id), optional instruct (CustomVoice)
  voice_design  no speaker row, the instruct describes the voice (VoiceDesign)

Every prefill starts the same way:
  1. instruct  : T(i) for every id of "<|im_start|>user\\n{instruct}<|im_end|>\\n"     (omitted when there is none)
  2. role      : T(i) for "<|im_start|>assistant\\n" = input_ids[:3]
  3. prefix    : T(tts_pad)+C(think), T(tts_pad)+C(think_bos), T(tts_pad)+C(lang), T(tts_pad)+C(think_eos),
                 [T(tts_pad)+speaker], T(tts_bos)+C(codec_pad)
                 language "auto" without a dialect speaker: C(nothink), C(think_bos), C(think_eos) (3 rows, no lang id).
                 speaker = SPK (icl, xvec) | C(spk_id) (custom_voice) | absent (voice_design).
Then, per mode and layout (X = ref_ids[3:-2] ++ input_ids[3:-5] ++ [tts_eos], Y = input_ids[3:-5] ++ [tts_eos]):
  icl  non_streaming : T(x)+C(codec_pad) for x in X, then T(tts_pad)+C(codec_bos), then T(tts_pad)+REF(t) per frame.
                       trailing = [tts_pad]
  icl  streaming     : codec column K = [C(codec_bos), REF(0), ..., REF(T-1)]. Row i = T(X[i])+K[i] for i < len(K), with X
                       padded by tts_pad when shorter. trailing = X[len(K):] when X is longer than K, else [tts_pad].
  other non_streaming: T(y)+C(codec_pad) for y in Y, then T(tts_pad)+C(codec_bos).  trailing = [tts_pad]
  other streaming    : one row T(input_ids[3])+C(codec_bos).  trailing = input_ids[4:-5] ++ [tts_eos]
Generation: generated frame k (k = 0 for the frame sampled from the prefill) is fed back as
  T(trailing[k]) + C(cb0) + sum_{g=1..15} code_predictor.codec_embedding[g-1](cb_g), with T(tts_pad) once trailing runs out.
Position ids: plain 0..L-1 on all 3 mrope axes (single unpadded sequence).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from .tokenizer import QwenTokenizer

MODES = ("icl", "xvec", "custom_voice", "voice_design")
LAYOUTS = ("non_streaming", "streaming")
CLONE_MODES = ("icl", "xvec")


def assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def ref_text_template(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


def instruct_text(direction: str) -> str:
    return f"<|im_start|>user\n{direction}<|im_end|>\n"


def normalize_layout(layout: str) -> str:
    v = str(layout).strip().lower().replace("-", "_")
    if v not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    return v


@dataclass(frozen=True)
class ModelIds:
    tts_bos: int = 151672
    tts_eos: int = 151673
    tts_pad: int = 151671
    codec_bos: int = 2149
    codec_eos: int = 2150
    codec_pad: int = 2148
    codec_think: int = 2154
    codec_nothink: int = 2155
    codec_think_bos: int = 2156
    codec_think_eos: int = 2157
    num_code_groups: int = 16
    codec_vocab_size: int = 3072
    language_ids: dict = field(default_factory=lambda: {"english": 2050})
    speaker_ids: dict = field(default_factory=dict)  # talker_config.spk_id: lowercase name -> codec id
    speaker_dialects: dict = field(default_factory=dict)  # talker_config.spk_is_dialect: name -> dialect key | False

    @classmethod
    def from_config(cls, cfg) -> "ModelIds":
        if isinstance(cfg, str):
            path = os.path.join(cfg, "config.json") if os.path.isdir(cfg) else cfg
            with open(path) as f:
                cfg = json.load(f)
        tc = cfg["talker_config"]
        return cls(
            tts_bos=cfg["tts_bos_token_id"], tts_eos=cfg["tts_eos_token_id"], tts_pad=cfg["tts_pad_token_id"],
            codec_bos=tc["codec_bos_id"], codec_eos=tc["codec_eos_token_id"], codec_pad=tc["codec_pad_id"],
            codec_think=tc["codec_think_id"], codec_nothink=tc["codec_nothink_id"],
            codec_think_bos=tc["codec_think_bos_id"], codec_think_eos=tc["codec_think_eos_id"],
            num_code_groups=tc["num_code_groups"], codec_vocab_size=tc["vocab_size"],
            language_ids=dict(tc.get("codec_language_id") or {}),
            speaker_ids={k.lower(): _scalar(v) for k, v in (tc.get("spk_id") or {}).items()},
            speaker_dialects={k.lower(): v for k, v in (tc.get("spk_is_dialect") or {}).items()},
        )

    def suppress_tokens(self) -> list[int]:
        """Codebook-0 ids masked to -inf at every step: [vocab-1024, vocab) except EOS (qwen-tts talker_kwargs)."""
        return [i for i in range(self.codec_vocab_size - 1024, self.codec_vocab_size) if i != self.codec_eos]

    def speaker_id(self, speaker: str) -> int:
        """Codec id of a preset speaker (case-insensitive), the row C(spk_id) of the custom_voice prefix."""
        key = speaker.lower()
        if key not in self.speaker_ids:
            raise ValueError(f"unknown speaker {speaker!r}; this model has: {sorted(self.speaker_ids) or 'none'}")
        return self.speaker_ids[key]

    def language_id(self, language: str, speaker: Optional[str] = None) -> Optional[int]:
        """Codec language id, or None for 'auto'. A dialect speaker with language chinese or auto switches to its
        dialect id (qwen-tts generate: spk_is_dialect)."""
        lang = language.lower()
        lid = None
        if lang != "auto":
            if lang not in self.language_ids:
                raise ValueError(f"language {language!r} not supported: {sorted(self.language_ids)}")
            lid = self.language_ids[lang]
        dialect = self.speaker_dialects.get(speaker.lower()) if speaker else None
        if lang in ("chinese", "auto") and dialect:
            lid = self.language_ids[dialect]
        return lid


def _scalar(v) -> int:
    return int(v[0]) if isinstance(v, (list, tuple)) else int(v)


@dataclass(frozen=True)
class PromptIds:
    input_ids: list[int]  # tokenized assistant_text(text)
    ref_ids: Optional[list[int]]  # tokenized ref_text_template(ref_text); None outside icl
    instruct_ids: Optional[list[int]]  # tokenized instruct_text(direction) or None

    @property
    def role_ids(self) -> list[int]:
        return self.input_ids[:3]

    @property
    def text_ids(self) -> list[int]:
        return self.input_ids[3:-5]

    @property
    def ref_text_ids(self) -> list[int]:
        return self.ref_ids[3:-2] if self.ref_ids else []

    def to_dict(self) -> dict:
        return {"input_ids": self.input_ids, "ref_ids": self.ref_ids, "instruct_ids": self.instruct_ids}


@dataclass(frozen=True)
class Prompt:
    """A full talker prompt: prefill rows, the trailing text rows fed one per generated frame, and named spans."""
    prefill: list
    trailing: list  # rows [text_id, None]; past the end the per-frame text input is T(tts_pad)
    spans: dict
    mode: str
    layout: str

    def to_dict(self) -> dict:
        return {"mode": self.mode, "layout": self.layout, "layout_rows": self.prefill, "trailing": self.trailing,
                "spans": self.spans}


def build_prompt_ids(tok: QwenTokenizer, text: str, ref_text: Optional[str] = None,
                     direction: Optional[str] = None) -> PromptIds:
    """Text exactly as given (tags stay plain text). Empty / whitespace-only direction counts as no direction.
    ref_text None (non-icl modes) gives ref_ids None."""
    ins = tok.encode(instruct_text(direction)) if direction and direction.strip() else None
    ref = tok.encode(ref_text_template(ref_text)) if ref_text is not None else None
    return PromptIds(tok.encode(assistant_text(text)), ref, ins)


def build_prompt(p: PromptIds, mid: ModelIds, *, mode: str = "icl", layout: str = "non_streaming",
                 language: str = "english", n_ref_frames: int = 0, speaker: Optional[str] = None) -> Prompt:
    """The qwen-tts prompt for one segment (see module doc). speaker: preset name, custom_voice only."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    layout = normalize_layout(layout)
    if mode == "icl" and p.ref_ids is None:
        raise ValueError("icl mode needs ref_ids (the reference transcript)")
    if (mode == "custom_voice") != (speaker is not None):
        raise ValueError("speaker is required for custom_voice and not allowed in other modes")
    slot = {"icl": "spk", "xvec": "spk", "voice_design": None}.get(mode)
    if mode == "custom_voice":
        slot = mid.speaker_id(speaker)
    return _build(p, mid, mode, layout, mid.language_id(language, speaker), n_ref_frames, slot)


def _build(p: PromptIds, mid: ModelIds, mode: str, layout: str, lang_id: Optional[int], n_ref: int, slot) -> Prompt:
    rows: list = []
    spans: dict = {}

    def block(name: str, new: list):
        spans[name] = [len(rows), len(rows) + len(new)]
        rows.extend(new)

    block("instruct", [[i, None] for i in p.instruct_ids or []])
    block("role", [[i, None] for i in p.role_ids])
    think = ([mid.codec_nothink, mid.codec_think_bos, mid.codec_think_eos] if lang_id is None
             else [mid.codec_think, mid.codec_think_bos, lang_id, mid.codec_think_eos])
    codec_seq = think + ([slot] if slot is not None else []) + [mid.codec_pad]
    # text side: tts_pad on every prefix row except the last, which carries tts_bos (over codec_pad)
    block("prefix", [[mid.tts_pad, c] for c in codec_seq[:-1]] + [[mid.tts_bos, mid.codec_pad]])
    pad_tail = [[mid.tts_pad, None]]

    if mode == "icl":
        x = p.ref_text_ids + p.text_ids + [mid.tts_eos]
        k = [mid.codec_bos] + [f"ref:{t}" for t in range(n_ref)]
        if layout == "non_streaming":
            block("icl_text", [[i, mid.codec_pad] for i in x])
            block("icl_codec", [[mid.tts_pad, c] for c in k])
            trailing = pad_tail
        else:
            block("icl_overlay", [[x[i] if i < len(x) else mid.tts_pad, c] for i, c in enumerate(k)])
            trailing = [[i, None] for i in x[len(k):]] or pad_tail
    elif layout == "non_streaming":
        block("text", [[i, mid.codec_pad] for i in p.text_ids + [mid.tts_eos]])
        block("codec_bos", [[mid.tts_pad, mid.codec_bos]])
        trailing = pad_tail
    else:
        block("first_text", [[p.input_ids[3], mid.codec_bos]])
        trailing = [[i, None] for i in p.input_ids[4:-5] + [mid.tts_eos]]
    spans["total"] = len(rows)
    return Prompt(rows, trailing, spans, mode, layout)


def build_layout(p: PromptIds, mid: ModelIds, n_ref_frames: int, language: str = "english", speaker: bool = True,
                 **kw) -> list[list]:
    """Prefill rows only. Without keywords this is the icl non_streaming layout (the Lure default); pass mode=,
    layout=, speaker_name= for the other modes. speaker=False drops the x-vector row (not a qwen-tts mode)."""
    return _legacy(p, mid, n_ref_frames, language, speaker, **kw).prefill


def layout_spans(p: PromptIds, mid: ModelIds, n_ref_frames: int, language: str = "english", speaker: bool = True,
                 **kw) -> dict:
    """Named [start, end) spans of the prefill, handy for backends that build the embedding in blocks."""
    return _legacy(p, mid, n_ref_frames, language, speaker, **kw).spans


def _legacy(p, mid, n_ref_frames, language, speaker, mode="icl", layout="non_streaming", speaker_name=None) -> Prompt:
    if speaker or mode != "icl":
        return build_prompt(p, mid, mode=mode, layout=layout, language=language, n_ref_frames=n_ref_frames,
                            speaker=speaker_name)
    return _build(p, mid, "icl", normalize_layout(layout), mid.language_id(language), n_ref_frames, None)


def generation_recipe(mid: ModelIds) -> dict:
    """Per-frame input + sampling masks, as normative data for backends."""
    return {
        "next_input": "T(trailing[k] if k < len(trailing) else tts_pad) + C(cb0) + sum_{g=1..15} CP_EMB[g-1](cb_g)",
        "trailing_text_hidden": "Prompt.trailing (non_streaming: [tts_pad])",
        "eos": mid.codec_eos,
        "suppress_cb0": [mid.codec_vocab_size - 1024, mid.codec_vocab_size, "except", mid.codec_eos],
        "min_new_tokens": 2,
        "repetition_penalty_scope": "codebook 0 only, all previously generated cb0 ids of this segment (unique set), "
                                    "logit<0 ? logit*p : logit/p, applied before temperature/top-k/top-p",
        "subtalker": "codebooks 1-15 sampled with subtalker temperature/top_k/top_p, no repetition penalty",
        "decode": "icl: codec decoder on [ref_codes ; generated codes]; drop the first int(ref_len/total_len * n_samples) "
                  "samples (qwen-tts proportional cut). Streaming: prime decoder context with the tail of ref_codes. "
                  "Other modes: decode the generated codes alone, no reference context.",
    }
