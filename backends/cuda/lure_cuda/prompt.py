"""Talker prefill for the CUDA backend, every prompt mode and layout.

`build_prefill` is a single-request copy of the prompt code in qwen-tts 0.1.1 `Qwen3TTSForConditionalGeneration.generate`,
with the SAME op grouping (so bf16 numerics match upstream) and the model's own `generate_icl_prompt`. In the notation
of the core's lure_tts.prompt, the rows are:

  [instruct T(i)...] [role T(i) x3]
  [T(pad)+C(think), T(pad)+C(think_bos), T(pad)+C(lang), T(pad)+C(think_eos), T(pad)+speaker, T(bos)+C(codec_pad)]
  then, per mode and layout:
    icl  non_streaming  T(x)+C(codec_pad) for x in ref text ++ text ++ [eos], T(pad)+C(codec_bos), T(pad)+REF(t)...
    icl  streaming      T(X[i])+K[i] with K = [C(codec_bos), REF(0)...], the rest of X is the trailing text
    other non_streaming T(y)+C(codec_pad) for y in text ++ [eos], T(pad)+C(codec_bos)
    other streaming     T(input_ids[3])+C(codec_bos), trailing text input_ids[4:-5] ++ [eos]

  speaker = SPK x-vector (icl, xvec), C(spk_id) (custom_voice), absent (voice_design)

Checked row for row against conformance/reference/prompt_ids.json and prompt_modes.json (tests/test_prompt_conformance.py
and tests/test_modes_cpu.py) with `symbolic_model`, a stub whose embeddings decode back to exact [text, codec] pairs.

The backend does not call the upstream faster-qwen3-tts voice clone helpers: their defaults append 0.5 s of silence
to the reference before encoding, which changes the reference codes and x-vector.
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from typing import List, Optional

import torch

MODES = ("icl", "xvec", "custom_voice", "voice_design")
LAYOUTS = ("non_streaming", "streaming")
VARIANT_MODES = {"base": ("icl", "xvec"), "custom_voice": ("custom_voice",), "voice_design": ("voice_design",)}


def assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def ref_text_template(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


def instruct_text(direction: str) -> str:
    return f"<|im_start|>user\n{direction}<|im_end|>\n"


def has_direction(direction: Optional[str]) -> bool:
    """Empty / whitespace-only direction == no direction (core rule)."""
    return bool(direction and direction.strip())


def normalize_layout(layout: str) -> str:
    v = str(layout).strip().lower().replace("-", "_")
    if v not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    return v


@dataclass(frozen=True)
class PromptIds:
    input_ids: List[int]
    ref_ids: Optional[List[int]]  # None outside icl
    instruct_ids: Optional[List[int]]

    def to_dict(self) -> dict:
        return {"input_ids": list(self.input_ids), "ref_ids": _opt_list(self.ref_ids),
                "instruct_ids": _opt_list(self.instruct_ids)}

    @classmethod
    def coerce(cls, p) -> "PromptIds":
        """Accept our PromptIds, the core's lure_tts.prompt.PromptIds, or a dict with the same keys."""
        if isinstance(p, cls):
            return p
        if isinstance(p, dict):
            return cls(list(p["input_ids"]), _opt_list(p.get("ref_ids")), _opt_list(p.get("instruct_ids")))
        return cls(list(p.input_ids), _opt_list(p.ref_ids), _opt_list(p.instruct_ids))


def _opt_list(x) -> Optional[List[int]]:
    return None if x is None else [int(i) for i in x]


def tokenize(processor, text: str) -> List[int]:
    """qwen-tts `_tokenize_texts` for one string (processor(text=..., padding=True), no BOS/EOS)."""
    ids = processor(text=text, return_tensors="pt", padding=True)["input_ids"]
    if ids.dim() == 2:
        ids = ids[0]
    return [int(i) for i in ids.tolist()]


def build_prompt_ids(processor, text: str, ref_text: Optional[str], direction: Optional[str] = None) -> PromptIds:
    """ref_text None (every mode except icl) gives ref_ids None."""
    ins = tokenize(processor, instruct_text(direction)) if has_direction(direction) else None
    ref = tokenize(processor, ref_text_template(ref_text)) if ref_text is not None else None
    return PromptIds(tokenize(processor, assistant_text(text)), ref, ins)


def _ids_tensor(ids: List[int], device) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long, device=device)


def language_id(tcfg, language: str, speaker: Optional[str] = None) -> Optional[int]:
    """Codec language id (None for auto). A dialect speaker with chinese or auto uses its dialect id (qwen-tts)."""
    lang = language.lower()
    lid = None
    if lang != "auto":
        if lang not in tcfg.codec_language_id:
            raise NotImplementedError(f"Language {language} not implemented")
        lid = tcfg.codec_language_id[lang]
    dialect = (getattr(tcfg, "spk_is_dialect", None) or {}).get(speaker.lower(), False) if speaker else False
    if lang in ("chinese", "auto") and dialect:
        lid = tcfg.codec_language_id[dialect]
    return lid


def speaker_codec_id(tcfg, speaker: Optional[str], speaker_id: Optional[int]) -> int:
    """Codec id of a preset speaker. speaker_id (from the core) must agree with the config when both are given."""
    table = {k.lower(): v for k, v in (getattr(tcfg, "spk_id", None) or {}).items()}
    if speaker is not None:
        if speaker.lower() not in table:
            raise ValueError(f"unknown speaker {speaker!r}; this model has: {sorted(table) or 'none'}")
        sid = table[speaker.lower()]
        sid = int(sid[0]) if isinstance(sid, (list, tuple)) else int(sid)
        if speaker_id is not None and int(speaker_id) != sid:
            raise ValueError(f"speaker_id {speaker_id} does not match {speaker!r} (codec id {sid})")
        return sid
    if speaker_id is None:
        raise ValueError("custom_voice needs a speaker")
    return int(speaker_id)


@torch.inference_mode()
def build_prefill(m, prompt: PromptIds, ref_code: Optional[torch.Tensor] = None,
                  spk_embedding: Optional[torch.Tensor] = None, language: str = "english", *, mode: str = "icl",
                  layout: str = "non_streaming", speaker: Optional[str] = None, speaker_id: Optional[int] = None):
    """Return (talker_input_embeds [1, L, H], trailing_text_hidden [1, N, H], tts_pad_embed [1, 1, H]).

    m: qwen-tts Qwen3TTSForConditionalGeneration (or a stub exposing the same attributes).
    ref_code: [T_ref, 16] long reference codes (icl). spk_embedding: [H] x-vector (icl, xvec).
    speaker / speaker_id: preset speaker name and codec id (custom_voice; one is enough).
    trailing_text_hidden row k is the text side of generated frame k; past its end the text side is tts_pad.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    streaming = normalize_layout(layout) == "streaming"
    talker = m.talker
    cfg = m.config
    tcfg = cfg.talker_config
    dev = talker.device
    T = lambda ids: talker.text_projection(talker.get_text_embeddings()(ids))  # noqa: E731
    C = talker.get_input_embeddings()
    input_id = _ids_tensor(prompt.input_ids, dev)

    parts = []
    if prompt.instruct_ids:
        parts.append(T(_ids_tensor(prompt.instruct_ids, dev)))

    if mode in ("icl", "xvec"):
        if spk_embedding is None:
            raise ValueError(f"{mode} needs the speaker x-vector")
        if spk_embedding.shape[-1] != tcfg.hidden_size:
            raise ValueError(f"x-vector has {spk_embedding.shape[-1]} dims, this talker expects {tcfg.hidden_size}; "
                             "encode the voice with this model")
        speaker_embed = spk_embedding.to(dev).to(talker.dtype)  # generate_speaker_prompt
        lang_id = language_id(tcfg, language)
    elif mode == "custom_voice":
        sid = speaker_codec_id(tcfg, speaker, speaker_id)
        speaker_embed = C(torch.tensor(sid, device=dev, dtype=input_id.dtype))
        lang_id = language_id(tcfg, language, speaker)
    else:
        speaker_embed = None
        lang_id = language_id(tcfg, language)

    tts_bos_embed, tts_eos_embed, tts_pad_embed = T(
        torch.tensor([[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]], device=dev,
                     dtype=input_id.dtype)).chunk(3, dim=1)

    if lang_id is None:
        codec_prefill_list = [[tcfg.codec_nothink_id, tcfg.codec_think_bos_id, tcfg.codec_think_eos_id]]
    else:
        codec_prefill_list = [[tcfg.codec_think_id, tcfg.codec_think_bos_id, lang_id, tcfg.codec_think_eos_id]]
    codec_0 = C(torch.tensor(codec_prefill_list, device=dev, dtype=input_id.dtype))
    codec_1 = C(torch.tensor([[tcfg.codec_pad_id, tcfg.codec_bos_id]], device=dev, dtype=input_id.dtype))
    if speaker_embed is None:
        codec = torch.cat([codec_0, codec_1], dim=1)
    else:
        codec = torch.cat([codec_0, speaker_embed.view(1, 1, -1), codec_1], dim=1)

    role = T(input_id[:, :3])
    body = torch.cat((tts_pad_embed.expand(-1, codec.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec[:, :-1]
    embed = torch.cat((role, body), dim=1)

    if mode == "icl":
        if prompt.ref_ids is None or ref_code is None:
            raise ValueError("icl needs the reference transcript ids and reference codes")
        icl, trailing = m.generate_icl_prompt(
            text_id=input_id[:, 3:-5],
            ref_id=_ids_tensor(prompt.ref_ids, dev)[:, 3:-2],
            ref_code=ref_code.to(dev).clone(),
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=not streaming,
        )
        embed = torch.cat([embed, icl], dim=1)
    elif not streaming:
        text = input_id[:, 3:-5]
        embed = torch.cat([
            embed,
            torch.cat((T(text), tts_eos_embed), dim=1)
            + C(torch.tensor([[tcfg.codec_pad_id] * (text.shape[1] + 1)], device=dev, dtype=input_id.dtype)),
            tts_pad_embed + C(torch.tensor([[tcfg.codec_bos_id]], device=dev, dtype=input_id.dtype)),
        ], dim=1)
        trailing = tts_pad_embed
    else:
        embed = torch.cat([embed, T(input_id[:, 3:4]) + codec[:, -1:]], dim=1)
        trailing = torch.cat((T(input_id[:, 4:-5]), tts_eos_embed), dim=1)
    parts.append(embed)
    return torch.cat(parts, dim=1), trailing, tts_pad_embed


def layout_from_core(prompt: PromptIds, model_dir_or_cfg, n_ref_frames: int, language: str = "english",
                     mode: str = "icl", layout: str = "non_streaming", speaker: Optional[str] = None):
    """The core's normative rows (lure_tts.prompt.build_prompt) as (prefill, trailing)."""
    from lure_tts.prompt import ModelIds, PromptIds as CorePromptIds, build_prompt

    mid = ModelIds.from_config(model_dir_or_cfg)
    pr = build_prompt(CorePromptIds(prompt.input_ids, prompt.ref_ids, prompt.instruct_ids), mid, mode=mode,
                      layout=layout, language=language, n_ref_frames=n_ref_frames, speaker=speaker)
    return pr.prefill, pr.trailing


# ------------------------------------------------------------------------------------------------ symbolic rows
# A stub model whose embeddings are one-hot style markers: column 0 = text id + 1, column 1 = codec id + 1,
# columns 2..16 = code predictor table g codes + 1, column 17 = the x-vector. Every prefill row then decodes back to
# the exact [text, codec] pair of the core's notation (dump_layout and the conformance tests use this).
SYM_D = 18
SYM_SPK = 17


def symbolic_model(config):
    """Stub with the attributes build_prefill reads, using the real `generate_icl_prompt`."""
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration

    dt = torch.float64

    def column(col):
        def emb(ids):
            out = torch.zeros(*ids.shape, SYM_D, dtype=dt)
            out[..., col] = ids.to(dt) + 1
            return out
        return emb

    talker = types.SimpleNamespace(
        device=torch.device("cpu"), dtype=dt,
        config=types.SimpleNamespace(num_code_groups=config.talker_config.num_code_groups),
        get_text_embeddings=lambda: column(0), text_projection=lambda x: x, get_input_embeddings=lambda: column(1),
        code_predictor=types.SimpleNamespace(get_input_embeddings=lambda: [column(1 + g) for g in range(1, 16)]),
    )
    # the x-vector check reads hidden_size: give the stub's talker config the marker width
    tcfg = types.SimpleNamespace(**{**vars(config.talker_config), "hidden_size": SYM_D})
    cfg = types.SimpleNamespace(**{**vars(config), "talker_config": tcfg})
    m = types.SimpleNamespace(config=cfg, talker=talker, device=torch.device("cpu"))
    m.generate_icl_prompt = types.MethodType(Qwen3TTSForConditionalGeneration.generate_icl_prompt, m)
    return m


def symbolic_voice(n_ref: int):
    """(ref_codes [n_ref, 16], x-vector marker) for symbolic_model. Codes are distinct per frame."""
    t = torch.arange(n_ref).view(-1, 1)
    g = torch.arange(16).view(1, -1)
    spk = torch.zeros(SYM_D, dtype=torch.float64)
    spk[SYM_SPK] = 1
    return ((37 * t + 101 * g + 7) % 2048).long(), spk


def decode_symbolic_row(v, ref_codes) -> list:
    v = [int(round(float(x))) for x in v]
    text = v[0] - 1 if v[0] else None
    cps = v[2:17]
    if any(cps):
        frame = [v[1] - 1] + [c - 1 for c in cps]
        hits = [t for t in range(ref_codes.shape[0]) if ref_codes[t].tolist() == frame]
        if len(hits) != 1:
            raise AssertionError(f"row is not one reference frame: {frame}")
        return [text, f"ref:{hits[0]}"]
    if v[SYM_SPK]:
        return [text, "spk"]
    return [text, v[1] - 1 if v[1] else None]


def symbolic_rows(config, prompt: PromptIds, n_ref_frames: int, language: str = "english", mode: str = "icl",
                  layout: str = "non_streaming", speaker: Optional[str] = None) -> dict:
    """build_prefill on the symbolic stub -> {"layout": prefill rows, "trailing": trailing rows, "tts_pad": row}."""
    stub = symbolic_model(config)
    codes, spk = symbolic_voice(n_ref_frames)
    clone = mode in ("icl", "xvec")
    emb, trailing, pad = build_prefill(stub, prompt, codes if mode == "icl" else None, spk if clone else None, language,
                                       mode=mode, layout=layout, speaker=speaker)
    return {"layout": [decode_symbolic_row(r, codes) for r in emb[0]],
            "trailing": [decode_symbolic_row(r, codes) for r in trailing[0]],
            "tts_pad": decode_symbolic_row(pad.reshape(-1), codes)}
