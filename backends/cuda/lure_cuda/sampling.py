"""Sampling for the Lure CUDA backend: per-request parameters, batched (row-wise) HF-equivalent logits processing,
and Gumbel-max sampling driven by per-request RNG streams.

Semantics follow transformers `generate()` as used by qwen-tts 0.1.1:
  codebook 0 (talker): logits.float() -> repetition penalty over all previously generated cb0 ids (unique set)
                       -> suppress [vocab-1024, vocab) except EOS -> EOS masked while frames < min_new_tokens
                       -> temperature -> top-k -> top-p -> sample
  codebooks 1-15 (sub-talker): logits.float() -> temperature -> top-k -> top-p -> sample (no penalty, no suppression)
Sampling uses the Gumbel-max trick, argmax(logits + Gumbel noise), which draws exactly from softmax(logits); the noise
comes from a per-request torch.Generator so a seeded request is reproducible whatever else shares the batch.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Optional

import torch

FRAME_RATE = 12.5


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    do_sample: bool = True
    subtalker_do_sample: bool = True
    min_new_tokens: int = 2
    max_new_tokens: Optional[int] = None  # None -> max_frames_for(text)

    def validate(self) -> "SamplingParams":
        if self.do_sample and not self.temperature > 0:
            raise ValueError("temperature must be > 0 when sampling")
        if self.subtalker_do_sample and not self.subtalker_temperature > 0:
            raise ValueError("subtalker_temperature must be > 0 when sampling")
        if self.top_k < 0 or self.subtalker_top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables)")
        if not (0.0 < self.top_p <= 1.0) or not (0.0 < self.subtalker_top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if not self.repetition_penalty > 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.min_new_tokens < 0:
            raise ValueError("min_new_tokens must be >= 0")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        return self


_FIELDS = {f.name for f in fields(SamplingParams)}
# accepted aliases (OpenAI-ish / qwen-tts kwarg names)
_ALIASES = {"subtalker_dosample": "subtalker_do_sample", "max_tokens": "max_new_tokens"}


def max_frames_for(text: str) -> int:
    """Per-segment frame cap (same rule as core lure_tts.defaults.max_frames_for)."""
    try:
        from lure_tts.defaults import max_frames_for as core_rule  # the core owns the rule when present

        return int(core_rule(text))
    except Exception:
        words = max(1, len(text.split()))
        return int(FRAME_RATE * (1.8 * words / 2.6 + 4))


def resolve_sampling(sampling: Any = None, **overrides) -> SamplingParams:
    """Accepts None, a dict, a SamplingParams, or the core's `lure_tts.defaults.Sampling` (anything with to_dict() or the
    same attribute names). Unknown keys raise. Keyword overrides with value None are ignored."""
    if sampling is None:
        d: dict = {}
    elif isinstance(sampling, SamplingParams):
        d = {f: getattr(sampling, f) for f in _FIELDS}
    elif isinstance(sampling, dict):
        d = dict(sampling)
    elif hasattr(sampling, "to_dict"):
        d = dict(sampling.to_dict())
    else:
        d = {f: getattr(sampling, f) for f in _FIELDS if hasattr(sampling, f)}
    d.update({k: v for k, v in overrides.items() if v is not None})
    out = {}
    for k, v in d.items():
        k = _ALIASES.get(k, k)
        if k not in _FIELDS:
            raise ValueError(f"unknown sampling field {k!r}; allowed: {sorted(_FIELDS)}")
        if v is not None:
            out[k] = v
    return SamplingParams(**out).validate()


# ----------------------------------------------------------------------------------------------- row-wise processing
def apply_repetition_penalty_rows(logits: torch.Tensor, seen: torch.Tensor, penalty: torch.Tensor) -> torch.Tensor:
    """HF RepetitionPenaltyLogitsProcessor, per row. logits [B, V] float, seen [B, V] bool, penalty [B] float."""
    p = penalty.view(-1, 1).to(logits.dtype)
    pen = torch.where(logits < 0, logits * p, logits / p)
    return torch.where(seen, pen, logits)


def warp_rows(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Temperature -> top-k -> top-p, per row, HF semantics. logits [B, V] float32.
    temperature [B] float (> 0), top_k [B] long (0 = off), top_p [B] float (1.0 = off). Removed entries become -inf."""
    V = logits.shape[-1]
    neg_inf = logits.new_full((), float("-inf"))  # fill kernel, no host->device copy (CUDA graph capture safe)
    x = logits / temperature.view(-1, 1).to(logits.dtype)
    # top-k: keep scores >= k-th largest (ties kept, like TopKLogitsWarper)
    k = torch.where(top_k > 0, top_k.clamp(max=V), torch.full_like(top_k, V))
    sorted_desc, _ = torch.sort(x, dim=-1, descending=True)
    kth = sorted_desc.gather(-1, (k - 1).view(-1, 1))
    x = torch.where(x < kth, neg_inf, x)
    # top-p (TopPLogitsWarper): ascending sort, drop tokens whose cumulative prob <= 1 - top_p, keep the best one
    s_asc, idx_asc = torch.sort(x, dim=-1, descending=False)
    cum = s_asc.softmax(dim=-1).cumsum(dim=-1)
    remove = cum <= (1 - top_p.view(-1, 1).to(cum.dtype))
    remove[..., -1] = False
    remove = remove & (top_p.view(-1, 1) < 1.0)
    remove_orig = torch.zeros_like(remove).scatter(-1, idx_asc, remove)
    return torch.where(remove_orig, neg_inf, x)


def gumbel_from_uniform(u: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(u.dtype).tiny
    return -torch.log(-torch.log(u.clamp(min=tiny, max=1.0 - 1e-7)))


def sample_rows(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    do_sample: torch.Tensor,
    uniform: torch.Tensor,
) -> torch.Tensor:
    """logits [B, V] (any float dtype, processed in float32); uniform [B, V] U(0,1) noise. Returns [B] long.
    Greedy rows (do_sample False) take argmax of the unwarped logits, like HF with do_sample=False."""
    x = logits.float()
    greedy = x.argmax(dim=-1)
    w = warp_rows(x, temperature, top_k, top_p)
    sampled = (w + gumbel_from_uniform(uniform.float())).argmax(dim=-1)
    return torch.where(do_sample.view(-1), sampled, greedy)


def suppress_mask(vocab_size: int, eos_id: int, device=None) -> torch.Tensor:
    """[V] bool: codebook-0 ids [vocab-1024, vocab) except EOS (qwen-tts talker suppress_tokens)."""
    m = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    m[max(0, vocab_size - 1024):] = True
    m[eos_id] = False
    return m


def mask_rows(logits: torch.Tensor, suppress: torch.Tensor, eos_id: int, block_eos: torch.Tensor) -> torch.Tensor:
    """Apply the static suppress mask [V] and per-row EOS blocking (min_new_tokens) [B] bool."""
    neg_inf = logits.new_full((), float("-inf"))  # fill kernel, no host->device copy (CUDA graph capture safe)
    x = torch.where(suppress.view(1, -1), neg_inf, logits)
    col = torch.zeros_like(x, dtype=torch.bool)
    col[:, eos_id] = True
    return torch.where(col & block_eos.view(-1, 1), neg_inf, x)


def make_generator(seed: Optional[int], device) -> torch.Generator:
    g = torch.Generator(device=device)
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)
    g.manual_seed(int(seed))
    return g


__all__ = [
    "SamplingParams", "resolve_sampling", "max_frames_for", "apply_repetition_penalty_rows", "warp_rows",
    "sample_rows", "suppress_mask", "mask_rows", "make_generator", "gumbel_from_uniform",
]
