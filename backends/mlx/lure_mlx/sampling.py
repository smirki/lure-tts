"""On-device sampling, matching the HF / qwen-tts logits pipeline:
repetition penalty (cb0 only, full history mask) -> suppress tokens -> min_new_tokens EOS mask -> temperature -> top-k -> top-p
-> sample. Logits are float32 (HF casts the last-step logits to float32 before processing).

Sampling uses the Gumbel-max trick with externally supplied noise: argmax(logits + G), G ~ Gumbel(0, 1), which draws from
softmax(logits). Passing the noise in (instead of calling mx.random inside) keeps the functions pure so they can be
mx.compile'd, and makes the eager and compiled paths consume the RNG identically.
"""
from __future__ import annotations

import mlx.core as mx

NEG_INF = float("-inf")


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    """Keep logits >= the k-th largest (ties kept, as HF TopKLogitsWarper)."""
    v = logits.shape[-1]
    if k <= 0 or k >= v:
        return logits
    kth = mx.partition(logits, v - k, axis=-1)[..., v - k:v - k + 1]
    return mx.where(logits < kth, NEG_INF, logits)


def top_p_filter(logits: mx.array, p: float) -> mx.array:
    """HF TopPLogitsWarper: drop the low tail whose cumulative probability is <= 1 - p (always keep the top token)."""
    if p >= 1.0:
        return logits
    order = mx.argsort(logits, axis=-1)  # ascending
    sorted_logits = mx.take_along_axis(logits, order, axis=-1)
    cum = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
    remove_sorted = cum <= (1.0 - p)
    keep_last = mx.zeros_like(remove_sorted)
    remove_sorted = mx.concatenate([remove_sorted[..., :-1], keep_last[..., -1:]], axis=-1)
    remove = mx.put_along_axis(mx.zeros_like(remove_sorted), order, remove_sorted, axis=-1)
    return mx.where(remove, NEG_INF, logits)


def repetition_penalty(logits: mx.array, seen: mx.array, penalty: float) -> mx.array:
    """seen: bool [.., V] mask of every cb0 id generated so far in this segment."""
    if penalty == 1.0:
        return logits
    pen = mx.where(logits < 0, logits * penalty, logits / penalty)
    return mx.where(seen, pen, logits)


def sample(logits: mx.array, noise: mx.array, temperature: float, top_k: int, top_p: float, do_sample: bool = True) -> mx.array:
    """logits float32 [B, V], noise float32 [B, V] Gumbel(0,1) -> int32 [B]."""
    if not do_sample or temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    x = logits / temperature
    x = top_k_filter(x, top_k)
    x = top_p_filter(x, top_p)
    return mx.argmax(x + noise, axis=-1).astype(mx.int32)


def gumbel(shape) -> mx.array:
    return mx.random.gumbel(shape=shape, dtype=mx.float32)
