"""Prefill embeddings straight from the core's prompt rows (lure_tts.prompt.build_prompt via req.build_prompt), so the
mlx prompt is the reference layout by construction, for every mode and layout.
Row = [text_id | None, codec]; codec = int id (codec tokens and preset speakers) | "spk" | "ref:<t>" | None."""
from __future__ import annotations

import mlx.core as mx
import numpy as np


def ref_block(talker, ref_codes: np.ndarray, dtype) -> mx.array:
    """REF(t) = C(codes[t,0]) + sum_g CP_EMB[g-1](codes[t,g]) for every reference frame -> [T, D].
    Accumulated in float32 and rounded once (torch .sum over bf16 accumulates in float32)."""
    codes = mx.array(np.asarray(ref_codes, dtype=np.int32))  # [T, 16]
    acc = talker.get_input_embeddings()(codes[:, 0]).astype(mx.float32)
    for g in range(1, codes.shape[1]):
        acc = acc + talker.code_predictor.codec_embedding[g - 1](codes[:, g]).astype(mx.float32)
    return acc.astype(dtype)


def split_rows(rows):
    n = len(rows)
    t_ids = np.zeros(n, np.int32)
    t_mask = np.zeros(n, np.float32)
    c_ids = np.zeros(n, np.int32)
    c_mask = np.zeros(n, np.float32)
    s_mask = np.zeros(n, np.float32)
    r_idx = np.zeros(n, np.int32)
    r_mask = np.zeros(n, np.float32)
    for i, (t, c) in enumerate(rows):
        if t is not None:
            t_ids[i], t_mask[i] = t, 1
        if c is None:
            continue
        if isinstance(c, str):
            if c == "spk":
                s_mask[i] = 1
            elif c.startswith("ref:"):
                r_idx[i], r_mask[i] = int(c[4:]), 1
            else:
                raise ValueError(f"bad codec cell {c!r}")
        else:
            c_ids[i], c_mask[i] = int(c), 1
    return t_ids, t_mask, c_ids, c_mask, s_mask, r_idx, r_mask


def embed_rows(talker, rows, spk: mx.array | None, refs: mx.array | None, dtype) -> mx.array:
    """-> [1, L, D] prefill embeddings: T(text) + C(codec) | SPK | REF(t) per row."""
    t_ids, t_mask, c_ids, c_mask, s_mask, r_idx, r_mask = split_rows(rows)
    col = lambda m: mx.array(m)[None, :, None].astype(dtype)  # noqa: E731
    out = talker.text_projection(talker.get_text_embeddings()(mx.array(t_ids)[None])).astype(dtype) * col(t_mask)
    out = out + talker.get_input_embeddings()(mx.array(c_ids)[None]).astype(dtype) * col(c_mask)
    if s_mask.any():
        if spk is None:
            raise ValueError("layout has a speaker row but no speaker embedding")
        out = out + spk.reshape(1, 1, -1).astype(dtype) * col(s_mask)
    if r_mask.any():
        if refs is None:
            raise ValueError("layout has ref rows but no reference codes")
        out = out + refs[mx.array(r_idx)][None].astype(dtype) * col(r_mask)
    return out
