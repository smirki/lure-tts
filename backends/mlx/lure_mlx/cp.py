"""Code predictor (codebooks 1..15) for one frame: eager and compiled variants with the same signature and RNG use.

    codes15, emb_sum = cp(hidden_last [1,1,D], cb0 int32 [1], noise float32 [15, V])
    emb_sum = C(cb0) + sum_g CP_EMB[g-1](cb_g)   (float32 accumulate, cast to model dtype) -> next talker input (+ T(tts_pad))

Compiled variant (opt-in, compile_cp=True): the whole 15-step loop is one fixed-shape graph. K/V live as graph-local arrays
concatenated step by step (no KVCache object, no per-frame zeros + slice-assign), RoPE tables precomputed for the 16 positions,
sampling by Gumbel-max with the noise passed in, top-k by partition threshold. mx.compile traces it once per sampling config.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .sampling import sample


def _sum_embeds(talker, cb0, toks, dtype):
    acc = talker.get_input_embeddings()(cb0[None]).astype(mx.float32)  # [1,1,D]
    for g, t in enumerate(toks):
        acc = acc + talker.code_predictor.codec_embedding[g](t[None]).astype(mx.float32)
    return acc.astype(dtype)


class EagerCodePredictor:
    """mlx-audio's module + KVCache, our sampler (float32 logits, Gumbel noise)."""

    def __init__(self, talker, temperature, top_k, top_p, do_sample=True):
        self.talker, self.cp = talker, talker.code_predictor
        self.samp = (temperature, top_k, top_p, do_sample)
        self.n = self.cp.num_code_groups - 1
        self.cache = self.cp.make_cache()

    def _reset(self):
        for c in self.cache:
            c.keys = None
            c.values = None
            c.offset = 0

    def __call__(self, hidden_last, cb0, noise):
        self._reset()
        dtype = hidden_last.dtype
        x = mx.concatenate([hidden_last, self.talker.get_input_embeddings()(cb0[None])], axis=1)
        toks = []
        for s in range(self.n):
            logits, _, _ = self.cp(x, cache=self.cache, generation_step=s)
            t = sample(logits[:, -1, :].astype(mx.float32), noise[s:s + 1], *self.samp)  # [1]
            toks.append(t)
            x = self.cp.codec_embedding[s](t[None])
        return mx.concatenate(toks), _sum_embeds(self.talker, cb0, toks, dtype)


class CompiledCodePredictor:
    def __init__(self, talker, temperature, top_k, top_p, do_sample=True):
        self.talker, self.cp = talker, talker.code_predictor
        self.samp = (temperature, top_k, top_p, do_sample)
        cfg = self.cp.config
        self.n = self.cp.num_code_groups - 1
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        dtype = talker.codec_head.weight.dtype
        pos = mx.arange(self.n + 1)[None]
        cos, sin = self.cp.model.rotary_emb(mx.zeros((1, self.n + 1, 1), dtype=dtype), pos)
        self.cos, self.sin = cos, sin
        self.mask2 = nn.MultiHeadAttention.create_additive_causal_mask(2).astype(dtype)
        mx.eval(self.cos, self.sin, self.mask2)
        self._fn = mx.compile(self._frame)

    def _layer(self, layer, x, k_prev, v_prev, cos, sin, mask):
        a = layer.self_attn
        b, L, _ = x.shape
        h = layer.input_layernorm(x)
        q = a.q_norm(a.q_proj(h).reshape(b, L, self.nh, self.hd)).transpose(0, 2, 1, 3)
        k = a.k_norm(a.k_proj(h).reshape(b, L, self.nkv, self.hd)).transpose(0, 2, 1, 3)
        v = a.v_proj(h).reshape(b, L, self.nkv, self.hd).transpose(0, 2, 1, 3)
        c, s = cos[:, None], sin[:, None]
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s
        if k_prev is not None:
            k = mx.concatenate([k_prev, k], axis=2)
            v = mx.concatenate([v_prev, v], axis=2)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=mask)
        x = x + a.o_proj(o.transpose(0, 2, 1, 3).reshape(b, L, -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        return x, k, v

    def _frame(self, hidden_last, cb0, noise):
        talker, cp = self.talker, self.cp
        dtype = hidden_last.dtype
        x = mx.concatenate([hidden_last, talker.get_input_embeddings()(cb0[None])], axis=1)
        if cp.small_to_mtp_projection is not None:
            x = cp.small_to_mtp_projection(x)
        layers = cp.model.layers
        ks, vs = [None] * len(layers), [None] * len(layers)
        toks, pos = [], 0
        for s in range(self.n):
            L = x.shape[1]
            cos, sin = self.cos[:, pos:pos + L], self.sin[:, pos:pos + L]
            mask = self.mask2 if L > 1 else None
            h = x
            for i, layer in enumerate(layers):
                h, ks[i], vs[i] = self._layer(layer, h, ks[i], vs[i], cos, sin, mask)
            h = cp.model.norm(h)
            logits = cp.lm_head[s](h[:, -1, :]).astype(mx.float32)
            t = sample(logits, noise[s:s + 1], *self.samp)
            toks.append(t)
            x = cp.codec_embedding[s](t[None])
            if cp.small_to_mtp_projection is not None:
                x = cp.small_to_mtp_projection(x)
            pos += L
        return mx.concatenate(toks), _sum_embeds(talker, cb0, toks, dtype)

    def __call__(self, hidden_last, cb0, noise):
        return self._fn(hidden_last, cb0, noise)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)
