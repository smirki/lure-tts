"""Lure Turbo on MLX: the grouped code-predictor head (3 group passes + an in-group sequential add-on).

Checked against a float64 reference implementation (conformance/turbo/fixtures/tiny_turbo.*). Per frame, with x2 = [h, E0(cb0)]
(talker hidden, talker codec embedding of cb0):

    X = x2
    for (f, l) in groups:                       # (1,3), (4,7), (8,15)
        Z = norm(layers(X ++ mask[f-1 .. l-2]))  # one causal pass over 4 / 8 / 16 positions, positions 0..L-1
        cb_f = sample(lm_head[f-1](Z[f]))
        csum = 0
        for k in f+1 .. l:
            prev = cemb[k-2](cb_{k-1}); csum += prev
            z = Z[k] + w2(silu(w1([n1(Z[k]); n2(csum); n3(prev)]) + addon_pos[k-1]))
            cb_k = sample(lm_head[k-1](z))
        X = X ++ [cemb[k-1](cb_k) for k in f .. min(l, 14)]

cemb = the talker's code_predictor.model.codec_embedding (shared with the original CP), n1..n3 LayerNorm (eps 1e-5, bias).
The head's 5 layers have the original CP layer architecture (mlx-audio CodePredictorDecoderLayer is reused).
Every position is sampled with the subtalker parameters (Turbo default 0.9 / 50 / 1.0), Gumbel-max with the noise passed in,
like cp.py, so the compiled and eager variants consume the RNG identically.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .cp import _sum_embeds
from .sampling import sample

DEFAULT_GROUPS = ((1, 3), (4, 7), (8, 15))


class _Addon(nn.Module):
    def __init__(self, H: int, A: int, eps: float):
        super().__init__()
        self.n1 = nn.LayerNorm(H, eps=eps)
        self.n2 = nn.LayerNorm(H, eps=eps)
        self.n3 = nn.LayerNorm(H, eps=eps)
        self.w1 = nn.Linear(3 * H, A)
        self.w2 = nn.Linear(A, H)


class TurboHead(nn.Module):
    """Turbo head parameters. Attribute paths equal the reference state_dict keys (turbo_head.safetensors), so
    `load_weights(list(mx.load(path).items()))` loads it directly."""

    def __init__(self, cp_config, talker_hidden: int, addon_hidden: int = 512, layernorm_eps: float = 1e-5,
                 groups=DEFAULT_GROUPS, with_proj: bool = False):
        super().__init__()
        from mlx_audio.tts.models.qwen3_tts.talker import CodePredictorDecoderLayer, RotaryEmbedding
        c = cp_config
        H = c.hidden_size
        if talker_hidden != H and not with_proj:
            raise ValueError("Turbo add-on needs talker hidden == CP hidden (proj = Identity), as in the 0.6B")
        self.groups = tuple(tuple(int(x) for x in g) for g in groups)
        self.n_cb = int(c.num_code_groups) - 1
        self.layers = [CodePredictorDecoderLayer(c, i) for i in range(c.num_hidden_layers)]
        self.norm = nn.RMSNorm(H, eps=c.rms_norm_eps)
        self.lm_head = [nn.Linear(H, c.vocab_size, bias=False) for _ in range(self.n_cb)]
        if with_proj:
            self.proj = nn.Linear(talker_hidden, H, bias=True)
        self.mask = mx.zeros((self.n_cb - 1, talker_hidden))
        self.addon = _Addon(H, addon_hidden, layernorm_eps)
        self.addon_pos = mx.zeros((self.n_cb, addon_hidden))
        self._rope = RotaryEmbedding(c.head_dim, max_position_embeddings=c.max_position_embeddings, base=c.rope_theta)  # "_": not a parameter

    @classmethod
    def from_file(cls, path: str, cp_config, talker_hidden: int, addon_hidden: int = 512, layernorm_eps: float = 1e-5,
                  groups=DEFAULT_GROUPS, dtype=None) -> "TurboHead":
        w = mx.load(path)
        head = cls(cp_config, talker_hidden, addon_hidden, layernorm_eps, groups, with_proj="proj.weight" in w)
        head.load_weights(list(w.items()), strict=True)
        if dtype is not None:
            head.set_dtype(dtype)
        mx.eval(head.parameters())
        return head


class TurboCodePredictor:
    """Same call signature and RNG use as cp.EagerCodePredictor:
        codes15 [15] int32, emb_sum [1,1,D] = cp(hidden_last [1,1,D], cb0 int32 [1], noise float32 [15, V])
    compile=True wraps the whole per-frame function (3 group passes + add-on sampling) in one mx.compile graph."""

    def __init__(self, talker, head: TurboHead, temperature, top_k, top_p, do_sample=True, compile: bool = False):
        self.talker, self.head = talker, head
        self.samp = (temperature, top_k, top_p, do_sample)
        self.cemb = talker.code_predictor.codec_embedding
        self.n = head.n_cb
        dtype = talker.codec_head.weight.dtype
        L = self.n + 1
        cos, sin = head._rope(mx.zeros((1, L, 1), dtype=dtype), mx.arange(L)[None])
        self.cos, self.sin = cos, sin
        self.masks = {}
        for _, last in head.groups:
            n = last + 1
            self.masks[n] = nn.MultiHeadAttention.create_additive_causal_mask(n).astype(dtype)
        mx.eval(self.cos, self.sin, *self.masks.values())
        self._fn = mx.compile(self._frame) if compile else self._frame

    # ------------------------------------------------------------------ pure per-frame math (batch B)
    def _run(self, X):
        hd = self.head
        if "proj" in hd:
            X = hd.proj(X)
        L = X.shape[1]
        pe = (self.cos[:, :L].astype(X.dtype), self.sin[:, :L].astype(X.dtype))
        z = X
        for layer in hd.layers:
            z = layer(z, pe, self.masks[L].astype(X.dtype))
        return hd.norm(z)

    def _add(self, z, csum, prev, k):
        a = self.head.addon
        u = a.w1(mx.concatenate([a.n1(z), a.n2(csum), a.n3(prev)], axis=-1)) + self.head.addon_pos[k - 1].astype(z.dtype)
        return z + a.w2(nn.silu(u))

    def codes(self, x2, noise=None, forced=None):
        """x2 [B, 2, H]; noise [15, B, V] Gumbel or None (argmax); forced [B, 15] int (tests: take these codes instead of
        sampling, the add-on still sees them). Returns (codes [B, 15] int32, logits [B, 15, V] float32)."""
        hd = self.head
        B = x2.shape[0]
        X = x2
        out = [None] * self.n
        logs = [None] * self.n

        def pick(k, lg):
            lg = lg.astype(mx.float32)
            logs[k - 1] = lg
            if forced is not None:
                t = forced[:, k - 1].astype(mx.int32)
            elif noise is None:
                t = mx.argmax(lg, axis=-1).astype(mx.int32)
            else:
                t = sample(lg, noise[k - 1], *self.samp)
            out[k - 1] = t
            return t

        for f, l in hd.groups:
            if l > f:
                m = mx.broadcast_to(hd.mask[f - 1:l - 1].astype(X.dtype)[None], (B, l - f, X.shape[-1]))
                Xg = mx.concatenate([X, m], axis=1)
            else:
                Xg = X
            Z = self._run(Xg)
            pick(f, hd.lm_head[f - 1](Z[:, f]))
            csum = None
            for k in range(f + 1, l + 1):
                prev = self.cemb[k - 2](out[k - 2]).astype(Z.dtype)
                csum = prev if csum is None else csum + prev
                pick(k, hd.lm_head[k - 1](self._add(Z[:, k], csum, prev, k)))
            app = [self.cemb[k - 1](out[k - 1])[:, None].astype(X.dtype) for k in range(f, min(l, self.n - 1) + 1)]
            X = mx.concatenate([X] + app, axis=1)
        return mx.stack(out, axis=1), mx.stack(logs, axis=1)

    def _frame(self, hidden_last, cb0, noise):
        x = mx.concatenate([hidden_last, self.talker.get_input_embeddings()(cb0[None])], axis=1)
        do_sample = self.samp[3] and self.samp[0] > 0
        codes, _ = self.codes(x, noise[:, None, :] if do_sample else None)
        toks = [codes[:, i] for i in range(self.n)]
        return codes[0], _sum_embeds(self.talker, cb0, toks, hidden_last.dtype)

    def __call__(self, hidden_last, cb0, noise):
        return self._fn(hidden_last, cb0, noise)


def load_head_for(talker, cfg, dtype) -> TurboHead:
    """cfg: lure_tts.turbo.TurboConfig (or anything with head_path, groups, addon_hidden, layernorm_eps)."""
    cpc = talker.code_predictor.config
    return TurboHead.from_file(cfg.head_path, cpc, talker.config.hidden_size, cfg.addon_hidden, cfg.layernorm_eps,
                               cfg.groups, dtype=dtype)

