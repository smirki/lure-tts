"""Lure Turbo for the CUDA backend: the grouped code-predictor head.

Per frame, instead of the 15 sequential sub-talker passes, the head runs 3 group passes over (1,3), (4,7), (8,15):
  one causal forward of the head's own 5 layers over [h, E0(cb0), real code embeddings so far, learned mask rows M_k],
  the group-first codebook is sampled from that pass directly, the later in-group codebooks one by one through the add-on
      z_k += w2 silu(w1 [ln1(z_k); ln2(csum of this group's picked code embeddings); ln3(emb of cb_{k-1})] + addon_pos[k-1]) + b2,
  then the group's picked code embeddings are appended before the next group.
`TurboHead` uses the module names of the reference head, so `turbo_head.safetensors` loads key for key.
`TurboPredictorGraph` is a drop-in for graphs.BatchedPredictorGraph (same buffers, same per-row sampling parameters and
uniform noise layout [B, 15, V]); its fixed-shape loop (sequence lengths 4, 8, 16) is captured as one CUDA graph.
Tested against the reference implementation in conformance/turbo/ref_fastcp.py (tests/test_turbo_cpu.py).
"""
from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from .graphs import _capture
from .sampling import gumbel_from_uniform, warp_rows

logger = logging.getLogger(__name__)

NCB = 16


# --------------------------------------------------------------------------------------------- config (core or fallback)
@dataclass(frozen=True)
class _Cfg:
    head_path: str
    groups: tuple
    layernorm_eps: float = 1e-5
    addon_hidden: int = 512
    hidden_size: Optional[int] = None  # turbo.json hidden_size; checked against the talker (backend.check_turbo)


def resolve_turbo(model_dir: str, turbo=None, turbo_head: Optional[str] = None):
    """lure_tts.turbo.resolve when the core is installed; otherwise the same rules on turbo.json directly."""
    try:
        from lure_tts.turbo import resolve
        return resolve(model_dir, turbo=turbo, turbo_head=turbo_head)
    except ImportError:
        pass
    p = os.path.join(model_dir, "turbo.json")
    flag = None if turbo in (None, "", "auto") else str(turbo).lower() in ("1", "true", "yes", "on")
    if flag is False or (flag is None and not os.path.isfile(p)):
        return None
    with open(p) as f:
        raw = json.load(f)
    return _Cfg(turbo_head or os.path.join(model_dir, raw.get("head_file", "turbo_head.safetensors")),
                tuple(tuple(g) for g in raw["groups"]), float(raw.get("addon", {}).get("layernorm_eps", 1e-5)),
                int(raw.get("addon", {}).get("hidden", 512)),
                int(raw["hidden_size"]) if "hidden_size" in raw else None)


# --------------------------------------------------------------------------------------------- head module
class TurboHead(torch.nn.Module):
    """Turbo head. Module names match the reference (layers, norm, rotary, lm_head, proj, mask, addon.{n1,n2,n3,w1,w2},
    addon_pos)."""

    def __init__(self, code_predictor, groups, addon_hidden: int = 512, ln_eps: float = 1e-5):
        super().__init__()
        cp = code_predictor
        self.groups = [tuple(int(x) for x in g) for g in groups]
        pos = [k for f, l in self.groups for k in range(f, l + 1)]
        if pos != list(range(1, NCB)):
            raise ValueError(f"groups must tile 1..15, got {self.groups}")
        self.layers = copy.deepcopy(cp.model.layers)
        self.norm = copy.deepcopy(cp.model.norm)
        self.rotary = copy.deepcopy(cp.model.rotary_emb)
        self.lm_head = copy.deepcopy(cp.lm_head)
        self.proj = copy.deepcopy(cp.small_to_mtp_projection)
        E = cp.model.codec_embedding[0].weight.shape[1]  # talker hidden (embedding dim)
        Hd = cp.model.config.hidden_size
        if E != Hd:
            raise ValueError(f"the Turbo head needs talker hidden == code predictor hidden (0.6B only), got {E} vs {Hd}")
        self.mask = torch.nn.Parameter(torch.zeros(NCB - 2, E), requires_grad=False)
        self.addon = torch.nn.ModuleDict({"n1": torch.nn.LayerNorm(Hd, eps=ln_eps), "n2": torch.nn.LayerNorm(Hd, eps=ln_eps),
                                          "n3": torch.nn.LayerNorm(Hd, eps=ln_eps), "w1": torch.nn.Linear(3 * Hd, addon_hidden),
                                          "w2": torch.nn.Linear(addon_hidden, Hd)})
        self.addon_pos = torch.nn.Parameter(torch.zeros(NCB - 1, addon_hidden), requires_grad=False)
        self._ext = (cp.model.codec_embedding,)  # shared with the talker; a tuple, so not a registered submodule
        self.requires_grad_(False)

    @property
    def cemb(self):
        return self._ext[0]

    def load_head(self, sd: dict):
        own = set(self.state_dict())
        missing, unexpected = own - set(sd), set(sd) - own
        if missing or unexpected:
            raise ValueError(f"turbo head keys mismatch: missing {sorted(missing)[:6]} unexpected {sorted(unexpected)[:6]}")
        with torch.no_grad():
            params = dict(self.named_parameters())
            params.update(dict(self.named_buffers()))
            for k, v in sd.items():
                params[k].copy_(v.to(params[k].dtype))

    # Same op order as the reference so bf16 results match bit for bit.
    def add(self, z, csum, prev, k: int):
        a = self.addon
        u = a["w1"](torch.cat([a["n1"](z), a["n2"](csum), a["n3"](prev)], -1)) + self.addon_pos[k - 1].to(z.dtype)
        return z + a["w2"](torch.nn.functional.silu(u))

    def run(self, X, attn_mask, pos_emb, position_ids, cache_position):
        z = self.proj(X)
        for lyr in self.layers:
            z = lyr(z, attention_mask=attn_mask, position_ids=position_ids, past_key_values=None, use_cache=False,
                    cache_position=cache_position, position_embeddings=pos_emb)[0]
        return self.norm(z)

    def make_static(self, device, dtype):
        """Causal masks + rotary tables for every group's sequence length (precomputed: capture-safe).
        With SDPA the mask is None, so attention runs with is_causal=True: the same kernel path the reference takes
        (create_causal_mask returns None for a plain causal SDPA mask), hence the same bf16 numerics on GPU."""
        st = {}
        sdpa = getattr(self.layers[0].self_attn.config, "_attn_implementation", None) == "sdpa"
        X = torch.zeros(1, NCB, self.mask.shape[1], dtype=dtype, device=device)
        for f, l in self.groups:
            L = l + 1
            cp_ = torch.arange(L, device=device)
            pid = cp_.unsqueeze(0)
            keys = torch.arange(L, device=device)
            allowed = keys.view(1, -1) <= keys.view(-1, 1)
            m = torch.where(allowed, torch.zeros((), dtype=dtype, device=device),
                            torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)).view(1, 1, L, L)
            st[(f, l)] = (None if sdpa else m, self.rotary(X[:, :L], pid), pid, cp_)
        return st

    def frame(self, x2, sample, static):
        """x2 [B, 2, H] = [h, E0(cb0)] -> codes [B, 15]. sample(logits [B, V], i) -> [B] long (i = codebook index - 1)."""
        B = x2.shape[0]
        out = []
        X = x2
        for f, l in self.groups:
            X_ = torch.cat([X, self.mask[f - 1:l - 1].to(X.dtype).unsqueeze(0).expand(B, -1, -1)], 1) if l > f else X
            m, pe, pid, cpos = static[(f, l)]
            Z = self.run(X_, m, pe, pid, cpos)
            out.append(sample(self.lm_head[f - 1](Z[:, f]), f - 1))
            csum = None
            for k in range(f + 1, l + 1):
                prev = self.cemb[k - 2](out[k - 2]).to(Z.dtype)
                csum = prev if csum is None else csum + prev
                out.append(sample(self.lm_head[k - 1](self.add(Z[:, k], csum, prev, k)), k - 1))
            X = torch.cat([X] + [self.cemb[k - 1](out[k - 1].unsqueeze(1)) for k in range(f, min(l, NCB - 2) + 1)], 1)
        return torch.stack(out, 1)


def load_turbo_head(code_predictor, cfg, device, dtype) -> TurboHead:
    from safetensors.torch import load_file
    head = TurboHead(code_predictor, cfg.groups, getattr(cfg, "addon_hidden", 512), getattr(cfg, "layernorm_eps", 1e-5))
    head.load_head(load_file(cfg.head_path))
    return head.to(device=device, dtype=dtype).eval()


# --------------------------------------------------------------------------------------------- batched graph
class TurboPredictorGraph:
    """Drop-in for BatchedPredictorGraph running the Turbo head: inputs [B, 2, H] -> tokens [B, 15]."""

    def __init__(self, head: TurboHead, talker_hidden_size: int, batch: int, device="cuda", dtype=torch.bfloat16,
                 use_graph: bool = True):
        self.head = head
        self.B, self.device, self.dtype, self.use_graph = batch, device, dtype, use_graph
        self.num_cb = NCB - 1
        self.vocab = head.lm_head[0].out_features
        self.input_buf = torch.zeros(batch, 2, talker_hidden_size, dtype=dtype, device=device)
        self.out_tokens = torch.zeros(batch, self.num_cb, dtype=torch.long, device=device)
        self.uniform = torch.full((batch, self.num_cb, self.vocab), 0.5, dtype=torch.float32, device=device)
        self.temperature = torch.full((batch,), 0.9, dtype=torch.float32, device=device)
        self.top_k = torch.full((batch,), 50, dtype=torch.long, device=device)
        self.top_p = torch.ones(batch, dtype=torch.float32, device=device)
        self.do_sample = torch.ones(batch, dtype=torch.bool, device=device)
        self.static = head.make_static(device, dtype)
        self.graph = None

    def _sample(self, logits: torch.Tensor, i: int) -> torch.Tensor:
        x = logits.float()
        greedy = x.argmax(dim=-1)
        w = warp_rows(x, self.temperature, self.top_k, self.top_p)
        sampled = (w + gumbel_from_uniform(self.uniform[:, i])).argmax(dim=-1)
        return torch.where(self.do_sample, sampled, greedy)

    def _loop(self):
        self.out_tokens.copy_(self.head.frame(self.input_buf, self._sample, self.static))

    @torch.inference_mode()
    def capture(self, num_warmup: int = 3):
        if not self.use_graph:
            return
        self.graph = _capture(self._loop, self.device, num_warmup)

    def set_row_params(self, row: int, temperature: float, top_k: int, top_p: float, do_sample: bool):
        self.temperature[row] = float(temperature)
        self.top_k[row] = int(top_k)
        self.top_p[row] = float(top_p)
        self.do_sample[row] = bool(do_sample)

    @torch.inference_mode()
    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        self.input_buf.copy_(pred_input)
        if self.use_graph:
            self.graph.replay()
        else:
            self._loop()
        return self.out_tokens.clone()
