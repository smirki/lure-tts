"""Reference implementation of the Lure Turbo head (make_fastcp) and its samplers. Test ground truth only; the
backend implementations live in backends/*/. Keep make_fastcp, sample and sample_multi unchanged: the fixture in
fixtures/ was produced from them.
"""
NCB = 16
HEADS = {"turbo": [(1, 3), (4, 7), (8, 15)]}

def make_fastcp(cp, emb0, groups, addon=False):
    import copy
    import torch
    from transformers.masking_utils import create_causal_mask

    class FastCP(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.groups = [tuple(g) for g in groups]
            s.layers = copy.deepcopy(cp.model.layers)
            s.norm = copy.deepcopy(cp.model.norm)
            s.rotary = copy.deepcopy(cp.model.rotary_emb)
            s.lm_head = copy.deepcopy(cp.lm_head)
            s.proj = copy.deepcopy(cp.small_to_mtp_projection)
            s.mask = torch.nn.Parameter(torch.stack([cp.model.codec_embedding[k - 2].weight.detach().float().mean(0) for k in range(2, 16)]))
            s._ext = (cp.model.codec_embedding, emb0)  # shared with the talker (tuple: not registered as submodules)
            s.cfg = cp.model.config
            s._W = {}  # inference cache of stacked lm_head weights per group (set after loading / dtype casts)
            s._mc = {}  # inference cache of causal mask / rotary per sequence length
            s.addon = None
            if addon:  # z_k += W2 silu(W1 [ln(z_k); ln(sum of in-group picked code embeddings); ln(emb of cb_{k-1})] + pos_k)
                Hd = cp.model.config.hidden_size
                s.addon = torch.nn.ModuleDict({"n1": torch.nn.LayerNorm(Hd), "n2": torch.nn.LayerNorm(Hd), "n3": torch.nn.LayerNorm(Hd),
                                               "w1": torch.nn.Linear(3 * Hd, 512), "w2": torch.nn.Linear(512, Hd)})
                s.addon_pos = torch.nn.Parameter(torch.zeros(NCB - 1, 512))
                torch.nn.init.zeros_(s.addon["w2"].weight)
                torch.nn.init.zeros_(s.addon["w2"].bias)

        def add(s, z, csum, prev, k):
            """z (..., H) CP output at position k (not group-first), csum = sum of embeddings of this group's codes already picked,
            prev = embedding of cb_{k-1}; k int or (n,) tensor of positions."""
            a = s.addon
            pos = s.addon_pos[k - 1] if isinstance(k, int) else s.addon_pos[k - 1]
            u = a["w1"](torch.cat([a["n1"](z), a["n2"](csum), a["n3"](prev)], -1)) + pos.to(z.dtype)
            return z + a["w2"](torch.nn.functional.silu(u))

        def run(s, X):
            z = s.proj(X)
            L = z.shape[1]
            key = (L, z.shape[0], z.dtype)
            if not s.training and key in s._mc:
                cp2, p2, mask, pe = s._mc[key]
            else:
                cp2 = torch.arange(L, device=z.device)
                p2 = cp2.unsqueeze(0)
                mask = create_causal_mask(config=s.cfg, input_embeds=z, attention_mask=None, cache_position=cp2, past_key_values=None, position_ids=p2)
                pe = s.rotary(z, p2)
                if not s.training:
                    s._mc[key] = (cp2, p2, mask, pe)
            for lyr in s.layers:
                z = lyr(z, attention_mask=mask, position_ids=p2, past_key_values=None, use_cache=False, cache_position=cp2, position_embeddings=pe)[0]
            return s.norm(z)

        def real_embeds(s, h, codes):
            """h (N, H) talker hidden, codes (N, 16) -> (N, 16, H) = [h, emb0(cb0), cemb[0](cb1), ..., cemb[13](cb14)]."""
            cemb, e0 = s._ext
            return torch.cat([h.unsqueeze(1), e0(codes[:, :1]).to(h.dtype)] + [cemb[i - 1](codes[:, i:i + 1]).to(h.dtype) for i in range(1, NCB - 1)], 1)

        def group_logits(s, E, g):
            """Teacher-forced logits of group g = (first, last): (N, last - first + 1, 2048)."""
            first, last = g
            X = E[:, :last + 1].clone()
            if last > first:
                X[:, first + 1:last + 1] = s.mask[first - 1:last - 1].to(X.dtype)
            Z = s.run(X)
            if s.addon is not None and last > first:  # teacher-forced: in-group context from the real codes
                ks = torch.arange(first + 1, last + 1, device=X.device)
                prevs = E[:, first + 1:last + 1]  # embeddings of cb_first .. cb_{last-1}
                csum = prevs.cumsum(1)
                zz = s.add(Z[:, first + 1:last + 1], csum, prevs, ks)
                Z = torch.cat([Z[:, :first + 1], zz], 1)
            return torch.stack([s.lm_head[k - 1](Z[:, k]) for k in range(first, last + 1)], 1)

        @torch.no_grad()
        def sample_codes(s, x2, temps, top_k=50):
            """x2 (B, 2, H) = [h, emb0(cb0)] -> cb1-15 (B, 15). temps(k, is_first) -> temperature (0 = argmax)."""
            cemb, _ = s._ext
            B = x2.shape[0]
            out = torch.zeros(B, NCB - 1, dtype=torch.long, device=x2.device)
            X = x2
            for first, last in s.groups:
                if last > first:
                    X_ = torch.cat([X, s.mask[first - 1:last - 1].to(X.dtype).unsqueeze(0).expand(B, -1, -1)], 1)
                else:
                    X_ = X
                Z = s.run(X_)
                if s.addon is not None and last > first:  # in-group codebooks one by one through the add-on
                    out[:, first - 1] = sample(s.lm_head[first - 1](Z[:, first]), temps(first, True), top_k)
                    csum = None
                    for k in range(first + 1, last + 1):
                        prev = cemb[k - 2](out[:, k - 2]).to(Z.dtype)
                        csum = prev if csum is None else csum + prev
                        out[:, k - 1] = sample(s.lm_head[k - 1](s.add(Z[:, k], csum, prev, k)), temps(k, False), top_k)
                elif last == first:
                    out[:, first - 1] = sample(s.lm_head[first - 1](Z[:, first]), temps(first, True), top_k)
                else:  # all heads of the group in one batched matmul + one batched sampling call
                    if (first, last) not in s._W:
                        s._W[(first, last)] = torch.stack([s.lm_head[k - 1].weight for k in range(first, last + 1)])  # (n, V, H)
                    lg = torch.einsum("bnh,nvh->bnv", Z[:, first:last + 1], s._W[(first, last)])
                    tv = torch.tensor([temps(k, k == first) for k in range(first, last + 1)], device=lg.device)
                    out[:, first - 1:last] = sample_multi(lg, tv, top_k)
                X = torch.cat([X] + [cemb[k - 1](out[:, k - 1:k]) for k in range(first, min(last, NCB - 2) + 1)], 1)
            return out

    return FastCP()

def sample(logits, temp, top_k=50):
    import torch
    if temp <= 0:
        return logits.argmax(-1)
    lg = logits.float() / temp
    v = lg.topk(top_k, -1).values[..., -1:]
    lg = lg.masked_fill(lg < v, float("-inf"))
    g = -torch.log(-torch.log(torch.rand_like(lg).clamp_(1e-10, 1.0)))
    return (lg + g).argmax(-1)

def sample_multi(lg, tv, top_k=50):
    """lg (B, n, V), tv (n,) temperatures (0 = argmax) -> (B, n)."""
    import torch
    am = lg.argmax(-1)
    l = lg.float() / tv.clamp(min=1e-4).view(1, -1, 1)
    v = l.topk(top_k, -1).values[..., -1:]
    l = l.masked_fill(l < v, float("-inf"))
    smp = (l - torch.log(-torch.log(torch.rand_like(l).clamp_(1e-10, 1.0)))).argmax(-1)
    return torch.where((tv > 0).view(1, -1), smp, am)

def temp_fn(spec):
    t = float(spec)
    return lambda k, first: t
