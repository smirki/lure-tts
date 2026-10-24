"""Batched CUDA-graph decode steps for the Lure CUDA backend.

Derived from faster-qwen3-tts talker_graph.py / predictor_graph.py (MIT, Andres Marafioti), extended to a fixed batch
of B slots so several requests decode in one graph replay.

Slot model (talker): all slots share one write position `p` in a StaticCache of max_seq_len. A request whose prefill has
length L is copied into its slot at [p - L, p); keys before `start[b] = p - L` are masked for that row, and its RoPE
position is `p + rope_delta[b]` with rope_delta[b] = L - p (plus the prefill rope delta), so the model sees exactly the
positions of an unpadded single-request run. Keys already carry their rotary phase, so a slot's KV block can be moved
(`rebase`) without changing results: attention is order-free given the mask.

`use_graph=False` runs the identical step eagerly (CPU unit tests, debugging).
"""
from __future__ import annotations

import torch
from transformers import StaticCache

from ._compat import static_layer_init
from .sampling import gumbel_from_uniform, warp_rows


def _static_cache(config, max_cache_len: int):
    try:
        return StaticCache(config=config, max_cache_len=max_cache_len)
    except TypeError:  # older signature
        return StaticCache(config=config, max_batch_size=1, max_cache_len=max_cache_len)


def _init_layers(cache, config, batch: int, dtype, device):
    num_kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dummy = torch.zeros(batch, num_kv, 1, head_dim, dtype=dtype, device=device)
    for layer in cache.layers:
        static_layer_init(layer, dummy, dummy)


def _cumulative(cache):
    """transformers >= 5.x StaticLayer ignores cache_position and writes at its own device-side `cumulative_length`
    (incremented in place, so it is replayed inside a CUDA graph). Older versions index by cache_position. We drive
    both: cache_position buffer + cumulative_length set to the same value before every step."""
    return [l.cumulative_length for l in cache.layers if isinstance(getattr(l, "cumulative_length", None), torch.Tensor)]


def _capture(fn, device, num_warmup: int = 3, pre=None):
    """Warm up `fn` then capture it into a CUDA graph on a side stream (faster-qwen3-tts recipe)."""
    for _ in range(num_warmup):
        if pre:
            pre()
        fn()
    torch.cuda.synchronize()
    idx = torch.device(device).index
    idx = idx if idx is not None else torch.cuda.current_device()
    with torch.cuda.device(idx):
        graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            if pre:
                pre()
            fn()
            torch.cuda.synchronize()
            if pre:
                pre()
            with torch.cuda.graph(graph):
                fn()
        torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    return graph


class BatchedTalkerGraph:
    """One talker decode step for B slots, inputs [B, 1, H] -> hidden [B, 1, H]."""

    def __init__(self, talker_model, talker_config, batch: int, max_seq_len: int, device="cuda",
                 dtype=torch.bfloat16, use_graph: bool = True):
        if getattr(talker_model.config, "sliding_window", None) is not None and getattr(
                talker_model.config, "use_sliding_window", False):
            raise NotImplementedError("batched talker graph assumes full attention (Lure: sliding_window=None)")
        self.model = talker_model
        self.B, self.max_seq_len = batch, max_seq_len
        self.device, self.dtype, self.use_graph = device, dtype, use_graph
        self.hidden_size = talker_config.hidden_size
        self.num_layers = talker_config.num_hidden_layers
        self.cache = _static_cache(talker_model.config, max_seq_len)
        _init_layers(self.cache, talker_model.config, batch, dtype, device)

        self.input_buf = torch.zeros(batch, 1, self.hidden_size, dtype=dtype, device=device)
        self.output_buf = torch.zeros(batch, 1, self.hidden_size, dtype=dtype, device=device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self.start = torch.zeros(batch, dtype=torch.long, device=device)  # first visible key per slot
        self.rope_delta = torch.zeros(batch, dtype=torch.long, device=device)  # position = cache_position + delta
        self.position_ids = torch.zeros(3, batch, 1, dtype=torch.long, device=device)
        self.attn_mask = torch.zeros(batch, 1, 1, max_seq_len, dtype=dtype, device=device)
        self._cum = _cumulative(self.cache)
        self._keys = torch.arange(max_seq_len, device=device).view(1, -1)
        self._zero = torch.zeros((), dtype=dtype, device=device)
        self._min = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)
        # KV-length buckets: a step attends only to keys [0, n) for the smallest bucket n > p, not all max_seq_len.
        # Attending over all 4096 keys at B=8 measured 113 ms per step on an L4. One graph per bucket.
        self.buckets = sorted({n for n in (256, 512, 1024, 2048) if n < max_seq_len} | {max_seq_len})
        self.view_len = max_seq_len  # python int read while tracing: baked into each bucket's graph
        for layer in self.cache.layers:
            layer.update = self._bounded_update(layer.update)
        self.graphs = {}

    # ------------------------------------------------------------------ kv access
    def layer_kv(self, li: int):
        layer = self.cache.layers[li]
        return layer.keys, layer.values

    def write_prefill(self, slot: int, past_key_values, p: int) -> int:
        """Copy a batch-1 prefill DynamicCache into `slot`, ending right before write position p. Returns L."""
        L = None
        for li in range(self.num_layers):
            src = past_key_values.layers[li]
            k, v = src.keys, src.values  # [1, kv, L, hd]
            L = k.shape[2]
            if L > p:
                raise RuntimeError(f"prefill length {L} exceeds write position {p}")
            dk, dv = self.layer_kv(li)
            dk[slot, :, p - L:p].copy_(k[0])
            dv[slot, :, p - L:p].copy_(v[0])
        return int(L)

    def move_slot(self, slot: int, old_p: int, new_p: int, n: int) -> None:
        """Move slot's last n keys/values from [old_p - n, old_p) to [new_p - n, new_p)."""
        if n <= 0 or old_p == new_p:
            return
        for li in range(self.num_layers):
            dk, dv = self.layer_kv(li)
            k = dk[slot, :, old_p - n:old_p].clone()
            v = dv[slot, :, old_p - n:old_p].clone()
            dk[slot, :, new_p - n:new_p].copy_(k)
            dv[slot, :, new_p - n:new_p].copy_(v)

    def _bounded_update(self, orig):
        """StaticLayer.update writes into the full cache as before but returns only keys/values [:, :, :view_len]
        (a view, no copy), so attention cost follows the live length instead of max_seq_len."""
        def update(*args, **kwargs):
            k, v = orig(*args, **kwargs)
            n = self.view_len
            return (k, v) if n >= k.shape[2] else (k[:, :, :n], v[:, :, :n])
        return update

    def bucket_for(self, p: int) -> int:
        for n in self.buckets:
            if p < n:
                return n
        raise ValueError(f"write position {p} >= max_seq_len {self.max_seq_len}")

    # ------------------------------------------------------------------ step
    def _prepare(self):
        """Mask + position ids from the device buffers (outside the graph; cheap)."""
        cp = self.cache_position.view(1, 1)
        for c in self._cum:
            c.copy_(self.cache_position[0])
        allowed = (self._keys >= self.start.view(-1, 1)) & (self._keys <= cp)
        self.attn_mask.copy_(torch.where(allowed, self._zero, self._min).view(self.B, 1, 1, -1))
        pos = (cp + self.rope_delta.view(-1, 1))  # [B, 1]
        self.position_ids.copy_(pos.unsqueeze(0).expand(3, -1, -1))

    def _step(self):
        n = self.view_len
        out = self.model(
            inputs_embeds=self.input_buf,
            attention_mask=self.attn_mask if n >= self.max_seq_len else self.attn_mask[..., :n],
            past_key_values=self.cache,
            cache_position=self.cache_position,
            position_ids=self.position_ids,
            use_cache=True,
        )
        self.output_buf.copy_(out.last_hidden_state)

    @torch.inference_mode()
    def capture(self, num_warmup: int = 3):
        if not self.use_graph:
            return
        # warm up at a harmless position (writes go to the bucket's last cache column, which is always re-written
        # before use: keys are only read at positions <= p after they were written at p)
        self.start.zero_()
        self.rope_delta.zero_()
        for n in self.buckets:
            self.view_len = n
            self.cache_position.fill_(n - 1)
            # pre=_prepare: every warmup/capture call advances StaticLayer.cumulative_length in place (transformers 5),
            # so reset it to cache_position before each one, or the second warmup writes past the end (index_copy_ assert)
            self.graphs[n] = _capture(self._step, self.device, num_warmup, pre=self._prepare)
        self.view_len = self.max_seq_len

    @torch.inference_mode()
    def run(self, inputs_embeds: torch.Tensor, p: int) -> torch.Tensor:
        self.input_buf.copy_(inputs_embeds)
        self.cache_position.fill_(p)
        self._prepare()
        n = self.bucket_for(p)
        if self.use_graph:
            self.graphs[n].replay()
        else:
            self.view_len = n
            try:
                self._step()
            finally:
                self.view_len = self.max_seq_len
        return self.output_buf


class BatchedPredictorGraph:
    """The 15-step code predictor for B rows, with per-row sub-talker sampling parameters and noise buffers.
    Inputs [B, 2, H_talker] -> tokens [B, 15]. Every predictor input (the [hidden, C(cb0)] pair and each CP_EMB[g-1]
    row, all in the talker space) goes through small_to_mtp_projection first: Linear(2048 -> 1024) on 1.7B, Identity
    on 0.6B. The KV cache and attention use the predictor config (hidden 1024 on both sizes)."""

    def __init__(self, code_predictor, pred_config, talker_hidden_size: int, batch: int, device="cuda",
                 dtype=torch.bfloat16, use_graph: bool = True):
        cp = code_predictor
        self.B, self.device, self.dtype, self.use_graph = batch, device, dtype, use_graph
        self.small_to_mtp = cp.small_to_mtp_projection
        self.pred_model = cp.model
        self.lm_heads = cp.lm_head
        self.codec_embeds = cp.model.codec_embedding
        self.num_cb = pred_config.num_code_groups - 1
        self.max_seq = 2 + self.num_cb
        self.vocab = self.lm_heads[0].out_features
        self.has_sliding = "sliding_attention" in (getattr(self.pred_model.config, "layer_types", None) or [])
        self.cache = _static_cache(pred_config, self.max_seq)
        _init_layers(self.cache, pred_config, batch, dtype, device)

        self.input_buf = torch.zeros(batch, 2, talker_hidden_size, dtype=dtype, device=device)
        self.out_tokens = torch.zeros(batch, self.num_cb, dtype=torch.long, device=device)
        self.uniform = torch.full((batch, self.num_cb, self.vocab), 0.5, dtype=torch.float32, device=device)
        self.temperature = torch.full((batch,), 0.9, dtype=torch.float32, device=device)
        self.top_k = torch.full((batch,), 50, dtype=torch.long, device=device)
        self.top_p = torch.ones(batch, dtype=torch.float32, device=device)
        self.do_sample = torch.ones(batch, dtype=torch.bool, device=device)
        self.prefill_pos = torch.arange(2, device=device)
        self.decode_pos = [torch.tensor([2 + i], device=device) for i in range(self.num_cb - 1)]
        self._cum = _cumulative(self.cache)
        self.prefill_attn = self._mask(2, self.prefill_pos)
        self.decode_attn = [self._mask(1, p) for p in self.decode_pos]
        self.graph = None

    def _mask(self, q: int, cache_position: torch.Tensor):
        keys = torch.arange(self.max_seq, device=self.device)
        allowed = keys.unsqueeze(0) <= cache_position.unsqueeze(1)
        mn = torch.finfo(self.dtype).min
        z = torch.zeros((), dtype=self.dtype, device=self.device)
        m = torch.full((), mn, dtype=self.dtype, device=self.device)
        full = torch.where(allowed, z, m).unsqueeze(0).unsqueeze(0)  # [1, 1, q, max_seq] broadcast over B
        if self.has_sliding:
            w = self.pred_model.config.sliding_window
            inw = keys.unsqueeze(0) > (cache_position.unsqueeze(1) - w)
            sl = torch.where(allowed & inw, z, m).unsqueeze(0).unsqueeze(0)
            return {"full_attention": full, "sliding_attention": sl}
        return {"full_attention": full}

    def _sample(self, logits: torch.Tensor, i: int) -> torch.Tensor:
        x = logits.float()
        greedy = x.argmax(dim=-1)
        w = warp_rows(x, self.temperature, self.top_k, self.top_p)
        sampled = (w + gumbel_from_uniform(self.uniform[:, i])).argmax(dim=-1)
        return torch.where(self.do_sample, sampled, greedy)

    def _loop(self):
        h = self.small_to_mtp(self.input_buf)
        out = self.pred_model(inputs_embeds=h, attention_mask=self.prefill_attn, past_key_values=self.cache,
                              cache_position=self.prefill_pos, use_cache=True)
        h = out.last_hidden_state
        tok = self._sample(self.lm_heads[0](h[:, -1, :]), 0)
        self.out_tokens[:, 0].copy_(tok)
        for i in range(1, self.num_cb):
            emb = self.small_to_mtp(self.codec_embeds[i - 1](tok.unsqueeze(1)))
            out = self.pred_model(inputs_embeds=emb, attention_mask=self.decode_attn[i - 1], past_key_values=self.cache,
                                  cache_position=self.decode_pos[i - 1], use_cache=True)
            tok = self._sample(self.lm_heads[i](out.last_hidden_state[:, -1, :]), i)
            self.out_tokens[:, i].copy_(tok)

    @torch.inference_mode()
    def capture(self, num_warmup: int = 3):
        if not self.use_graph:
            return
        self.graph = _capture(self._loop, self.device, num_warmup, pre=self.cache.reset)

    def set_row_params(self, row: int, temperature: float, top_k: int, top_p: float, do_sample: bool):
        self.temperature[row] = float(temperature)
        self.top_k[row] = int(top_k)
        self.top_p[row] = float(top_p)
        self.do_sample[row] = bool(do_sample)

    @torch.inference_mode()
    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        """pred_input [B, 2, H]; caller fills self.uniform first. Returns a clone of [B, 15] tokens."""
        self.input_buf.copy_(pred_input)
        for c in self._cum:  # rewind (keys need no zeroing: every read position is re-written first)
            c.zero_()
        if self.use_graph:
            self.graph.replay()
        else:
            self._loop()
        return self.out_tokens.clone()

