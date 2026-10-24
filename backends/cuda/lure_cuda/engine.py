"""Continuous-batching decode engine for the Lure CUDA backend.

One background thread owns the talker/predictor graphs and steps all active slots together. Requests are admitted
between steps: the prefill runs eagerly at batch 1 (HF forward, dynamic cache), its KV is copied into a free slot of the
batched static cache, and from then on the slot advances with every graph replay. Finished / cancelled slots free up
immediately. Generated frames go to a per-request queue as CPU int lists; codec decoding happens in the consumer.

Per step (B = max_batch, every slot computed, inactive rows ignored):
  codes 1-15  = predictor(past_hidden, C(cb0))                 sub-talker sampling per row
  next input  = C(cb0) + sum CP_EMB(codes) + text_in             text_in = T(trailing[k]) for frame k, then T(tts_pad)
  hidden      = talker step at write position p
  cb0 next    = sample(rep-penalty, suppress, min_new EOS block, temperature/top-k/top-p)
One device->host copy per step ([B, 17] longs) drives EOS / max-frame checks, as the upstream loop's token.item().
The engine is mode-agnostic: a request is its prefill embeddings plus its trailing text rows, so prompts of every mode and
layout share the batch. In the non_streaming layouts the trailing text is tts_pad only; in the streaming layouts it
carries the rest of the text and then tts_eos, one row per generated frame.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .graphs import BatchedPredictorGraph, BatchedTalkerGraph
from .sampling import SamplingParams, apply_repetition_penalty_rows, mask_rows, sample_rows, suppress_mask

logger = logging.getLogger(__name__)

_END = object()


@dataclass
class Request:
    prefill: torch.Tensor  # [1, L, H]
    tts_pad: torch.Tensor  # [1, 1, H]
    sampling: SamplingParams
    max_new_tokens: int
    generator: torch.Generator
    trailing: Optional[torch.Tensor] = None  # [1, N, H]: text side of generated frame k < N; None = tts_pad only
    out: "queue.Queue" = field(default_factory=queue.Queue)
    cancelled: bool = False
    # filled by the engine
    frames: int = 0
    ended_by_eos: bool = False
    stop_reason: str = ""
    t_submit: float = field(default_factory=time.perf_counter)
    t_first_frame: Optional[float] = None
    t_done: Optional[float] = None
    error: Optional[BaseException] = None

    def cancel(self):
        self.cancelled = True

    @property
    def frame_cap(self) -> int:
        """qwen-tts / HF semantics: max_new_tokens counts sampled codebook-0 tokens; the last sampled token never gets
        its sub-talker frame, so a run without EOS yields max_new_tokens - 1 frames."""
        return max(1, self.max_new_tokens - 1)


class _Slot:
    __slots__ = ("req", "ctx", "logical")

    def __init__(self, req: Request, ctx: int, logical: int):
        self.req, self.ctx, self.logical = req, ctx, logical  # ctx = keys held; logical = next rope position


class BatchEngine:
    def __init__(self, talker, talker_config, max_batch: int = 8, max_seq_len: int = 4096, device="cuda",
                 dtype=torch.bfloat16, use_graphs: bool = True, turbo_head=None):
        self.talker = talker
        self.cfg = talker_config
        self.B, self.max_seq_len, self.device, self.dtype = max_batch, max_seq_len, device, dtype
        self.eos = int(talker_config.codec_eos_token_id)
        self.vocab = int(talker_config.vocab_size)
        self.ncg = int(talker_config.num_code_groups)
        pred = talker.code_predictor
        self.codec_embed = talker.get_input_embeddings()
        self.codec_head = talker.codec_head
        self.pred_embeds = pred.get_input_embeddings()
        self.tgraph = BatchedTalkerGraph(talker.model, talker_config, max_batch, max_seq_len, device, dtype, use_graphs)
        self.turbo = turbo_head is not None
        if self.turbo:  # Turbo head: 3 group passes instead of 15 sub-talker steps
            from .turbo import TurboPredictorGraph
            self.pgraph = TurboPredictorGraph(turbo_head, talker_config.hidden_size, max_batch, device, dtype, use_graphs)
        else:
            self.pgraph = BatchedPredictorGraph(pred, pred.model.config, talker_config.hidden_size, max_batch, device, dtype,
                                                use_graphs)
        H = talker_config.hidden_size
        self.tokens = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.past_hidden = torch.zeros(max_batch, 1, H, dtype=dtype, device=device)
        self.text_in = torch.zeros(max_batch, 1, H, dtype=dtype, device=device)  # text side of each row's next input
        self.seen = torch.zeros(max_batch, self.vocab, dtype=torch.bool, device=device)
        self.uniform0 = torch.full((max_batch, self.vocab), 0.5, dtype=torch.float32, device=device)
        self.temperature = torch.full((max_batch,), 0.9, dtype=torch.float32, device=device)
        self.top_k = torch.full((max_batch,), 50, dtype=torch.long, device=device)
        self.top_p = torch.ones(max_batch, dtype=torch.float32, device=device)
        self.do_sample = torch.ones(max_batch, dtype=torch.bool, device=device)
        self.rep = torch.ones(max_batch, dtype=torch.float32, device=device)
        self.min_new = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.nframes = torch.zeros(max_batch, dtype=torch.long, device=device)
        self.suppress = suppress_mask(self.vocab, self.eos, device)
        self.slots: List[Optional[_Slot]] = [None] * max_batch
        self.p = 0  # shared write position
        self.pending: "queue.Queue[Request]" = queue.Queue()
        self._head: Optional[Request] = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self.steps = 0
        self.rebases = 0

    # ---------------------------------------------------------------------------------------------- lifecycle
    def capture(self):
        self.pgraph.capture()
        self.tgraph.capture()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="lure-cuda-engine", daemon=True)
            self._thread.start()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def submit(self, req: Request) -> Request:
        L = req.prefill.shape[1]
        if L + req.max_new_tokens + 1 > self.max_seq_len:
            raise ValueError(f"prefill {L} + max_new_tokens {req.max_new_tokens} exceeds max_seq_len {self.max_seq_len}; "
                             "shorten the segment / reference or load with a larger max_seq_len")
        self.pending.put(req)
        with self._cv:
            self._cv.notify_all()
        return req

    @property
    def active(self) -> int:
        return sum(s is not None for s in self.slots)

    # ---------------------------------------------------------------------------------------------- scheduling
    def _need(self, s: _Slot) -> int:
        return s.req.frame_cap - s.req.frames + 1

    def _try_admit(self, req: Request) -> bool:
        """Admit req into a free slot at the current write position, rebasing all slots if needed. False = wait."""
        free = [i for i, s in enumerate(self.slots) if s is None]
        if not free:
            return False
        L = req.prefill.shape[1]
        live = [s for s in self.slots if s is not None]
        need = max([self._need(s) for s in live] + [req.frame_cap + 1])
        p = self.p
        if not live:
            p = L
        elif L > p or p + need > self.max_seq_len:
            ctx = max([s.ctx for s in live] + [L])
            if ctx + need > self.max_seq_len:
                return False
            self._rebase(ctx)
            p = self.p
        self.p = p
        self._prefill_into(free[0], req)
        return True

    def _rebase(self, new_p: int):
        old = self.p
        if new_p == old:
            return
        for b, s in enumerate(self.slots):
            if s is None:
                continue
            self.tgraph.move_slot(b, old, new_p, s.ctx)
        self.p = new_p
        self.rebases += 1
        self._sync_slot_buffers()

    def _sync_slot_buffers(self):
        start = torch.zeros(self.B, dtype=torch.long)
        delta = torch.zeros(self.B, dtype=torch.long)
        for b, s in enumerate(self.slots):
            if s is not None:
                start[b] = self.p - s.ctx
                delta[b] = s.logical - self.p
        self.tgraph.start.copy_(start)
        self.tgraph.rope_delta.copy_(delta)

    @torch.inference_mode()
    def _prefill_into(self, b: int, req: Request):
        sp = req.sampling
        talker = self.talker
        talker.rope_deltas = None
        L = req.prefill.shape[1]
        out = talker.forward(
            inputs_embeds=req.prefill,
            attention_mask=torch.ones(1, L, dtype=torch.long, device=req.prefill.device),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=req.trailing if req.trailing is not None else req.tts_pad,
            tts_pad_embed=req.tts_pad,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )
        self.tgraph.write_prefill(b, out.past_key_values, self.p)
        rd = getattr(talker, "rope_deltas", None)
        rd = int(rd.reshape(-1)[0].item()) if isinstance(rd, torch.Tensor) else 0
        self.slots[b] = _Slot(req, ctx=L, logical=L + rd)
        self._sync_slot_buffers()
        # per-row parameters
        self.temperature[b] = sp.temperature
        self.top_k[b] = sp.top_k
        self.top_p[b] = sp.top_p
        self.do_sample[b] = sp.do_sample
        self.rep[b] = sp.repetition_penalty
        self.min_new[b] = sp.min_new_tokens
        self.nframes[b] = 0
        self.seen[b].zero_()
        self.pgraph.set_row_params(b, sp.subtalker_temperature, sp.subtalker_top_k, sp.subtalker_top_p,
                                   sp.subtalker_do_sample)
        self._set_text_in(b, req, 0)
        self.past_hidden[b].copy_(out.past_hidden[0])
        # first codebook-0 token (no history yet; EOS blocked if min_new_tokens > 0)
        logits = out.logits[:, -1, :].float()
        self.uniform0[b].uniform_(generator=req.generator)
        row = slice(b, b + 1)
        logits = mask_rows(logits, self.suppress, self.eos, (self.min_new[row] > 0))
        tok = sample_rows(logits, temperature=self.temperature[row], top_k=self.top_k[row], top_p=self.top_p[row],
                          do_sample=self.do_sample[row], uniform=self.uniform0[row])
        self.tokens[b] = tok[0]
        if int(tok[0].item()) == self.eos:  # only possible with min_new_tokens == 0
            self._finish(b, eos=True)

    def _set_text_in(self, b: int, req: Request, k: int):
        """Text side of row b's next input: trailing[k] while it lasts, then tts_pad (qwen-tts generation_step)."""
        if k < _trailing_len(req):
            self.text_in[b].copy_(req.trailing[0, k:k + 1])
        else:
            self.text_in[b].copy_(req.tts_pad[0])

    def _finish(self, b: int, eos: bool = False, reason: str = ""):
        s = self.slots[b]
        if s is None:
            return
        r = s.req
        r.ended_by_eos = eos
        r.stop_reason = "eos" if eos else (reason or "max_new_tokens")
        r.t_done = time.perf_counter()
        self.slots[b] = None
        r.out.put(_END)

    def _admit_pending(self):
        while True:
            if self._head is None:
                try:
                    self._head = self.pending.get_nowait()
                except queue.Empty:
                    return
            if self._head.cancelled:
                self._head.stop_reason = "cancelled"
                self._head.out.put(_END)
                self._head = None
                continue
            try:
                ok = self._try_admit(self._head)
            except BaseException as e:  # noqa: BLE001 - report to the request, keep the engine alive
                logger.exception("prefill failed")
                self._head.error = e
                self._head.out.put(_END)
                self._head = None
                continue
            if not ok:
                return  # FIFO: wait for slots / capacity
            self._head = None

    # ---------------------------------------------------------------------------------------------- stepping
    @torch.inference_mode()
    def step(self):
        B = self.B
        act = [b for b, s in enumerate(self.slots) if s is not None]
        for b in act:
            r = self.slots[b].req
            self.pgraph.uniform[b].uniform_(generator=r.generator)
        last = self.codec_embed(self.tokens.view(B, 1))  # [B, 1, H]
        codes = self.pgraph.run(torch.cat((self.past_hidden, last), dim=1))  # [B, 15]
        hid = [last] + [self.pred_embeds[i](codes[:, i:i + 1]) for i in range(self.ncg - 1)]
        x = torch.cat(hid, dim=1).sum(1, keepdim=True) + self.text_in
        h = self.tgraph.run(x, self.p)
        logits = self.codec_head(h[:, -1, :]).float()  # [B, V]
        frame = torch.cat([self.tokens.view(B, 1), codes], dim=1)  # [B, 16]
        self.seen.scatter_(1, self.tokens.view(B, 1), True)
        self.nframes += 1
        logits = apply_repetition_penalty_rows(logits, self.seen, self.rep)
        logits = mask_rows(logits, self.suppress, self.eos, self.nframes < self.min_new)
        for b in act:
            self.uniform0[b].uniform_(generator=self.slots[b].req.generator)
        nxt = sample_rows(logits, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
                          do_sample=self.do_sample, uniform=self.uniform0)
        host = torch.cat([frame, nxt.view(B, 1)], dim=1).cpu().tolist()  # the one sync per step
        self.past_hidden.copy_(h[:, -1:, :])
        self.tokens.copy_(nxt)
        self.p += 1
        self.steps += 1
        now = time.perf_counter()
        for b in act:
            s = self.slots[b]
            r = s.req
            s.ctx += 1
            s.logical += 1
            r.frames += 1
            if r.frames <= _trailing_len(r):  # still in the trailing text, or the step that switches to tts_pad
                self._set_text_in(b, r, r.frames)
            if r.t_first_frame is None:
                r.t_first_frame = now
            r.out.put(host[b][:16])
            if r.cancelled:
                self._finish(b, reason="cancelled")
            elif host[b][16] == self.eos:
                self._finish(b, eos=True)
            elif r.frames >= r.frame_cap:
                self._finish(b, reason="max_new_tokens")
            elif self.p >= self.max_seq_len - 1:
                self._finish(b, reason="max_seq_len")

    def _run(self):
        torch.set_grad_enabled(False)
        if str(self.device).startswith("cuda"):
            torch.cuda.set_device(torch.device(self.device))
        while True:
            with self._cv:
                while not self._stop and self.active == 0 and self.pending.empty() and self._head is None:
                    self._cv.wait(timeout=0.5)
                if self._stop:
                    break
            try:
                self._admit_pending()
                if self.active:
                    self.step()
                else:
                    time.sleep(0.0005)
            except BaseException as e:  # noqa: BLE001
                logger.exception("engine step failed; failing active requests")
                for b, s in enumerate(self.slots):
                    if s is not None:
                        s.req.error = e
                        self._finish(b, reason="error")
        for b, s in enumerate(self.slots):
            if s is not None:
                s.req.error = RuntimeError("engine stopped")
                self._finish(b, reason="stopped")

    # ---------------------------------------------------------------------------------------------- sync helper
    def run_until_idle(self, max_steps: int = 1_000_000):
        """Synchronous driver (tests / single-threaded use): admit + step until nothing is left."""
        for _ in range(max_steps):
            self._admit_pending()
            if not self.active and self._head is None and self.pending.empty():
                return
            if self.active:
                self.step()
        raise RuntimeError("run_until_idle: step budget exhausted")


def _trailing_len(req: Request) -> int:
    return 0 if req.trailing is None else int(req.trailing.shape[1])


def drain(req: Request, timeout: Optional[float] = None):
    """Yield frames ([16] int lists) of a request until it ends. Raises the engine error, if any."""
    while True:
        item = req.out.get(timeout=timeout)
        if item is _END:
            if req.error is not None:
                raise req.error
            return
        yield item
