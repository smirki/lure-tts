"""Test backend: no model. Emits a quiet tone, ~3 frames per word, in the requested chunk sizes. Records every request
and the prompt it would run (req.build_prompt), so tests see the mode / layout / speaker the core asked for."""
from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np

from ..defaults import SAMPLE_RATE, SAMPLES_PER_FRAME
from .base import Chunk, GenerateRequest, SegmentInfo


class FakeBackend:
    name = "fake"
    sample_rate = SAMPLE_RATE

    def __init__(self, model_dir: str = "", spec=None, **opts):
        self.model_dir = model_dir
        self.spec = spec
        self.opts = opts
        self.requests: list[GenerateRequest] = []
        self.prompts: list = []
        self.concurrent = bool(opts.get("concurrent", False))  # test the core's one-thread-per-stream path
        self.delay_s = float(opts.get("delay_s", 0.0))  # per chunk, to make overlap observable
        self.active, self.max_active = 0, 0

    @classmethod
    def load(cls, model_dir: str, quant: Optional[str] = None, **opts) -> "FakeBackend":
        return cls(model_dir, quant=quant, **opts)

    def encode_voice(self, ref_wav: np.ndarray, ref_text: str):
        t = max(1, int(np.ceil(len(ref_wav) / SAMPLES_PER_FRAME)))
        codes = (np.arange(t * 16, dtype=np.int64).reshape(t, 16) * 7919 % 2048).astype(np.int32)
        dim = self.spec.speaker_encoder_dim if self.spec is not None and self.spec.speaker_encoder_dim else 1024
        return codes, np.zeros(dim, dtype=np.float32)

    def generate(self, req: GenerateRequest) -> Iterator[Chunk]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            yield from self._generate(req)
        finally:
            self.active -= 1

    def _generate(self, req: GenerateRequest) -> Iterator[Chunk]:
        self.requests.append(req)
        if self.spec is not None:
            self.prompts.append(req.build_prompt(self.spec.ids))
        t0 = time.time()
        frames = min(req.max_new_tokens, 3 * max(1, len(req.text.split())) + 2)
        rng = np.random.default_rng(req.seed)
        f0 = 180.0 + 40.0 * rng.random()
        n = frames * SAMPLES_PER_FRAME
        pcm = (0.05 * np.sin(2 * np.pi * f0 * np.arange(n) / SAMPLE_RATE)).astype(np.float32)
        ttfa = None
        if not req.stream:
            yield Chunk(pcm, SegmentInfo(frames, frames < req.max_new_tokens, n, time.time() - t0, time.time() - t0))
            return
        pos, step = 0, req.first_chunk_frames
        while pos < frames:
            k = min(step, frames - pos)
            if ttfa is None:
                ttfa = time.time() - t0
            if self.delay_s:
                time.sleep(self.delay_s)
            yield Chunk(pcm[pos * SAMPLES_PER_FRAME:(pos + k) * SAMPLES_PER_FRAME])
            pos += k
            step = req.chunk_frames
        yield Chunk(np.zeros(0, np.float32), SegmentInfo(frames, frames < req.max_new_tokens, n, ttfa, time.time() - t0))

    def dump_layout(self, text, ref_text, direction, n_ref_frames, language="english", mode="icl",
                    layout="non_streaming", speaker=None):
        """The prompt-check hook (conformance/runner.py), answered from the core builder."""
        from ..prompt import build_prompt, build_prompt_ids
        from ..tokenizer import QwenTokenizer
        tok = QwenTokenizer.from_dir(self.model_dir)
        p = build_prompt_ids(tok, text, ref_text if mode == "icl" else None, direction)
        pr = build_prompt(p, self.spec.ids, mode=mode, layout=layout, language=language, n_ref_frames=n_ref_frames,
                          speaker=speaker)
        return {"layout": pr.prefill, "trailing": pr.trailing}

    def close(self):
        pass
