"""Codec (12 Hz speech tokenizer) decoding for the CUDA backend.

Only the icl mode decodes with the reference. xvec, custom_voice and voice_design decode the generated codes alone
(qwen-tts generate_voice_clone with x_vector_only_mode, generate_custom_voice, generate_voice_design): ref_codes=None
below, so there is no cut and the streaming windows take their left context from earlier generated frames only.

icl reference (qwen-tts generate_voice_clone): decode [ref_codes ; generated] with the
tokenizer's chunked_decode (300-frame chunks, 25 frames of left context) and drop the first
int(ref_len / total_len * n_samples) samples. The decoder is causal and emits exactly `upsample` (1920) samples per frame,
so that cut equals ref_len * 1920.

Streaming: every `chunk_frames` new frames (first chunk after `first_chunk_frames`) we decode
codes[a - ctx : b] of the same [ref ; generated] sequence and keep the samples of [a, b). ctx defaults to 72 frames (the
decoder transformer's sliding window), so the first chunk is primed with the tail of the reference codes like the full
decode is. Measured on an L4 at chunk joins, streamed vs full decode of the same frames: ctx 25 gives 9-15 dB SNR, ctx 72 gives
22-30 dB.
"""
from __future__ import annotations

import threading
from typing import Iterable, Iterator, List, Optional

import numpy as np
import torch


def _decoder_parts(speech_tokenizer):
    """(decoder module with forward(codes [B, Q, T]) -> wav [B, 1, T*up], samples per frame)."""
    inner = getattr(speech_tokenizer, "model", speech_tokenizer)
    dec = getattr(inner, "decoder", None)
    up = None
    for obj, name in ((inner, "get_decode_upsample_rate"), (speech_tokenizer, "get_decode_upsample_rate")):
        f = getattr(obj, name, None)
        if callable(f):
            try:
                up = int(f())
                break
            except Exception:
                pass
    if up is None:
        up = int(getattr(dec, "total_upsample", 1920))
    return dec, up


class CodecDecoder:
    """Thread-safe wrapper over the codec decoder (one decode at a time; decodes are short)."""

    def __init__(self, speech_tokenizer, device, lock: Optional[threading.Lock] = None):
        self.st = speech_tokenizer
        self.dec, self.upsample = _decoder_parts(speech_tokenizer)
        self.device = device
        self.lock = lock or threading.Lock()

    @torch.inference_mode()
    def decode_window(self, codes: torch.Tensor) -> np.ndarray:
        """codes [T, Q] long -> float32 [T * upsample] (decoder forward, no chunking)."""
        c = codes.to(self.device).clamp(min=0).t().unsqueeze(0)  # [1, Q, T]
        with self.lock:
            wav = self.dec(c)
        return wav.reshape(-1).float().cpu().numpy()

    @torch.inference_mode()
    def decode_full(self, ref_codes: Optional[torch.Tensor], gen_codes: torch.Tensor) -> np.ndarray:
        """Exactly the reference path: speech_tokenizer.decode([ref ; gen]) then the proportional cut."""
        g = gen_codes.to(self.device)
        allc = torch.cat([ref_codes.to(self.device), g], 0) if ref_codes is not None else g
        with self.lock:
            wavs, _sr = self.st.decode([{"audio_codes": allc}])
        w = wavs[0]
        w = w.detach().float().cpu().numpy() if isinstance(w, torch.Tensor) else np.asarray(w, dtype=np.float32)
        ref_len = 0 if ref_codes is None else ref_codes.shape[0]
        cut = int(ref_len / max(allc.shape[0], 1) * w.shape[0])
        return np.asarray(w[cut:], dtype=np.float32)


class StreamDecoder:
    """Incremental decode of one request's frames."""

    def __init__(self, codec: CodecDecoder, ref_codes: Optional[torch.Tensor], chunk_frames: int = 4,
                 first_chunk_frames: Optional[int] = None, context_frames: int = 72):
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be >= 1")
        self.codec = codec
        self.ref = ref_codes.detach().to("cpu").long() if ref_codes is not None else torch.zeros(0, 16, dtype=torch.long)
        self.n_ref = self.ref.shape[0]
        self.chunk = chunk_frames
        self.first = first_chunk_frames or chunk_frames
        self.ctx = context_frames
        self.gen: List[List[int]] = []
        self.emitted = 0  # generated frames already turned into audio

    def _window(self, a: int, b: int) -> np.ndarray:
        """Audio for generated frames [a, b)."""
        A = self.n_ref + a  # index into [ref ; gen]
        s = max(0, A - self.ctx)
        rows = []
        if s < self.n_ref:
            rows.append(self.ref[s:self.n_ref])
        lo = max(0, s - self.n_ref)
        rows.append(torch.tensor(self.gen[lo:b], dtype=torch.long))
        codes = torch.cat(rows, 0)
        wav = self.codec.decode_window(codes)
        return wav[(A - s) * self.codec.upsample:]

    def feed(self, frame: List[int]) -> Optional[np.ndarray]:
        self.gen.append(frame)
        need = self.first if self.emitted == 0 else self.chunk
        if len(self.gen) - self.emitted >= need:
            a, b = self.emitted, len(self.gen)
            self.emitted = b
            return self._window(a, b)
        return None

    def flush(self) -> Optional[np.ndarray]:
        if len(self.gen) > self.emitted:
            a, b = self.emitted, len(self.gen)
            self.emitted = b
            return self._window(a, b)
        return None


def stream_audio(frames: Iterable[List[int]], dec: StreamDecoder) -> Iterator[np.ndarray]:
    for f in frames:
        out = dec.feed(f)
        if out is not None and out.size:
            yield out
    out = dec.flush()
    if out is not None and out.size:
        yield out
