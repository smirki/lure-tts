"""Backend interface. Every backend (mlx, cuda, cpp, fake) implements `Backend`.

A backend package exposes a class with this shape and registers it either with an entry point
    [project.entry-points."lure.backends"]  mlx = "lure_mlx:MLXBackend"
or by living at the conventional import path (lure_mlx:MLXBackend, lure_cuda:CudaBackend, lure_cpp:CppBackend).

The core owns model inspection (lure_tts.model.ModelSpec), text segmentation, prompt ids (PromptIds, see lure_tts.prompt),
defaults and the voice cache. A backend receives ONE segment per generate() call, plus the core-built PromptIds and the
prompt mode / layout / speaker for it (tokenizing again is allowed, but must give the same ids). The full contract is in
docs/architecture.md ("Backend contract").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol, runtime_checkable

import numpy as np

from ..defaults import Sampling
from ..prompt import ModelIds, Prompt, PromptIds, build_prompt


@dataclass
class Voice:
    """Cached voice prompt. ref_codes: int32 [T_ref, 16] (12.5 Hz codec frames of the reference wav, codebook-major per frame).
    spk_emb: float32 [enc_dim] speaker-encoder x-vector (1024 on 0.6B, 2048 on 1.7B). Both are backend-portable (same
    model, same encoder). ref_text "" marks an x-vector-only voice (no transcript, mode xvec)."""
    voice_id: str
    ref_text: str
    ref_codes: np.ndarray
    spk_emb: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_ref_frames(self) -> int:
        return int(self.ref_codes.shape[0])

    @property
    def has_transcript(self) -> bool:
        return bool(self.ref_text)


@dataclass
class SegmentInfo:
    frames: int  # generated codec frames (EOS frame excluded)
    ended_by_eos: bool  # False = hit max_new_tokens
    samples: int = 0
    ttfa_s: Optional[float] = None  # time to first audio chunk of this segment
    wall_s: Optional[float] = None
    extra: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """float32 mono PCM @ 24 kHz. The last chunk of a segment carries info (its pcm may be empty)."""
    pcm: np.ndarray
    info: Optional[SegmentInfo] = None

    @property
    def final(self) -> bool:
        return self.info is not None


@dataclass
class GenerateRequest:
    """One segment. mode: icl | xvec | custom_voice | voice_design; layout: non_streaming | streaming.
    voice is set for icl / xvec and None otherwise; speaker / speaker_id (preset name and its codec id) only for
    custom_voice. prompt.ref_ids is None outside icl."""
    text: str
    voice: Optional[Voice]
    prompt: PromptIds
    instruct: Optional[str] = None
    sampling: Sampling = field(default_factory=Sampling)
    seed: Optional[int] = None
    max_new_tokens: int = 2048
    stream: bool = True
    chunk_frames: int = 4  # ~0.32 s
    first_chunk_frames: int = 4
    language: str = "english"
    mode: str = "icl"
    layout: str = "non_streaming"
    speaker: Optional[str] = None
    speaker_id: Optional[int] = None

    def build_prompt(self, ids: ModelIds) -> Prompt:
        """The normative prefill + trailing rows for this request (lure_tts.prompt.build_prompt)."""
        n_ref = self.voice.n_ref_frames if self.mode == "icl" and self.voice is not None else 0
        return build_prompt(self.prompt, ids, mode=self.mode, layout=self.layout, language=self.language,
                            n_ref_frames=n_ref, speaker=self.speaker)


@runtime_checkable
class Backend(Protocol):
    name: str
    sample_rate: int  # 24000

    @classmethod
    def load(cls, model_dir: str, quant: Optional[str] = None, **opts) -> "Backend":
        """opts carries spec=ModelSpec (lure_tts.model.inspect, already resolved by the core) plus backend options."""
        ...

    def encode_voice(self, ref_wav: np.ndarray, ref_text: str) -> tuple[np.ndarray, np.ndarray]:
        """Base models only. ref_wav: float32 mono @ 24 kHz -> (ref_codes int32 [T,16], spk_emb float32 [enc_dim]).
        ref_text may be "" (x-vector-only voice)."""
        ...

    def generate(self, req: GenerateRequest) -> Iterator[Chunk]:
        """Yield PCM chunks for ONE segment; the final chunk has .info set. stream=False may yield a single chunk."""
        ...

    def close(self) -> None: ...
