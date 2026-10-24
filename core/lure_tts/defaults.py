"""Core defaults (the qwen-tts defaults). Every backend uses these unless the model's generation_config.json or the
caller overrides them (docs/architecture.md)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

SAMPLE_RATE = 24000
FRAME_RATE = 12.5  # codec frames per second
SAMPLES_PER_FRAME = 1920  # 24000 / 12.5
LANGUAGE = "english"
STREAM_CHUNK_S = 0.32  # ~4 codec frames per streamed chunk
FIRST_CHUNK_FRAMES = 4
DTYPE = "bfloat16"


@dataclass(frozen=True)
class Sampling:
    """Talker (codebook 0) and sub-talker (codebooks 1-15) sampling. Identical to qwen-tts 0.1.1 defaults and to
    the checkpoint's generation_config.json. repetition_penalty applies to codebook 0 only, over ALL previously generated
    codebook-0 tokens of the segment (HF RepetitionPenaltyLogitsProcessor semantics, unique set, no window)."""
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    do_sample: bool = True
    subtalker_do_sample: bool = True
    min_new_tokens: int = 2  # EOS is masked for the first 2 frames (qwen-tts talker_kwargs)
    max_new_tokens: int | None = None  # None -> max_frames_for(text)

    def with_(self, **kw) -> "Sampling":
        kw = {k: v for k, v in kw.items() if v is not None}
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Sampling":
        if not d:
            return cls()
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown sampling fields: {sorted(unknown)}")
        return cls().with_(**d)


DEFAULT_SAMPLING = Sampling()


def max_frames_for(text: str) -> int:
    """Per-segment frame cap: 1.8x the expected duration at 2.6 words/s, plus 4 s."""
    words = max(1, len(text.split()))
    return int(FRAME_RATE * (1.8 * words / 2.6 + 4))
