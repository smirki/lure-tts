"""Public API.

    from lure_tts import Lure
    tts = Lure.load("/path/to/model")                  # backend auto: mlx (Apple silicon) > cuda > cpp
    print(tts.spec.variant, tts.spec.speakers)          # what the checkpoint is (lure_tts.model.ModelSpec)
    v = tts.make_voice("ref.wav", "ref transcript", name="willow")       # Base models: clone
    for pcm in tts.stream("(sigh) Fine. I'm in.", v, instruct="Tired, then a smile."):
        play(pcm)                                       # float32 mono 24 kHz chunks
    audio = tts.speak("Hello there.", "willow")        # whole clip (still generated segment by segment)

    tts.speak("Hi.", speaker="vivian", instruct="Cheerful.")   # CustomVoice: preset speaker
    tts.speak("Hi.", instruct="A deep, calm male voice.")     # VoiceDesign: the instruct is the voice

    live = tts.live(v)                                  # incremental text, e.g. LLM tokens
    for delta in llm: [play(p) for p in live.feed(delta)]
    [play(p) for p in live.flush()]
"""
from __future__ import annotations

import os
from typing import Iterator, Optional, Union

import numpy as np

from .audio import load_audio
from .backends import load_backend, select
from .backends.base import SegmentInfo, Voice
from .defaults import SAMPLE_RATE, Sampling
from .model import ModelSpec, inspect
from .prompt import normalize_layout
from .scheduler import BackendWorker, Event, StreamOptions, run_segments
from .text import Segment, Segmenter, segment_text
from .tokenizer import QwenTokenizer
from .voice import VoiceCache, model_fingerprint

VoiceLike = Union[Voice, str]


class ModeError(ValueError):
    """The requested voice / speaker / instruct / language does not fit the loaded model."""


def default_model_dir() -> Optional[str]:
    return os.environ.get("LURE_MODEL")


class Lure:
    def __init__(self, backend, backend_name: str, model_dir: str, cache_dir: Optional[str], worker: BackendWorker,
                 spec: Optional[ModelSpec] = None):
        self.backend, self.backend_name, self.model_dir = backend, backend_name, model_dir
        self.worker = worker
        self.spec = spec or inspect(model_dir)
        self.tokenizer = QwenTokenizer.from_dir(model_dir)
        self.model_ids = self.spec.ids
        self.voices = VoiceCache(cache_dir)
        self.fingerprint = model_fingerprint(model_dir)
        self.sample_rate = SAMPLE_RATE
        self.last_infos: list[SegmentInfo] = []
        self.turbo = bool(getattr(backend, "turbo", False))  # Lure Turbo head active (lure_tts.turbo)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_dir: Optional[str] = None, backend: str = "auto", quant: Optional[str] = None,
             cache_dir: Optional[str] = None, layout: Optional[str] = None, inline_tags: Optional[bool] = None,
             **backend_opts) -> "Lure":
        """layout ("non_streaming" | "streaming") and inline_tags override the model's defaults (see lure_tts.model)."""
        model_dir = model_dir or default_model_dir()
        if not model_dir or not os.path.isdir(model_dir):
            raise FileNotFoundError(f"model dir not found: {model_dir!r} (pass model_dir or set LURE_MODEL)")
        spec = inspect(model_dir, layout=layout, inline_tags=inline_tags, turbo=backend_opts.get("turbo"),
                       turbo_head=backend_opts.get("turbo_head"))
        name = select(backend)
        worker = BackendWorker()
        be = worker.call(load_backend, name, model_dir, quant, spec=spec, **backend_opts)
        worker.concurrent = bool(getattr(be, "concurrent", False))
        return cls(be, name, model_dir, cache_dir, worker, spec)

    def close(self):
        try:
            self.worker.call(self.backend.close)
        finally:
            self.worker.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ voices
    def make_voice(self, ref_wav, ref_text: Optional[str] = None, name: Optional[str] = None) -> Voice:
        """Base models only. ref_wav: path, bytes or (array, sr). With ref_text the voice clones in icl mode; without it
        the voice is x-vector only. Cached on disk; re-encoded only if wav/ref_text/model changed."""
        if not self.spec.is_clone:
            raise ModeError(f"{self.spec.name} is a {self.spec.variant} model and cannot clone a voice; "
                            + self._mode_hint())
        wav = load_audio(ref_wav)
        return self.worker.call(self.voices.get_or_make, name, wav, (ref_text or "").strip(), self.backend.encode_voice,
                                self.backend_name, self.fingerprint)

    def voice(self, v: VoiceLike) -> Voice:
        voice = v if isinstance(v, Voice) else self.voices.load(v)
        dim = self.spec.speaker_encoder_dim
        if dim and voice.spk_emb.size != dim:
            raise ModeError(f"voice {voice.voice_id!r} has a {voice.spk_emb.size}-d speaker embedding but {self.spec.name} "
                            f"expects {dim}; it was made with another model, re-create it with make_voice")
        return voice

    # ------------------------------------------------------------------ modes
    def _mode_hint(self) -> str:
        s = self.spec
        if s.variant == "custom_voice":
            return f"pass speaker= one of {list(s.speakers)}"
        if s.variant == "voice_design":
            return "pass instruct= describing the voice (no voice or speaker)"
        return "pass voice= (a cached voice name or a Voice from make_voice)"

    def resolve_mode(self, voice: Optional[VoiceLike] = None, speaker: Optional[str] = None,
                     instruct: Optional[str] = None, language: Optional[str] = None,
                     x_vector_only: bool = False) -> tuple[Optional[Voice], dict]:
        """Check the request against the model and return (voice, StreamOptions fields). Raises ModeError."""
        s = self.spec
        instruct = instruct.strip() if instruct and instruct.strip() else None
        if voice is not None and speaker is not None:
            raise ModeError("pass voice= (clone) or speaker= (preset), not both")
        v, spk, spk_id = None, None, None
        if s.variant == "base":
            if speaker is not None:
                raise ModeError(f"{s.name} is a Base model with no preset speakers; " + self._mode_hint())
            if voice is None:
                raise ModeError(f"{s.name} is a Base model and needs a voice to clone; " + self._mode_hint())
            v = self.voice(voice)
            mode = "xvec" if x_vector_only or not v.has_transcript else "icl"
        elif s.variant == "custom_voice":
            if voice is not None:
                raise ModeError(f"{s.name} does not clone voices; " + self._mode_hint())
            if speaker is None:
                raise ModeError(f"{s.name} needs a preset speaker; " + self._mode_hint())
            if speaker.lower() not in s.speakers:
                raise ModeError(f"unknown speaker {speaker!r}; {s.name} has {list(s.speakers)}")
            mode, spk = "custom_voice", speaker.lower()
            spk_id = s.ids.speaker_id(spk)
        else:
            if voice is not None or speaker is not None:
                raise ModeError(f"{s.name} is a VoiceDesign model: the voice comes from instruct, so drop voice= / speaker=")
            mode = "voice_design"
        if instruct and s.instruct == "none":
            why = ("qwen-tts drops instruct on the 0.6B CustomVoice" if s.variant == "custom_voice"
                   else "stock Qwen3-TTS Base takes no instruct (qwen-tts generate_voice_clone has none)")
            raise ModeError(f"{s.name} does not accept instruct: {why}")
        if not instruct and s.instruct == "required":
            raise ModeError(f"{s.name} needs instruct= describing the voice")
        lang = (language or s.default_language).strip().lower()
        if lang not in s.languages:
            raise ModeError(f"language {language!r} not supported by {s.name}: {list(s.languages)}")
        return v, dict(instruct=instruct, mode=mode, layout=s.layout, speaker=spk, speaker_id=spk_id, language=lang)

    # ------------------------------------------------------------------ synthesis
    def _options(self, voice, instruct, sampling, seed, chunk_s, speaker=None, language=None, x_vector_only=False,
                 layout=None, **kw) -> tuple[Optional[Voice], StreamOptions]:
        if isinstance(sampling, dict):
            sampling = Sampling.from_dict(sampling)
        v, fields = self.resolve_mode(voice, speaker, instruct, language, x_vector_only)
        if layout:
            fields["layout"] = normalize_layout(layout)
        o = StreamOptions(sampling=sampling or self.spec.sampling, seed=seed, **fields)
        if chunk_s:
            o.chunk_s = chunk_s
        for k, val in kw.items():
            if val is not None:
                setattr(o, k, val)
        return v, o

    def segmenter(self, **kw) -> Segmenter:
        return Segmenter(tags=self.spec.inline_tags, **kw)

    @property
    def concurrent(self) -> bool:
        """True when the backend accepts overlapping generate() calls (one producer thread per stream)."""
        return self.worker.concurrent

    def events(self, segments, voice: Optional[Voice], opt: StreamOptions, infos: Optional[list] = None) -> Iterator[Event]:
        """voice: the resolved Voice (None outside clone modes). infos: optional per-call list that receives each
        SegmentInfo (use it instead of last_infos when streams overlap)."""
        self.last_infos = mine = []

        def make(cancelled):
            return run_segments(self.backend, self.tokenizer, segments, voice, opt, cancelled)

        for ev in self.worker.stream(make):
            if ev.info is not None:
                mine.append(ev.info)
                if infos is not None:
                    infos.append(ev.info)
            yield ev

    def stream(self, text: str, voice: Optional[VoiceLike] = None, instruct: Optional[str] = None, sampling=None,
               seed: Optional[int] = None, chunk_s: Optional[float] = None, infos: Optional[list] = None, *,
               speaker: Optional[str] = None, language: Optional[str] = None, **opts) -> Iterator[np.ndarray]:
        """voice= clones (Base), speaker= picks a preset (CustomVoice), neither runs VoiceDesign. Extra opts: layout,
        x_vector_only, stream, first_chunk_frames, pauses."""
        v, opt = self._options(voice, instruct, sampling, seed, chunk_s, speaker=speaker, language=language, **opts)
        segs = segment_text(text, tags=self.spec.inline_tags)
        for ev in self.events(segs, v, opt, infos):
            if len(ev.pcm):
                yield ev.pcm

    def speak(self, text: str, voice: Optional[VoiceLike] = None, instruct: Optional[str] = None, sampling=None,
              seed: Optional[int] = None, out: Optional[str] = None, infos: Optional[list] = None, *,
              speaker: Optional[str] = None, language: Optional[str] = None, **opts) -> np.ndarray:
        parts = list(self.stream(text, voice, instruct, sampling, seed, infos=infos, speaker=speaker, language=language,
                                 **opts))
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        if out:
            from .audio import write_wav
            write_wav(out, audio)
        return audio

    def live(self, voice: Optional[VoiceLike] = None, instruct: Optional[str] = None, sampling=None,
             seed: Optional[int] = None, *, speaker: Optional[str] = None, language: Optional[str] = None,
             **seg_kw) -> "LiveSession":
        v, opt = self._options(voice, instruct, sampling, seed, None, speaker=speaker, language=language)
        return LiveSession(self, v, opt, self.segmenter(**seg_kw))


class LiveSession:
    """Incremental text in, audio out. Each committed segment is generated with the full segment text up front."""

    def __init__(self, tts: Lure, voice: Optional[Voice], opt: StreamOptions, seg: Segmenter):
        self.tts, self.v, self.opt, self.seg = tts, voice, opt, seg

    def _run(self, segs: list[Segment]) -> Iterator[np.ndarray]:
        if not segs:
            return
        for ev in self.tts.events(segs, self.v, self.opt):
            if len(ev.pcm):
                yield ev.pcm

    def feed(self, delta: str) -> Iterator[np.ndarray]:
        yield from self._run(self.seg.feed(delta))

    def flush(self) -> Iterator[np.ndarray]:
        yield from self._run(self.seg.flush())
