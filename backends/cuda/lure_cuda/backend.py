"""CudaBackend: the core Backend protocol (lure_tts.backends.base) on NVIDIA GPUs.

    from lure_cuda import CudaBackend
    be = CudaBackend.load("/path/to/model", spec=spec)            # HF layout dir: config.json, model.safetensors, speech_tokenizer/
    codes, spk = be.encode_voice(wav24k_float32, "reference transcript")   # Base checkpoints only
    for chunk in be.generate(GenerateRequest(...)):                # core request object
        play(chunk.pcm); info = chunk.info                         # final chunk: frames, ended_by_eos, ttfa_s, ...

Checkpoints: every Qwen3-TTS 12Hz checkpoint (0.6B / 1.7B; Base, CustomVoice, VoiceDesign) and the Lure family.
Modes: icl and xvec (Base), custom_voice (CustomVoice), voice_design (VoiceDesign), each in the non_streaming and
streaming layouts. Only icl decodes with the reference codes as left context (and cuts their share); the other modes
decode the generated codes alone.

Convenience API for direct use (no core): `load()`, `make_voice(ref_wav, ref_text)`, `generate_segment(text, voice, ...)`.

Concurrency: `concurrent = True`. generate() only enqueues work for the engine thread and then reads a per-request queue,
so several generate() iterators may be consumed at the same time from different threads; they decode together in one
batched CUDA-graph step (up to max_batch). A core that serializes all backend calls on one worker gets batch 1.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Tuple

import numpy as np

from .prompt import VARIANT_MODES, normalize_layout

logger = logging.getLogger(__name__)

CLONE_MODES = ("icl", "xvec")

SAMPLE_RATE = 24000


# ------------------------------------------------------------------------------------- core types (with fallbacks)
try:  # the core package owns these; fall back to identical local shapes when used standalone
    from lure_tts.backends.base import Chunk, GenerateRequest, SegmentInfo  # type: ignore
    from lure_tts.backends.base import Voice as CoreVoice  # type: ignore
except Exception:  # pragma: no cover - exercised only without the core
    CoreVoice = None

    @dataclass
    class SegmentInfo:  # type: ignore[no-redef]
        frames: int
        ended_by_eos: bool
        samples: int = 0
        ttfa_s: Optional[float] = None
        wall_s: Optional[float] = None
        extra: dict = field(default_factory=dict)

    @dataclass
    class Chunk:  # type: ignore[no-redef]
        pcm: np.ndarray
        info: Optional[SegmentInfo] = None

        @property
        def final(self) -> bool:
            return self.info is not None

    GenerateRequest = None  # type: ignore[assignment]


@dataclass
class Voice:
    """Standalone voice (same fields as the core's lure_tts.backends.base.Voice)."""
    voice_id: str
    ref_text: str
    ref_codes: np.ndarray  # int32 [T_ref, 16]
    spk_emb: np.ndarray  # float32 [H]
    meta: dict = field(default_factory=dict)

    @property
    def n_ref_frames(self) -> int:
        return int(self.ref_codes.shape[0])


def _load_wav_24k(ref_wav) -> np.ndarray:
    """Path -> librosa.load(sr=24000, mono=True), as the reference pipeline loads voice prompts;
    (array, sr) -> resampled to 24 kHz; array -> assumed 24 kHz mono."""
    if isinstance(ref_wav, (str, os.PathLike)):
        import librosa

        x, _ = librosa.load(str(ref_wav), sr=24000, mono=True)
        return x.astype(np.float32)
    if isinstance(ref_wav, tuple):
        x, sr = ref_wav
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        if int(sr) != 24000:
            import librosa

            x = librosa.resample(x, orig_sr=int(sr), target_sr=24000)
        return x.astype(np.float32)
    x = np.asarray(ref_wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    return x


VARIANT_NAMES = {"base": "Base", "custom_voice": "CustomVoice", "voice_design": "VoiceDesign"}


def check_model(m, spec=None) -> str:
    """Validate the loaded qwen-tts model against itself and the core's ModelSpec. Returns the variant."""
    cfg = m.config
    tc = cfg.talker_config
    variant = getattr(m, "tts_model_type", None) or getattr(cfg, "tts_model_type", "base")
    if variant not in VARIANT_MODES:
        raise ValueError(f"unknown tts_model_type {variant!r}; expected one of {sorted(VARIANT_MODES)}")
    if int(tc.num_code_groups) != 16:
        raise ValueError(f"expected 16 code groups, got {tc.num_code_groups}")
    H = int(tc.hidden_size)
    enc = getattr(m, "speaker_encoder", None)
    if variant == "base":
        if enc is None:
            raise ValueError("a Base checkpoint must have a speaker encoder")
        enc_dim = int(cfg.speaker_encoder_config.enc_dim)
        if enc_dim != H:
            raise ValueError(f"speaker encoder dim {enc_dim} != talker hidden {H}: the x-vector row would not fit")
    if spec is not None:
        want = {"variant": (spec.variant, variant), "talker_hidden_size": (spec.talker_hidden_size, H),
                "code_predictor_hidden_size": (spec.code_predictor_hidden_size,
                                               int(tc.code_predictor_config.hidden_size))}
        if variant == "base":
            want["speaker_encoder_dim"] = (spec.speaker_encoder_dim, int(cfg.speaker_encoder_config.enc_dim))
        bad = {k: v for k, v in want.items() if v[0] != v[1]}
        if bad:
            raise ValueError("model spec does not match the loaded weights: "
                             + ", ".join(f"{k} spec {a} vs model {b}" for k, (a, b) in bad.items()))
    return variant


def check_turbo(tcfg, talker_config) -> None:
    """The Turbo head runs in the talker space with no projection, so it exists only where the code predictor hidden
    equals the talker hidden (0.6B). turbo.json `hidden_size` must equal the talker hidden."""
    H = int(talker_config.hidden_size)
    cp = int(talker_config.code_predictor_config.hidden_size)
    th = getattr(tcfg, "hidden_size", None)
    if th is not None and int(th) != H:
        raise ValueError(f"turbo.json hidden_size {th} does not match the talker hidden size {H}")
    if cp != H:
        raise ValueError(f"the Turbo head needs code predictor hidden == talker hidden, got {cp} vs {H} "
                         "(Turbo exists for 0.6B checkpoints only)")


def _inspect(model_dir: str, turbo=None, turbo_head=None):
    """The core's ModelSpec when the core is installed and model_dir is a folder; None otherwise."""
    if not os.path.isdir(model_dir):
        return None
    try:
        from lure_tts.model import inspect
    except ImportError:
        return None
    return inspect(model_dir, turbo=turbo, turbo_head=turbo_head)


class CudaBackend:
    name = "cuda"
    sample_rate = SAMPLE_RATE
    concurrent = True
    turbo = False

    def __init__(self, base_model, engine, codec, model_dir: str, device: str, dtype, opts: dict, spec=None):
        self.base = base_model  # qwen_tts.Qwen3TTSModel
        self.m = base_model.model
        self.engine = engine
        self.codec = codec
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.opts = opts
        self.spec = spec
        self.variant = check_model(self.m, spec)
        self.modes = VARIANT_MODES[self.variant]
        self.layout = normalize_layout(opts.get("layout") or (spec.layout if spec is not None else "non_streaming"))
        self._gpu_voices: "OrderedDict[str, Tuple[Any, Any]]" = OrderedDict()

    # ----------------------------------------------------------------------------------------------- load
    @classmethod
    def load(cls, model_dir: str, quant: Optional[str] = None, *, spec=None, device: str = "cuda",
             dtype: str = "bfloat16", codec_dtype: str = "float32", attn_implementation: str = "sdpa",
             max_batch: int = 8, max_seq_len: int = 4096, use_graphs: bool = True, warmup: bool = True,
             verify_prompt_ids: bool = False, decode_context_frames: int = 72, turbo=None,
             turbo_head: Optional[str] = None, layout: Optional[str] = None, **unused) -> "CudaBackend":
        """Load a Qwen3-TTS 12Hz checkpoint in the HF layout (or a hub id): 0.6B or 1.7B, Base, CustomVoice or
        VoiceDesign, including the Lure family. quant: None / "bf16" only (the torch path has no weight quant).

        spec: the core's ModelSpec (lure_tts.model.inspect). Without it the backend inspects model_dir itself when
            the core is installed. The loaded weights are checked against it.
        dtype: talker/predictor/speaker-encoder weights + compute (bf16 default).
        codec_dtype: 12 Hz speech tokenizer (encoder + decoder) dtype; float32 default (cheap, closest to reference).
        max_batch: CUDA-graph batch (concurrent segments). max_seq_len: KV slots per batch row (prefill + frames).
        use_graphs=False: eager steps (debug / parity), same math.
        turbo: None/"auto" = Turbo head iff model_dir/turbo.json exists; 1 = required; 0 = original 15-pass CP.
        turbo_head: path to another head safetensors; groups still come from turbo.json.
        layout: default prompt layout for direct calls without a core request (spec.layout otherwise).
        """
        import torch

        from ._compat import import_qwen_tts
        from .decode import CodecDecoder
        from .engine import BatchEngine

        if unused:
            logger.info("CudaBackend.load: ignoring options %s", sorted(unused))
        if quant not in (None, "", "bf16", "bfloat16", "none"):
            raise NotImplementedError(f"quant={quant!r}: the CUDA torch backend runs bf16 weights only")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA backend: torch.cuda.is_available() is False")
        if use_graphs and not device.startswith("cuda"):
            raise ValueError("CUDA graphs need a CUDA device; pass use_graphs=False for CPU")
        tdtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        cdtype = getattr(torch, codec_dtype) if isinstance(codec_dtype, str) else codec_dtype
        if device.startswith("cuda"):
            if torch.device(device).index is None:  # set_device needs an index; "cuda" means the current device
                device = f"cuda:{torch.cuda.current_device()}"
            torch.cuda.set_device(torch.device(device))
        if spec is None:
            spec = _inspect(model_dir, turbo, turbo_head)

        qwen_tts = import_qwen_tts()
        t0 = time.perf_counter()
        base = qwen_tts.Qwen3TTSModel.from_pretrained(model_dir, device_map=device, dtype=tdtype,
                                                     attn_implementation=attn_implementation)
        m = base.model
        variant = check_model(m, spec)
        st = m.speech_tokenizer
        if cdtype != tdtype and hasattr(st, "model"):
            st.model.to(cdtype)
        m.eval()
        from .turbo import load_turbo_head, resolve_turbo
        if spec is not None:
            tcfg = spec.turbo
        else:
            tcfg = resolve_turbo(model_dir, turbo, turbo_head) if os.path.isdir(model_dir) else None
        head = None
        if tcfg is not None:
            check_turbo(tcfg, m.config.talker_config)
            head = load_turbo_head(m.talker.code_predictor, tcfg, device, tdtype)
        engine = BatchEngine(m.talker, m.config.talker_config, max_batch=max_batch, max_seq_len=max_seq_len,
                             device=device, dtype=tdtype, use_graphs=use_graphs, turbo_head=head)
        if warmup:
            engine.capture()
        engine.start()
        codec = CodecDecoder(st, device)
        logger.info("CudaBackend loaded %s (%s, talker hidden %d) in %.1fs (batch %d, max_seq_len %d, graphs %s, "
                    "turbo %s)", model_dir, variant, m.config.talker_config.hidden_size, time.perf_counter() - t0,
                    max_batch, max_seq_len, use_graphs, tcfg is not None)
        be = cls(base, engine, codec, model_dir, device, tdtype,
                 dict(verify_prompt_ids=verify_prompt_ids, decode_context_frames=decode_context_frames,
                      max_batch=max_batch, max_seq_len=max_seq_len, layout=layout), spec=spec)
        be.turbo = tcfg is not None
        be.build_info = {"turbo": be.turbo, "turbo_head": getattr(tcfg, "head_path", None), "graphs": use_graphs,
                         "max_batch": max_batch, "variant": variant,
                         "talker_hidden_size": int(m.config.talker_config.hidden_size)}
        return be

    def close(self) -> None:
        if self.engine is not None:
            self.engine.stop()

    # ----------------------------------------------------------------------------------------------- voices
    def encode_voice(self, ref_wav, ref_text: str):
        """Base only. ref_wav float32 mono 24 kHz (or path / (array, sr)) -> (ref_codes int32 [T,16], spk_emb float32
        [H]). Same as qwen-tts create_voice_clone_prompt, with no appended silence. An empty ref_text gives an
        x-vector-only voice (mode xvec); the codes are returned anyway."""
        if self.variant != "base":
            raise ValueError(f"{VARIANT_NAMES[self.variant]} checkpoints have no speaker encoder; voices need a Base "
                             "checkpoint")
        x = _load_wav_24k(ref_wav)
        if ref_text and ref_text.strip():
            item = self.base.create_voice_clone_prompt(ref_audio=[(x, 24000)], ref_text=[ref_text],
                                                       x_vector_only_mode=False)[0]
            codes, spk = item.ref_code, item.ref_spk_embedding
        else:  # create_voice_clone_prompt(x_vector_only_mode=True) drops the codes; compute both the same way
            codes = self.m.speech_tokenizer.encode([x], sr=24000).audio_codes[0]
            spk = self.m.extract_speaker_embedding(audio=x, sr=24000)
        codes = codes.detach().to("cpu").numpy().astype(np.int32)
        spk = spk.detach().float().to("cpu").numpy().astype(np.float32)
        return codes, spk

    def make_voice(self, ref_wav, ref_text: str = "", voice_id: Optional[str] = None):
        codes, spk = self.encode_voice(ref_wav, ref_text)
        vid = voice_id or (os.path.splitext(os.path.basename(str(ref_wav)))[0] if isinstance(ref_wav, (str, os.PathLike))
                           else hashlib.sha256(codes.tobytes()).hexdigest()[:12])
        V = CoreVoice if CoreVoice is not None else Voice
        return V(vid, ref_text, codes, spk, {"encoder": "cuda"})

    def _voice_tensors(self, voice):
        import torch

        if voice is None:
            return None, None
        codes = np.asarray(voice.ref_codes)
        spk = np.asarray(voice.spk_emb, dtype=np.float32)
        key = hashlib.sha1(codes.astype(np.int32).tobytes() + spk.tobytes()).hexdigest()
        hit = self._gpu_voices.get(key)
        if hit is not None:
            self._gpu_voices.move_to_end(key)
            return hit
        rc = torch.as_tensor(codes.astype(np.int64), device=self.device)
        se = torch.as_tensor(spk, device=self.device)
        self._gpu_voices[key] = (rc, se)
        while len(self._gpu_voices) > 64:
            self._gpu_voices.popitem(last=False)
        return rc, se

    # ----------------------------------------------------------------------------------------------- prompt
    def prompt_ids(self, text: str, ref_text: Optional[str], instruct: Optional[str] = None):
        """qwen-tts processor ids. ref_text None (every mode but icl) gives ref_ids None."""
        from .prompt import build_prompt_ids

        return build_prompt_ids(self.base.processor, text, ref_text, instruct)

    def prefill_rows(self, prompt, voice=None, language: str = "english", mode: str = "icl",
                     layout: str = "non_streaming", speaker: Optional[str] = None, speaker_id: Optional[int] = None):
        """(prefill embeds [1, L, H], trailing text [1, N, H], tts_pad [1, 1, H]) for any mode and layout."""
        from .prompt import PromptIds, build_prefill

        rc, se = self._voice_tensors(voice if mode in CLONE_MODES else None)
        return build_prefill(self.m, PromptIds.coerce(prompt), rc if mode == "icl" else None, se, language, mode=mode,
                             layout=layout, speaker=speaker, speaker_id=speaker_id)

    def prefill(self, prompt, voice, language: str = "english"):
        """(prefill embeds [1, L, H], tts_pad [1, 1, H]) for the icl non_streaming layout (GPU scripts)."""
        emb, _trailing, pad = self.prefill_rows(prompt, voice, language)
        return emb, pad

    def dump_layout(self, text, ref_text, direction, n_ref_frames, language="english", mode="icl",
                    layout="non_streaming", speaker=None):
        """Prompt-check hook (conformance/runner.py): this backend's own build_prefill run on a symbolic stub of the
        loaded config, decoded back to [text, codec] rows."""
        from .prompt import symbolic_rows

        p = self._tokenize(text, ref_text if mode == "icl" else None, direction)
        rows = symbolic_rows(self.m.config, p, n_ref_frames, language, mode, layout, speaker)
        return {"layout": rows["layout"], "trailing": rows["trailing"]}

    def _tokenize(self, text, ref_text, direction):
        """Core tokenizer when available (the normative ids), else the qwen-tts processor (same ids, tested)."""
        try:
            from lure_tts.prompt import build_prompt_ids
            from lure_tts.tokenizer import QwenTokenizer
            from .prompt import PromptIds

            return PromptIds.coerce(build_prompt_ids(QwenTokenizer.from_dir(self.model_dir), text, ref_text, direction))
        except (ImportError, OSError, FileNotFoundError):
            return self.prompt_ids(text, ref_text, direction)

    # ----------------------------------------------------------------------------------------------- generate
    def generate(self, req=None, **kw) -> Iterator["Chunk"]:
        """Core path: generate(GenerateRequest). Keyword path: see generate_segment."""
        if req is None:
            return self.generate_segment(**kw)
        core_prompt = req.build_prompt(self.spec.ids) if self.spec is not None and hasattr(req, "build_prompt") else None
        return self._generate(
            text=req.text, voice=req.voice, prompt=getattr(req, "prompt", None), instruct=req.instruct,
            sampling=req.sampling, seed=req.seed, max_new_tokens=req.max_new_tokens, stream=req.stream,
            chunk_frames=req.chunk_frames, first_chunk_frames=req.first_chunk_frames, language=req.language,
            mode=getattr(req, "mode", "icl"), layout=getattr(req, "layout", "non_streaming"),
            speaker=getattr(req, "speaker", None), speaker_id=getattr(req, "speaker_id", None), core_prompt=core_prompt)

    def generate_segment(self, text: str, voice=None, instruct: Optional[str] = None, sampling=None,
                         seed: Optional[int] = None, stream: bool = True, language: str = "english",
                         chunk_frames: int = 4, first_chunk_frames: Optional[int] = None,
                         max_new_tokens: Optional[int] = None, prompt=None, mode: Optional[str] = None,
                         layout: Optional[str] = None, speaker: Optional[str] = None,
                         speaker_id: Optional[int] = None) -> Iterator["Chunk"]:
        """Direct use without the core. mode defaults from the variant and arguments: Base gives icl for a voice with a
        transcript and xvec without one, CustomVoice gives custom_voice (needs speaker), VoiceDesign gives
        voice_design (needs instruct). layout defaults to the load-time layout."""
        return self._generate(text=text, voice=voice, prompt=prompt, instruct=instruct, sampling=sampling, seed=seed,
                              max_new_tokens=max_new_tokens, stream=stream, chunk_frames=chunk_frames,
                              first_chunk_frames=first_chunk_frames, language=language, mode=mode, layout=layout,
                              speaker=speaker, speaker_id=speaker_id)

    def _default_mode(self, voice) -> str:
        if self.variant != "base":
            return self.variant
        return "icl" if voice is not None and voice.ref_text else "xvec"

    def _generate(self, *, text, voice, prompt, instruct, sampling, seed, max_new_tokens, stream, chunk_frames,
                  first_chunk_frames, language, mode=None, layout=None, speaker=None, speaker_id=None,
                  core_prompt=None) -> Iterator["Chunk"]:
        import torch

        from .engine import Request
        from .prompt import PromptIds, has_direction
        from .sampling import make_generator, max_frames_for, resolve_sampling

        t0 = time.perf_counter()
        mode = mode or self._default_mode(voice)
        layout = normalize_layout(layout or self.layout)
        if mode not in self.modes:
            raise ValueError(f"mode {mode!r} is not available on a {VARIANT_NAMES[self.variant]} checkpoint "
                             f"(modes: {', '.join(self.modes)})")
        if mode in CLONE_MODES and voice is None:
            raise ValueError(f"{mode} needs a voice")
        if mode == "icl" and not voice.ref_text:
            raise ValueError("icl needs a voice with a reference transcript (use xvec without one)")
        ref_text = voice.ref_text if mode == "icl" else None
        sp = resolve_sampling(sampling)
        mnt = int(max_new_tokens or sp.max_new_tokens or max_frames_for(text))
        if prompt is None:
            prompt = self.prompt_ids(text, ref_text, instruct)
        else:
            prompt = PromptIds.coerce(prompt)
            if has_direction(instruct) != bool(prompt.instruct_ids):
                raise ValueError("instruct and prompt.instruct_ids disagree")
            if self.opts.get("verify_prompt_ids"):
                mine = self.prompt_ids(text, ref_text, instruct)
                if mine != prompt:
                    raise AssertionError(f"core prompt ids differ from qwen-tts processor ids:\n{prompt}\n{mine}")
        with torch.inference_mode():
            emb, trailing, pad = self.prefill_rows(prompt, voice, language, mode, layout, speaker, speaker_id)
        if core_prompt is not None:
            _check_rows(core_prompt, emb, trailing)
        rc = self._voice_tensors(voice)[0] if mode == "icl" else None  # only icl decodes with the reference
        r = Request(prefill=emb, tts_pad=pad, trailing=trailing, sampling=sp, max_new_tokens=mnt,
                    generator=make_generator(seed, self.device))
        self.engine.submit(r)
        return self._consume(r, rc, stream, chunk_frames, first_chunk_frames, t0, {"mode": mode, "layout": layout})

    def _consume(self, r, ref_codes, stream, chunk_frames, first_chunk_frames, t0, extra=None) -> Iterator["Chunk"]:
        from .decode import StreamDecoder
        from .engine import drain

        ttfa = None
        n = 0
        try:
            if stream:
                dec = StreamDecoder(self.codec, ref_codes, chunk_frames=chunk_frames or 4,
                                    first_chunk_frames=first_chunk_frames or chunk_frames or 4,
                                    context_frames=self.opts.get("decode_context_frames", 72))
                for frame in drain(r):
                    pcm = dec.feed(frame)
                    if pcm is not None and pcm.size:
                        if ttfa is None:
                            ttfa = time.perf_counter() - t0
                        n += pcm.size
                        yield Chunk(pcm)
                pcm = dec.flush()
                if pcm is not None and pcm.size:
                    if ttfa is None:
                        ttfa = time.perf_counter() - t0
                    n += pcm.size
                    yield Chunk(pcm)
                last = np.zeros(0, np.float32)
            else:
                import torch

                frames = list(drain(r))
                if frames:
                    last = self.codec.decode_full(ref_codes, torch.tensor(frames, dtype=torch.long))
                else:
                    last = np.zeros(0, np.float32)
                n = last.size
                ttfa = time.perf_counter() - t0
            info = SegmentInfo(r.frames, r.ended_by_eos, n, ttfa, time.perf_counter() - t0,
                               {"stop_reason": r.stop_reason, "prefill_len": int(r.prefill.shape[1]),
                                "backend": "cuda", "turbo": self.turbo, "first_frame_s": None if r.t_first_frame is None
                                else r.t_first_frame - r.t_submit, **(extra or {})})
            yield Chunk(last, info)
        finally:
            if r.t_done is None:
                r.cancel()  # consumer went away: free the slot at the next step


def _check_rows(core_prompt, emb, trailing) -> None:
    """The prefill mirrors the core's rows; a length mismatch means the two builders disagree on the layout."""
    L, N = int(emb.shape[1]), int(trailing.shape[1])
    if L != len(core_prompt.prefill) or N != len(core_prompt.trailing):
        raise AssertionError(f"cuda prefill ({L} rows, {N} trailing) disagrees with the core prompt "
                             f"({len(core_prompt.prefill)} rows, {len(core_prompt.trailing)} trailing) for "
                             f"{core_prompt.mode}/{core_prompt.layout}")


def load(model_dir: str, quant: Optional[str] = None, **opts) -> CudaBackend:
    return CudaBackend.load(model_dir, quant=quant, **opts)


def make_voice(backend: CudaBackend, ref_wav, ref_text: str):
    return backend.make_voice(ref_wav, ref_text)


def generate(backend: CudaBackend, segment_text: str, voice=None, instruct: Optional[str] = None, sampling=None,
             seed: Optional[int] = None, stream: bool = True, **kw):
    return backend.generate_segment(segment_text, voice, instruct=instruct, sampling=sampling, seed=seed, stream=stream, **kw)

