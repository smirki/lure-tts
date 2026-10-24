"""Lure `cpp` backend: ctypes binding over liblure_tts (built from backends/cpp).

Implements lure_tts.backends.base.Backend: load / encode_voice / generate / close, for every Qwen3-TTS 12Hz
checkpoint (Base: icl / xvec, CustomVoice: custom_voice, VoiceDesign: voice_design; both prompt layouts), plus the
optional dump_layout() used by the conformance prompt check.
Library lookup: $LURE_CPP_LIB, then backends/cpp/build/<preset>/bin/liblure_tts.{dylib,so} or lure_tts.dll.
The C side tokenizes itself. generate() checks its ids against the core's PromptIds before running
(set LURE_CPP_CHECK_IDS=0 to skip the check).
"""
from __future__ import annotations

import ctypes
import json
import os
import queue
import sys
import threading
import time
from typing import Iterator, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SAMPLE_RATE = 24000


class _Params(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("temperature", ctypes.c_float), ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float), ("repetition_penalty", ctypes.c_float), ("subtalker_temperature", ctypes.c_float),
        ("subtalker_top_k", ctypes.c_int32), ("subtalker_top_p", ctypes.c_float), ("do_sample", ctypes.c_int32),
        ("subtalker_do_sample", ctypes.c_int32), ("max_new_tokens", ctypes.c_int32), ("has_seed", ctypes.c_int32),
        ("seed", ctypes.c_uint32), ("chunk_frames", ctypes.c_int32), ("chunk_grow", ctypes.c_int32),
    ]


class _Request(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("mode", ctypes.c_int32), ("layout", ctypes.c_int32),
        ("speaker_id", ctypes.c_int32), ("text", ctypes.c_char_p), ("instruct", ctypes.c_char_p),
        ("language", ctypes.c_char_p), ("speaker", ctypes.c_char_p), ("voice", ctypes.c_void_p),
    ]


ABI_VERSION = 2
MODES = {"icl": 0, "xvec": 1, "custom_voice": 2, "voice_design": 3}
LAYOUTS = {None: -1, "default": -1, "non_streaming": 0, "streaming": 1}


class _Info(ctypes.Structure):
    _fields_ = [
        ("frames", ctypes.c_int64), ("ended_by_eos", ctypes.c_int32), ("cancelled", ctypes.c_int32),
        ("samples", ctypes.c_int64), ("sample_rate", ctypes.c_int32), ("talker_ms", ctypes.c_double),
        ("decoder_ms", ctypes.c_double), ("first_chunk_ms", ctypes.c_double), ("total_ms", ctypes.c_double),
    ]


_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int32, ctypes.c_void_p)


def _find_lib() -> str:
    env = os.environ.get("LURE_CPP_LIB")
    if env:
        return env
    name = {"darwin": "liblure_tts.dylib", "win32": "lure_tts.dll"}.get(sys.platform, "liblure_tts.so")
    for preset in ("macos-metal", "linux-cuda", "linux-vulkan", "linux-cpu", "linux-cpu-native", "macos-cpu",
                   "windows-cuda", "windows-vulkan", "windows-cpu"):
        path = os.path.join(ROOT, "build", preset, "bin", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} not found: build backends/cpp and install this package with pip install -e, "
                            "or set LURE_CPP_LIB to the library path")


def _bind(lib):
    vp, cp, i32, sz = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_size_t
    req = ctypes.POINTER(_Request)
    sigs = {
        "lure_abi_version": (ctypes.c_uint32, []),
        "lure_build_info": (cp, []),
        "lure_last_error": (cp, []),
        "lure_default_params": (None, [ctypes.POINTER(_Params)]),
        "lure_default_request": (None, [req]),
        "lure_load": (vp, [cp, cp, i32, i32, cp]),
        "lure_free": (None, [vp]),
        "lure_variant": (cp, [vp]),
        "lure_speaker_count": (i32, [vp]),
        "lure_speaker_name": (cp, [vp, i32]),
        "lure_speaker_id": (i32, [vp, i32]),
        "lure_model_info": (vp, [vp]),
        "lure_model_info_offline": (vp, [cp]),
        "lure_voice_create": (vp, [vp, ctypes.POINTER(ctypes.c_float), sz, i32, cp]),
        "lure_voice_from_codes": (vp, [vp, ctypes.POINTER(ctypes.c_int32), i32, i32, ctypes.POINTER(ctypes.c_float), i32, cp]),
        "lure_voice_codes": (ctypes.c_int64, [vp, ctypes.POINTER(ctypes.c_int32), sz]),
        "lure_voice_speaker": (ctypes.c_int64, [vp, ctypes.POINTER(ctypes.c_float), sz]),
        "lure_voice_frames": (i32, [vp]),
        "lure_voice_has_transcript": (i32, [vp]),
        "lure_voice_free": (None, [vp]),
        "lure_generate": (i32, [vp, req, ctypes.POINTER(_Params), _CB, vp, ctypes.POINTER(_Info)]),
        "lure_dump_prefill": (vp, [vp, req]),
        "lure_dump_prefill_offline": (vp, [cp, req, cp, i32]),
        "lure_tokenize_offline": (vp, [cp, cp]),
        "lure_string_free": (None, [vp]),
    }
    for name, (res, args) in sigs.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    return lib


def recipe_rows(recipe: dict, prefix: str = "") -> list:
    """Prefill dump (lure.prefill.v2) -> conformance rows [text, codec], codec = id | "spk" | "ref:<t>" | None.
    prefix "trailing_" converts the trailing rows."""
    n = len(recipe[prefix + "text_ids"])
    out = []
    for i in range(n):
        t = recipe[prefix + "text_ids"][i]
        c = recipe.get(prefix + "codec_ids", [-1] * n)[i]
        r = recipe.get(prefix + "ref_frames", [-1] * n)[i]
        s = recipe.get(prefix + "speaker", [0] * n)[i]
        codec = "spk" if s else f"ref:{r}" if r >= 0 else None if c < 0 else c
        out.append([None if t < 0 else t, codec])
    return out


def make_request(lib, text: str, mode: str = "icl", layout: Optional[str] = None, language: Optional[str] = None,
                 instruct: Optional[str] = None, speaker: Optional[str] = None, speaker_id: Optional[int] = None,
                 voice=None) -> _Request:
    """A lure_request. The returned struct keeps the encoded strings alive through its own fields."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    key = None if layout is None else str(layout).strip().lower().replace("-", "_")
    if key not in LAYOUTS:
        raise ValueError(f"layout must be non_streaming or streaming, got {layout!r}")
    r = _Request()
    lib.lure_default_request(ctypes.byref(r))
    r.mode, r.layout = MODES[mode], LAYOUTS[key]
    r.text, r.instruct, r.language, r.speaker = _b(text), _b(instruct or None), _b(language), _b(speaker)
    if speaker_id is not None:
        r.speaker_id = int(speaker_id)
    r.voice = voice
    return r


def _b(s: Optional[str]):
    return None if s is None else s.encode("utf-8")


def _flag(v) -> str:
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on", "turbo"):
        return "1"
    if s in ("0", "false", "no", "off", "none"):
        return "0"
    raise ValueError(f"turbo must be auto/1/0, got {v!r}")


class CppBackend:
    name = "cpp"
    sample_rate = SAMPLE_RATE

    def __init__(self, model_dir: str, quant: Optional[str] = None, backend: str = "auto", device: int = 0,
                 threads: int = 0, lib_path: Optional[str] = None, options: Optional[str] = None,
                 ggml_backend: Optional[str] = None, turbo=None, turbo_head: Optional[str] = None, **_):
        # `ggml_backend` is an alias of `backend` for callers going through Lure.load(..., backend="cpp"),
        # where the core already uses the `backend` keyword (for example `--opt ggml_backend=cuda`).
        backend = ggml_backend or backend
        self.lib = _bind(ctypes.CDLL(lib_path or _find_lib()))
        if self.lib.lure_abi_version() != ABI_VERSION:
            raise RuntimeError(f"liblure_tts ABI {self.lib.lure_abi_version()}, this binding needs {ABI_VERSION}")
        opts = [o for o in (options or "").split(";") if o]
        if quant:  # native|f32|f16|bf16|q8_0 (talker + code predictor); or point model_dir at a quantized GGUF
            opts.append(f"qwen3_tts.weight_type={quant}")
        # Turbo: auto (default) runs the Turbo head when the model has turbo.json (HF dir or GGUF sidecar);
        # turbo=1 requires it, turbo=0 forces the 15-pass code predictor, turbo_head overrides the head file.
        if turbo is not None and str(turbo).strip().lower() not in ("", "auto"):
            opts.append(f"qwen3_tts.turbo={_flag(turbo)}")
        if turbo_head:
            opts.append(f"qwen3_tts.turbo_head={turbo_head}")
        flag = None if turbo is None or str(turbo).strip().lower() in ("", "auto") else _flag(turbo) == "1"
        if flag is None and os.path.isdir(model_dir):
            flag = os.path.isfile(os.path.join(model_dir, "turbo.json"))
        self.turbo = flag  # True/False, or None for a GGUF on auto (decided by its embedded turbo.json)
        self.model_dir = model_dir
        self.tts = self.lib.lure_load(_b(model_dir), _b(backend), device, threads, _b(";".join(opts)) if opts else None)
        if not self.tts:
            raise RuntimeError(self.lib.lure_last_error().decode())
        self.build_info = self.lib.lure_build_info().decode()
        self.model_info = self._json(self.lib.lure_model_info(self.tts), "model_info")
        self.variant = self.lib.lure_variant(self.tts).decode()
        self.speakers = {self.lib.lure_speaker_name(self.tts, k).decode(): self.lib.lure_speaker_id(self.tts, k)
                         for k in range(self.lib.lure_speaker_count(self.tts))}
        self._lock = threading.Lock()
        self._voices: dict[tuple, int] = {}  # (voice_id, n_frames) -> native handle

    @classmethod
    def load(cls, model_dir: str, quant: Optional[str] = None, **opts) -> "CppBackend":
        return cls(model_dir, quant=quant, **opts)

    def _err(self, what: str):
        return RuntimeError(f"{what}: {self.lib.lure_last_error().decode()}")

    def _json(self, ptr, what: str):
        if not ptr:
            raise self._err(what)
        try:
            return json.loads(ctypes.string_at(ptr).decode("utf-8"))
        finally:
            self.lib.lure_string_free(ptr)

    def encode_voice(self, ref_wav: np.ndarray, ref_text: str):
        """Base only. ref_text "" gives an x-vector-only voice (its codes are still returned)."""
        pcm = np.ascontiguousarray(ref_wav, dtype=np.float32).reshape(-1)
        with self._lock:
            v = self.lib.lure_voice_create(self.tts, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), pcm.size,
                                             SAMPLE_RATE, _b(ref_text))
            if not v:
                raise self._err("encode_voice")
            try:
                n = self.lib.lure_voice_codes(v, None, 0)
                codes = np.empty(n, np.int32)
                self.lib.lure_voice_codes(v, codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), n)
                d = self.lib.lure_voice_speaker(v, None, 0)
                spk = np.empty(d, np.float32)
                self.lib.lure_voice_speaker(v, spk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), d)
            finally:
                self.lib.lure_voice_free(v)
        return codes.reshape(-1, 16), spk

    def _voice_handle(self, codes: np.ndarray, spk: np.ndarray, ref_text: str) -> int:
        codes = np.ascontiguousarray(codes, dtype=np.int32).reshape(-1, 16)
        spk = np.ascontiguousarray(spk, dtype=np.float32).reshape(-1)
        cptr = codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if codes.size else None
        h = self.lib.lure_voice_from_codes(self.tts, cptr, codes.shape[0], 16,
                                             spk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), spk.size, _b(ref_text))
        if not h:
            raise self._err("voice")
        return h

    def _native_voice(self, voice) -> int:
        key = (voice.voice_id, int(voice.ref_codes.shape[0]), voice.ref_text)
        h = self._voices.get(key)
        if h:
            return h
        h = self._voice_handle(voice.ref_codes, voice.spk_emb, voice.ref_text)
        if len(self._voices) >= 16:  # cap the native voice cache
            old_key = next(iter(self._voices))
            self.lib.lure_voice_free(self._voices.pop(old_key))
        self._voices[key] = h
        return h

    def _request(self, req) -> _Request:
        mode = getattr(req, "mode", "icl")
        voice = self._native_voice(req.voice) if mode in ("icl", "xvec") else None
        prompt = getattr(req, "prompt", None)
        # The core's PromptIds decide whether an instruct goes in (instruct_ids None = none).
        instruct = req.instruct if prompt is None or prompt.instruct_ids is not None else None
        return make_request(self.lib, req.text, mode, getattr(req, "layout", None), req.language, instruct,
                            getattr(req, "speaker", None), getattr(req, "speaker_id", None), voice)

    def _params(self, req) -> _Params:
        p = _Params()
        self.lib.lure_default_params(ctypes.byref(p))
        s = req.sampling
        p.temperature, p.top_k, p.top_p = s.temperature, s.top_k, s.top_p
        p.repetition_penalty = s.repetition_penalty
        p.subtalker_temperature, p.subtalker_top_k, p.subtalker_top_p = (
            s.subtalker_temperature, s.subtalker_top_k, s.subtalker_top_p)
        p.do_sample, p.subtalker_do_sample = int(s.do_sample), int(s.subtalker_do_sample)
        p.max_new_tokens = int(req.max_new_tokens or 0)
        if req.seed is not None:
            p.has_seed, p.seed = 1, int(req.seed) & 0xFFFFFFFF
        p.chunk_frames = int(req.chunk_frames) if req.stream else 0
        # first_chunk_frames < chunk_frames -> grow policy (1,2,4,...); else fixed chunks
        p.chunk_grow = 1 if req.stream and req.first_chunk_frames < req.chunk_frames else 0
        return p

    def prefill_dump(self, req) -> dict:
        r = self._request(req)
        with self._lock:
            return self._json(self.lib.lure_dump_prefill(self.tts, ctypes.byref(r)), "prefill_dump")

    def dump_layout(self, text: str, ref_text: Optional[str], direction: Optional[str], n_ref_frames: int,
                    language: str = "english", mode: str = "icl", layout: str = "non_streaming",
                    speaker: Optional[str] = None) -> dict:
        """Conformance hook: prefill rows and trailing rows for one prompt, with a symbolic voice of n_ref_frames."""
        voice = None
        if mode in ("icl", "xvec"):
            dims = int(self.model_info["talker_hidden_size"])
            frames = int(n_ref_frames) if mode == "icl" else 0
            voice = self._voice_handle(np.zeros((frames, 16), np.int32), np.zeros(dims, np.float32),
                                       ref_text if mode == "icl" else "")
        try:
            r = make_request(self.lib, text, mode, layout, language, direction, speaker, None, voice)
            with self._lock:
                recipe = self._json(self.lib.lure_dump_prefill(self.tts, ctypes.byref(r)), "dump_layout")
        finally:
            if voice:
                self.lib.lure_voice_free(voice)
        return {"layout": recipe_rows(recipe), "trailing": recipe_rows(recipe, "trailing_"),
                "input_ids": recipe["input_ids"], "ref_ids": recipe["ref_ids"] or None,
                "instruct_ids": recipe["instruct_ids"] or None}

    def generate(self, req) -> Iterator:
        from lure_tts.backends.base import Chunk, SegmentInfo  # core types

        if os.environ.get("LURE_CPP_CHECK_IDS", "1") != "0" and getattr(req, "prompt", None) is not None:
            dump = self.prefill_dump(req)
            want = req.prompt.to_dict()
            got = {"input_ids": dump["input_ids"], "ref_ids": dump["ref_ids"] or None,
                   "instruct_ids": dump["instruct_ids"] or None}
            if got != want:
                raise RuntimeError(f"cpp backend prompt ids differ from the core's PromptIds: {got} vs {want}")
        request = self._request(req)
        params = self._params(req)
        q: "queue.Queue" = queue.Queue(maxsize=64)
        stop = threading.Event()
        info = _Info()
        t0 = time.time()

        @_CB
        def on_pcm(ptr, n, sr, _user):
            if stop.is_set():
                return 0
            q.put(np.ctypeslib.as_array(ptr, shape=(n,)).copy())
            return 1

        def run():
            with self._lock:
                rc = self.lib.lure_generate(self.tts, ctypes.byref(request), ctypes.byref(params), on_pcm, None,
                                              ctypes.byref(info))
            q.put(("done", rc, None if rc == 0 else self.lib.lure_last_error().decode()))

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        ttfa = None
        try:
            while True:
                item = q.get()
                if isinstance(item, tuple):
                    _, rc, err = item
                    if rc != 0:
                        raise RuntimeError(f"generate: {err}")
                    break
                if ttfa is None:
                    ttfa = time.time() - t0
                yield Chunk(item)
        finally:
            stop.set()
            while worker.is_alive():  # drain so the callback never blocks on a full queue
                try:
                    q.get(timeout=0.05)
                except queue.Empty:
                    pass
            worker.join()
        yield Chunk(np.zeros(0, np.float32), SegmentInfo(
            frames=int(info.frames), ended_by_eos=bool(info.ended_by_eos), samples=int(info.samples), ttfa_s=ttfa,
            wall_s=time.time() - t0,
            extra={"talker_ms": info.talker_ms, "decoder_ms": info.decoder_ms, "first_chunk_ms": info.first_chunk_ms,
                   "cancelled": bool(info.cancelled), "build": self.build_info}))

    def close(self):
        for h in self._voices.values():
            self.lib.lure_voice_free(h)
        self._voices.clear()
        if self.tts:
            self.lib.lure_free(self.tts)
            self.tts = None
