"""On-disk voice cache: one folder per voice with the reference codes, x-vector and metadata.

  <cache>/voices/<voice_id>/voice.json      {voice_id, ref_text, wav_sha256, n_ref_frames, encoder, model_fingerprint, created}
  <cache>/voices/<voice_id>/ref_codes.npy   int32 [T_ref, 16]
  <cache>/voices/<voice_id>/spk_emb.npy     float32 [1024]
  <cache>/voices/<voice_id>/ref.wav         the 24 kHz mono reference actually encoded (for re-encoding on another backend)

A cached voice is reused when wav hash, ref_text and model fingerprint all match; otherwise it is re-encoded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Optional

import numpy as np

from .audio import write_wav
from .backends.base import Voice


def default_cache_dir() -> str:
    return os.environ.get("LURE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "lure"))


def model_fingerprint(model_dir: str) -> str:
    """Cheap and stable: config + speech tokenizer config + weight file sizes (hashing 1.8 GB on every start is too slow)."""
    h = hashlib.sha256()
    for rel in ("config.json", "speech_tokenizer/config.json", "model.safetensors", "speech_tokenizer/model.safetensors"):
        p = os.path.join(model_dir, rel)
        if not os.path.exists(p):
            continue
        h.update(rel.encode())
        if p.endswith(".json"):
            h.update(open(p, "rb").read())
        else:
            h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()[:16]


def wav_sha(wav: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(wav, dtype=np.float32).tobytes()).hexdigest()[:16]


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_id(name: str) -> str:
    s = _SAFE.sub("_", name).strip("._")
    if not s:
        raise ValueError(f"bad voice id {name!r}")
    return s[:80]


class VoiceCache:
    def __init__(self, root: Optional[str] = None):
        self.root = os.path.join(root or default_cache_dir(), "voices")
        self._mem: dict[str, Voice] = {}

    def path(self, voice_id: str) -> str:
        return os.path.join(self.root, safe_id(voice_id))

    def list(self) -> list[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(d for d in os.listdir(self.root) if os.path.exists(os.path.join(self.root, d, "voice.json")))

    def load(self, voice_id: str) -> Voice:
        vid = safe_id(voice_id)
        if vid in self._mem:
            return self._mem[vid]
        d = self.path(vid)
        if not os.path.exists(os.path.join(d, "voice.json")):
            raise KeyError(f"voice {voice_id!r} not in cache {self.root}")
        meta = json.load(open(os.path.join(d, "voice.json")))
        v = Voice(vid, meta["ref_text"], np.load(os.path.join(d, "ref_codes.npy")).astype(np.int32),
                  np.load(os.path.join(d, "spk_emb.npy")).astype(np.float32), meta)
        self._mem[vid] = v
        return v

    def save(self, v: Voice, ref_wav: Optional[np.ndarray] = None) -> str:
        d = self.path(v.voice_id)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "ref_codes.npy"), np.asarray(v.ref_codes, dtype=np.int32))
        np.save(os.path.join(d, "spk_emb.npy"), np.asarray(v.spk_emb, dtype=np.float32))
        if ref_wav is not None:
            write_wav(os.path.join(d, "ref.wav"), ref_wav)
        meta = dict(v.meta, voice_id=v.voice_id, ref_text=v.ref_text, n_ref_frames=v.n_ref_frames)
        tmp = os.path.join(d, "voice.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, os.path.join(d, "voice.json"))
        self._mem[safe_id(v.voice_id)] = v
        return d

    def get_or_make(self, voice_id: Optional[str], ref_wav: np.ndarray, ref_text: str, encode, encoder: str, fingerprint: str) -> Voice:
        """encode(ref_wav, ref_text) -> (ref_codes, spk_emb). voice_id None -> content hash id."""
        sha = wav_sha(ref_wav)
        vid = safe_id(voice_id) if voice_id else f"v_{hashlib.sha256((sha + ref_text).encode()).hexdigest()[:12]}"
        try:
            v = self.load(vid)
            m = v.meta
            if m.get("wav_sha256") == sha and v.ref_text == ref_text and m.get("model_fingerprint") == fingerprint:
                return v
        except KeyError:
            pass
        codes, spk = encode(ref_wav, ref_text)
        codes = np.asarray(codes, dtype=np.int32)
        if codes.ndim != 2 or codes.shape[1] != 16:
            raise ValueError(f"ref_codes must be [T,16], got {codes.shape}")
        v = Voice(vid, ref_text, codes, np.asarray(spk, dtype=np.float32).reshape(-1),
                  {"wav_sha256": sha, "encoder": encoder, "model_fingerprint": fingerprint, "created": time.time()})
        self._mem.pop(vid, None)
        self.save(v, ref_wav)
        return v
