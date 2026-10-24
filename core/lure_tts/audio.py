"""Audio I/O helpers: load + resample reference wavs, float32 <-> PCM16, streaming WAV header."""
from __future__ import annotations

import io
import struct

import numpy as np

from .defaults import SAMPLE_RATE


def resample(x: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target:
        return x.astype(np.float32)
    try:
        import soxr  # same family librosa (qwen-tts) uses by default: soxr_hq
        return soxr.resample(x, sr, target, quality="HQ").astype(np.float32)
    except ImportError:
        pass
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, target)
        return resample_poly(x, target // g, sr // g).astype(np.float32)
    except ImportError:
        n = int(round(len(x) * target / sr))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def load_audio(src, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Path, bytes, or (array, sr) -> float32 mono at target_sr."""
    if isinstance(src, tuple):
        x, sr = src
        x = np.asarray(x, dtype=np.float32)
    else:
        import soundfile as sf
        x, sr = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    return resample(x, int(sr), target_sr)


def to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def wav_header(sample_rate: int = SAMPLE_RATE, channels: int = 1, bits: int = 16, n_samples: int | None = None) -> bytes:
    """n_samples None -> streaming header with 0xFFFFFFFF sizes (players read until EOF)."""
    block = channels * bits // 8
    data = 0xFFFFFFFF if n_samples is None else n_samples * block
    riff = 0xFFFFFFFF if n_samples is None else 36 + data
    return (b"RIFF" + struct.pack("<I", riff) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
            sample_rate * block, block, bits) + b"data" + struct.pack("<I", data))


def wav_bytes(x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    return wav_header(sample_rate, n_samples=len(x)) + to_pcm16(x)


def write_wav(path: str, x: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    with open(path, "wb") as f:
        f.write(wav_bytes(x, sample_rate))
