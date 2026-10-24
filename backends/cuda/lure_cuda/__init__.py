"""Lure backend for NVIDIA GPUs. The CUDA graph code is derived from faster-qwen3-tts (MIT, Andres Marafioti; see
LICENSE). The core loads it as `lure_cuda:CudaBackend`."""
from .backend import CudaBackend, Voice, generate, load, make_voice
from .sampling import SamplingParams, resolve_sampling

__all__ = ["CudaBackend", "Voice", "load", "make_voice", "generate", "SamplingParams", "resolve_sampling"]
__version__ = "0.1.0"
