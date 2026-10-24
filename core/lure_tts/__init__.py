"""Lure TTS: runs Qwen3-TTS 12Hz and Lure-family checkpoints through one API over the mlx (Apple silicon), cuda (NVIDIA)
and cpp (ggml) backends."""
from .defaults import DEFAULT_SAMPLING, SAMPLE_RATE, Sampling
from .model import ModelSpec, inspect
from .prompt import LAYOUTS, MODES, ModelIds, Prompt, PromptIds, build_layout, build_prompt, build_prompt_ids
from .text import Segmenter, segment_text

__version__ = "0.1.0"


def __getattr__(name):  # lazy: keep `import lure_tts` free of backend / numpy-heavy imports
    if name in ("Lure", "LiveSession"):
        from . import api
        return getattr(api, name)
    if name == "Voice":
        from .backends.base import Voice
        return Voice
    raise AttributeError(name)


__all__ = ["Lure", "LiveSession", "Voice", "Sampling", "DEFAULT_SAMPLING", "SAMPLE_RATE", "ModelSpec", "inspect",
           "ModelIds", "PromptIds", "Prompt", "MODES", "LAYOUTS", "build_prompt_ids", "build_prompt", "build_layout",
           "Segmenter", "segment_text", "__version__"]
