"""Lure backend for Apple silicon, an adapter over mlx-audio (pinned at 402e3b9)."""
from .backend import MLX_AUDIO_COMMIT, MLXBackend

__all__ = ["MLXBackend", "MLX_AUDIO_COMMIT"]
