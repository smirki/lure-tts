"""Backend registry and auto-selection: mlx on Apple silicon, cuda if torch.cuda is available, else cpp."""
from __future__ import annotations

import importlib
import importlib.util
import os
import platform
from typing import Optional

from .base import Backend, Chunk, GenerateRequest, SegmentInfo, Voice

ENTRY_POINT_GROUP = "lure.backends"
CONVENTIONAL = {
    "mlx": "lure_mlx:MLXBackend",
    "cuda": "lure_cuda:CudaBackend",
    "cpp": "lure_cpp:CppBackend",
    "fake": "lure_tts.backends.fake:FakeBackend",
}
AUTO_ORDER = ("mlx", "cuda", "cpp")


def _entry_points() -> dict[str, str]:
    try:
        from importlib.metadata import entry_points
        return {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}
    except Exception:
        return {}


def _target(name: str) -> str:
    eps = _entry_points()
    if name in eps:
        return eps[name]
    if name in CONVENTIONAL:
        return CONVENTIONAL[name]
    if ":" in name:  # "pkg.module:Class"
        return name
    raise KeyError(f"unknown backend {name!r}; known: {sorted(set(CONVENTIONAL) | set(eps))}")


def _module_present(target: str) -> bool:
    mod = target.split(":")[0]
    try:
        return importlib.util.find_spec(mod.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


def get_backend_class(name: str):
    mod, _, cls = _target(name).partition(":")
    return getattr(importlib.import_module(mod), cls)


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def platform_ok(name: str) -> bool:
    if name == "mlx":
        return is_apple_silicon()
    if name == "cuda":
        return cuda_available()
    return True


def available() -> dict[str, dict]:
    out = {}
    for name in sorted(set(CONVENTIONAL) | set(_entry_points())):
        t = _target(name)
        out[name] = {"target": t, "installed": _module_present(t), "platform_ok": platform_ok(name) if _module_present(t) else None}
    return out


def select(name: Optional[str] = "auto") -> str:
    """Resolve 'auto' (or env LURE_BACKEND) to a concrete backend name."""
    name = name or "auto"
    if name == "auto":
        name = os.environ.get("LURE_BACKEND", "auto")
    if name != "auto":
        return name
    for cand in AUTO_ORDER:
        t = _target(cand)
        if _module_present(t) and platform_ok(cand):
            return cand
    raise RuntimeError("no usable backend: install backends/mlx on Apple silicon, backends/cuda with an NVIDIA GPU, "
                       "or backends/cpp/python (cpp backend). `lure backends` lists what is installed.")


def load_backend(name: str, model_dir: str, quant: Optional[str] = None, **opts) -> Backend:
    return get_backend_class(name).load(model_dir, quant=quant, **opts)


__all__ = ["Backend", "Chunk", "GenerateRequest", "SegmentInfo", "Voice", "available", "select", "load_backend",
           "get_backend_class", "is_apple_silicon", "cuda_available"]
