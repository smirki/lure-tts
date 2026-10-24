"""Import-time compatibility helpers for the Lure CUDA backend.

qwen-tts(-hf) imports its 25 Hz tokenizer at package import, which pulls `sox`, `onnxruntime` and
`torchaudio.compliance.kaldi`. Lure only uses the 12 Hz tokenizer, so none of those are needed. When they are
missing (for example a Windows install done with `pip install --no-deps qwen-tts-hf`), we register inert stub
modules so `import qwen_tts` succeeds. A stub raises a clear error if anything actually touches it.
"""
from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import sys
import types


class _MissingModule(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []  # lets `import a.b.c` resolve through the stub chain
        self.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        self.__lure_stub__ = True

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        raise ImportError(
            f"optional dependency '{self.__name__}' is not installed (only needed by the 25 Hz Qwen tokenizer, "
            f"never by Lure). Install it if you really need '{self.__name__}.{attr}'."
        )


def _available(name: str) -> bool:
    if name in sys.modules:
        return not getattr(sys.modules[name], "__lure_stub__", False)
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def install_optional_stubs() -> list[str]:
    """Register stubs for missing optional modules. Returns the names that were stubbed."""
    groups = {"sox": ("sox",), "onnxruntime": ("onnxruntime",),
              "torchaudio": ("torchaudio", "torchaudio.compliance", "torchaudio.compliance.kaldi")}
    stubbed = []
    for root, names in groups.items():
        if _available(root):
            continue
        for name in names:
            if name not in sys.modules:
                sys.modules[name] = _MissingModule(name)
                stubbed.append(name)
        for name in names:
            parent, _, child = name.rpartition(".")
            if parent:
                setattr(sys.modules[parent], child, sys.modules[name])
    return stubbed


def remove_optional_stubs(names) -> None:
    for name in names:
        mod = sys.modules.get(name)
        if getattr(mod, "__lure_stub__", False):
            del sys.modules[name]


class _FilteredStdout:
    def __init__(self, stream, suppress_substrings):
        self._stream = stream
        self._suppress = suppress_substrings

    def write(self, data):
        if any(s in data for s in self._suppress):
            return len(data)
        return self._stream.write(data)

    def flush(self):
        return self._stream.flush()


@contextlib.contextmanager
def suppress_flash_attn_warning():
    """qwen_tts prints a flash-attn notice at import; the backend uses SDPA, so the notice is noise.
    From faster-qwen3-tts utils.py (MIT, Andres Marafioti)."""
    filtered = _FilteredStdout(sys.stdout, ("flash-attn is not installed", "manual PyTorch version",
                                            "Please install flash-attn"))
    with contextlib.redirect_stdout(filtered):
        yield


def import_qwen_tts():
    """Import qwen_tts with optional deps stubbed (only for the duration of the import, so transformers never sees a
    fake torchaudio afterwards) and the flash-attn notice silenced."""
    if "qwen_tts" in sys.modules:
        return sys.modules["qwen_tts"]
    import transformers  # noqa: F401  - settle transformers' own availability probes before stubbing
    try:
        from transformers.utils import import_utils as _iu

        for probe in ("is_torchaudio_available", "is_onnx_available"):
            f = getattr(_iu, probe, None)
            if callable(f):
                f()
    except Exception:
        pass
    stubbed = install_optional_stubs()
    try:
        with suppress_flash_attn_warning():
            import qwen_tts
    finally:
        remove_optional_stubs(stubbed)
    patch_default_rope()
    return qwen_tts


def _rope_param(config, key, default=None):
    """transformers 5 moved rope_theta / partial_rotary_factor into config.rope_parameters (MimiConfig has no
    top-level rope_theta any more); 4.57 keeps them on the config."""
    v = getattr(config, key, None)
    if v is not None:
        return v
    rp = getattr(config, "rope_parameters", None) or {}
    if isinstance(rp, dict) and key not in rp and rp and all(isinstance(x, dict) for x in rp.values()):
        rp = next(iter(rp.values()))  # per-layer-type dict
    return rp.get(key, default) if isinstance(rp, dict) else default


def _default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    """qwen_tts._transformers_compat._default_rope_parameters, reading rope_theta from either config layout."""
    import torch

    base = _rope_param(config, "rope_theta", 10000.0)
    factor = _rope_param(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def patch_default_rope() -> None:
    """qwen-tts-hf registers a "default" RoPE init that reads config.rope_theta, which crashes the 12 Hz codec's Mimi
    decoder init on transformers 5.20 (AttributeError: 'MimiConfig' object has no attribute 'rope_theta'). Replace it
    with a version that also reads config.rope_parameters. Same numbers whenever the old one worked."""
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:  # pragma: no cover
        return
    qc = sys.modules.get("qwen_tts._transformers_compat")
    if qc is not None and hasattr(qc, "_default_rope_parameters"):
        old = qc._default_rope_parameters
        qc._default_rope_parameters = _default_rope_parameters
        if ROPE_INIT_FUNCTIONS.get("default") is old:
            ROPE_INIT_FUNCTIONS["default"] = _default_rope_parameters
    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _default_rope_parameters


def static_layer_init(layer, k, v) -> None:
    """transformers 5: lazy_initialization(key, value); 4.57: lazy_initialization(key)."""
    if getattr(layer, "is_initialized", False):
        return
    try:
        layer.lazy_initialization(k, v)
    except TypeError:
        layer.lazy_initialization(k)
