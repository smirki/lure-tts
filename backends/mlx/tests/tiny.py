"""Tiny random-weight Qwen3-TTS checkpoints, one per variant, in the Hugging Face layout (config.json, tokenizer files,
model.safetensors). No real weights are involved.

The configs are the reference configs in conformance/reference/configs (ids, speakers, languages, dialects unchanged)
with the layer sizes shrunk. The two size shapes keep the ratios that matter:

  0b6  talker hidden 32 == code predictor hidden 32, no small_to_mtp_projection, x-vector 32
  1b7  talker hidden 64 != code predictor hidden 32, small_to_mtp_projection 64 -> 32, x-vector 64,
       code predictor codec embeddings in the 64-d talker space

Weights are drawn from numpy's PCG64 with a seed per tensor name, so they are identical on every machine. Parameter
names and shapes come from mlx-audio's own modules; conv weights are written in the PyTorch layout, as in a real
checkpoint, and mlx-audio's sanitize turns them back.
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REF = os.path.join(ROOT, "conformance", "reference")
TOKDIR = os.path.join(REF, "tokenizer")
TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json", "generation_config.json")

SHAPES = {"0b6": {"hidden": 32, "cp_hidden": 32}, "1b7": {"hidden": 64, "cp_hidden": 32}}
VARIANTS = {  # name -> (source config, size shape)
    "base06": ("Qwen3-TTS-12Hz-0.6B-Base.json", "0b6"),
    "base17": ("Qwen3-TTS-12Hz-1.7B-Base.json", "1b7"),
    "cv06": ("Qwen3-TTS-12Hz-0.6B-CustomVoice.json", "0b6"),
    "cv17": ("Qwen3-TTS-12Hz-1.7B-CustomVoice.json", "1b7"),
    "vd17": ("Qwen3-TTS-12Hz-1.7B-VoiceDesign.json", "1b7"),
    "lure": (None, "0b6"),  # the Lure family config (conformance/reference/tokenizer) plus lure.json
}
CONFIG_VARIANT = {src: name for name, (src, _) in VARIANTS.items() if src}
SEEDS = {name: 1000 + i for i, name in enumerate(VARIANTS)}
SPEAKER_ENCODER = {"mel_dim": 128, "enc_channels": [72, 72, 72, 72, 216], "enc_kernel_sizes": [5, 3, 3, 3, 1],
                   "enc_dilations": [1, 2, 3, 4, 1], "enc_attention_channels": 72, "enc_res2net_scale": 8,
                   "enc_se_channels": 72, "sample_rate": 24000}  # 1x1 convs wider than 64 keep sanitize's layout guess right


def source_config(name: str) -> dict:
    src, _ = VARIANTS[name]
    path = os.path.join(TOKDIR, "config.json") if src is None else os.path.join(REF, "configs", src)
    with open(path) as f:
        return json.load(f)


def tiny_config(name: str) -> dict:
    cfg = source_config(name)
    shape = SHAPES[VARIANTS[name][1]]
    h, ch = shape["hidden"], shape["cp_hidden"]
    tc = cfg["talker_config"]
    tc.update(hidden_size=h, intermediate_size=3 * h, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
              head_dim=16, text_hidden_size=16,
              rope_scaling={"interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default", "type": "default"})
    tc["code_predictor_config"].update(hidden_size=ch, intermediate_size=3 * ch, num_hidden_layers=2, num_attention_heads=4,
                                       num_key_value_heads=2, head_dim=16, layer_types=["full_attention"] * 2)
    if cfg["tts_model_type"] == "base":
        cfg["speaker_encoder_config"] = dict(SPEAKER_ENCODER, enc_dim=h)
    else:
        cfg.pop("speaker_encoder_config", None)
    for k in ("_name_or_path", "transformers_version"):
        cfg.pop(k, None)
    return cfg


def param_shapes(cfg: dict) -> dict:
    """{name: (shape, mlx_layout_conv)} from mlx-audio's modules for this config."""
    from mlx.utils import tree_flatten
    from mlx_audio.tts.models.qwen3_tts.config import ModelConfig
    from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model
    m = Model(ModelConfig.from_dict(cfg))
    return {k: tuple(v.shape) for k, v in tree_flatten(m.parameters())}


def is_conv(name: str, shape) -> bool:
    """The tensors mlx-audio's sanitize treats as Conv1d weights."""
    return len(shape) == 3 and ("conv" in name or "speaker_encoder.fc" in name) and "weight" in name


def _value(name: str, shape, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    n = rng.standard_normal(shape, dtype=np.float32)
    leaf = name.rsplit(".", 1)[-1]
    if "running_var" in name:
        return 1.0 + 0.1 * np.abs(n)
    if leaf == "bias" or "running_mean" in name:
        return 0.02 * n
    if len(shape) == 1:  # norm scales
        return 1.0 + 0.1 * n
    if "embedding" in name:
        return 0.5 * n
    fan_in = int(np.prod(shape[1:]))
    scale = 3.0 if ("codec_head" in name or "lm_head" in name) else 1.0  # spread the logits so greedy picks are clear
    return scale * n / np.sqrt(fan_in)


def tiny_weights(cfg: dict, seed: int) -> dict:
    """{HF name: float32 array}. Conv weights in the PyTorch layout [out, in, kernel]."""
    out = {}
    for name, shape in sorted(param_shapes(cfg).items()):
        v = _value(name, shape, seed)
        out[name] = np.ascontiguousarray(v.transpose(0, 2, 1)) if is_conv(name, shape) else v
    return out


def make_model_dir(root: str, name: str) -> str:
    """Write the tiny checkpoint `name` under root and return its path (reused when already there)."""
    import mlx.core as mx
    d = os.path.join(root, name)
    if os.path.isfile(os.path.join(d, "model.safetensors")):
        return d
    os.makedirs(d, exist_ok=True)
    cfg = tiny_config(name)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    for fn in TOKENIZER_FILES + (("lure.json",) if VARIANTS[name][0] is None else ()):
        os.symlink(os.path.join(TOKDIR, fn), os.path.join(d, fn))
    w = tiny_weights(cfg, SEEDS[name])
    mx.save_safetensors(os.path.join(d, "model.safetensors"), {k: mx.array(v) for k, v in w.items()})
    return d


def ref_codes(n: int) -> np.ndarray:
    """Deterministic reference codes [n, 16] (the symbolic codes of conformance/reference/gen_reference.py)."""
    t = np.arange(n).reshape(-1, 1)
    g = np.arange(16).reshape(1, -1)
    return ((37 * t + 101 * g + 7) % 2048).astype(np.int32)


def x_vector(dim: int, seed: int = 7) -> np.ndarray:
    return (0.5 * np.random.default_rng(seed).standard_normal(dim)).astype(np.float32)


def projection(dim: int) -> np.ndarray:
    """Fixed vector used to fingerprint embedding rows (row @ projection) in the qwen-tts fixture."""
    return np.random.default_rng(123).standard_normal(dim).astype(np.float32)


GREEDY_FRAMES = 10
LURE_ENTRIES = (("a", "icl"), ("c", "icl"), ("x.auto", "icl"), ("c", "xvec"))  # prompt_ids.json ids, Lure family


def greedy_cases() -> list:
    """Inputs of the qwen-tts greedy parity cases: every prompt_modes.json entry on the tiny checkpoint of its config,
    plus Lure-family entries from prompt_ids.json (icl with and without an instruct, auto language, and xvec)."""
    out = []
    with open(os.path.join(REF, "prompt_modes.json")) as f:
        modes = json.load(f)["entries"]
    with open(os.path.join(REF, "prompt_ids.json")) as f:
        lure = {e["id"]: e for e in json.load(f)["entries"]}
    picked = [(CONFIG_VARIANT[e["config"]], e, e["mode"], e["prompt_layout"], e["id"]) for e in modes]
    picked += [("lure", lure[i], mode, "non_streaming", f"lure.{i}.{mode}") for i, mode in LURE_ENTRIES]
    for variant, e, mode, layout, cid in picked:
        hidden = SHAPES[VARIANTS[variant][1]]["hidden"]
        icl = mode == "icl"
        clone = mode in ("icl", "xvec")
        n_ref = int(e["n_ref_frames"]) if icl else 0
        out.append({
            "id": cid, "variant": variant, "mode": mode, "layout": layout, "language": e["language"].lower(),
            "speaker": e.get("speaker"), "input_ids": e["input_ids"], "ref_ids": e["ref_ids"] if icl else None,
            "instruct_ids": e["instruct_ids"], "ref_text": e["ref_text"] if icl else "",
            "ref_codes": ref_codes(n_ref).tolist() if icl else None,
            "x_vector": x_vector(hidden).tolist() if clone else None, "max_new_tokens": GREEDY_FRAMES,
        })
    return out
