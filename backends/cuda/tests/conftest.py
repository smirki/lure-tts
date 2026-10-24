"""CPU-only fixtures: tiny randomly initialised Qwen3-TTS models (the real architecture code, toy sizes, float64).
Nothing here loads real weights.

- tiny_model: the Base layout with talker hidden 64 != code predictor hidden 32 (the 1.7B structure).
- tiny_variant(name): a tiny model built from the stock config.json of any checkpoint in
  conformance/reference/configs (0.6B-Base, 0.6B-CustomVoice, 1.7B-Base, 1.7B-CustomVoice, 1.7B-VoiceDesign) with the
  ids, speakers and languages kept. 0.6B-shaped: talker hidden == code predictor hidden == 32, so small_to_mtp is the
  identity. 1.7B-shaped: talker hidden 64, code predictor hidden 32, so small_to_mtp is a real Linear(64 -> 32).
  Base variants get a speaker encoder with enc_dim == talker hidden.

MODEL_DIR (config.json and tokenizer files for the prompt checks) is LURE_MODEL_DIR, else models/lure, else the
copies in conformance/reference/tokenizer."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(BACKEND))
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(REPO, "core"))


def _model_dir():
    for d in (os.environ.get("LURE_MODEL_DIR"), os.path.join(REPO, "models", "lure"),
              os.path.join(REPO, "conformance", "reference", "tokenizer")):
        if d and os.path.exists(os.path.join(d, "config.json")):
            return os.path.abspath(d)
    return os.path.abspath(os.path.join(REPO, "models", "lure"))


MODEL_DIR = _model_dir()
REFERENCE = os.path.join(REPO, "conformance", "reference", "prompt_ids.json")
MODES_REFERENCE = os.path.join(REPO, "conformance", "reference", "prompt_modes.json")
CONFIGS = os.path.join(REPO, "conformance", "reference", "configs")
VARIANTS = ("0.6B-Base", "0.6B-CustomVoice", "1.7B-Base", "1.7B-CustomVoice", "1.7B-VoiceDesign")
UP = 1920  # codec samples per frame


def tiny_config():
    from lure_cuda._compat import import_qwen_tts

    import_qwen_tts()
    from qwen_tts.core.models import Qwen3TTSConfig

    with open(os.path.join(MODEL_DIR, "config.json")) if os.path.exists(os.path.join(MODEL_DIR, "config.json")) else _fallback() as f:
        c = json.load(f)
    t = c["talker_config"]
    cp = t["code_predictor_config"]
    t.update(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
             head_dim=16, text_hidden_size=32, rope_scaling={"interleaved": True, "mrope_section": [4, 2, 2],
                                                            "rope_type": "default", "type": "default"})
    cp.update(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
              head_dim=16, layer_types=["full_attention"] * 2)
    c["speaker_encoder_config"] = {"enc_dim": 64, "sample_rate": 24000, "enc_channels": [16, 16, 16, 16, 48]}
    for k in ("_name_or_path", "transformers_version", "architectures"):
        c.pop(k, None)
    return Qwen3TTSConfig(**c)


class _fallback:
    """Minimal config used when no checkpoint is available."""

    def __enter__(self):
        import io

        cfg = {
            "tts_bos_token_id": 151672, "tts_eos_token_id": 151673, "tts_pad_token_id": 151671,
            "im_start_token_id": 151644, "im_end_token_id": 151645, "assistant_token_id": 77091,
            "tokenizer_type": "qwen3_tts_tokenizer_12hz", "tts_model_size": "0b6", "tts_model_type": "base",
            "talker_config": {
                "codec_bos_id": 2149, "codec_eos_token_id": 2150, "codec_think_id": 2154, "codec_nothink_id": 2155,
                "codec_pad_id": 2148, "codec_think_bos_id": 2156, "codec_think_eos_id": 2157,
                "codec_language_id": {"english": 2050, "chinese": 2055}, "num_code_groups": 16, "vocab_size": 3072,
                "text_vocab_size": 151936, "rope_theta": 1000000, "rms_norm_eps": 1e-6, "hidden_act": "silu",
                "max_position_embeddings": 32768, "sliding_window": None, "use_sliding_window": False,
                "spk_id": {}, "spk_is_dialect": {}, "position_id_per_seconds": 13, "attention_bias": False,
                "attention_dropout": 0,
                "code_predictor_config": {"num_code_groups": 16, "vocab_size": 2048, "rope_theta": 1000000,
                                          "rms_norm_eps": 1e-6, "hidden_act": "silu", "max_position_embeddings": 65536,
                                          "sliding_window": None, "use_sliding_window": False,
                                          "attention_bias": False, "attention_dropout": 0},
            },
        }
        return io.StringIO(json.dumps(cfg))

    def __exit__(self, *a):
        return False


@pytest.fixture(scope="session")
def tiny_model():
    return build_tiny(tiny_config(), 1234)


def _shrink(c: dict) -> dict:
    """Toy sizes with the stock structure: 1.7B keeps talker hidden != code predictor hidden."""
    t = c["talker_config"]
    cp = t["code_predictor_config"]
    H = 64 if c.get("tts_model_size") == "1b7" else 32
    t.update(hidden_size=H, intermediate_size=2 * H, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
             head_dim=16, text_hidden_size=32, rope_scaling={"interleaved": True, "mrope_section": [4, 2, 2],
                                                            "rope_type": "default", "type": "default"})
    cp.update(hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
              head_dim=16, layer_types=["full_attention"] * 2)
    if c.get("tts_model_type") == "base":
        c["speaker_encoder_config"] = {"enc_dim": H, "sample_rate": 24000, "enc_channels": [16, 16, 16, 16, 48]}
    for k in ("_name_or_path", "transformers_version", "architectures"):
        c.pop(k, None)
    return c


def tiny_variant_config(name: str):
    from lure_cuda._compat import import_qwen_tts

    import_qwen_tts()
    from qwen_tts.core.models import Qwen3TTSConfig

    with open(os.path.join(CONFIGS, f"Qwen3-TTS-12Hz-{name}.json")) as f:
        return Qwen3TTSConfig(**_shrink(json.load(f)))


def build_tiny(cfg, seed: int, text_gain: float = 1.0):
    import torch

    from lure_cuda._compat import import_qwen_tts

    import_qwen_tts()
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration

    torch.manual_seed(seed)
    cfg._attn_implementation = "sdpa"
    m = Qwen3TTSForConditionalGeneration(cfg).to(torch.float64).eval()
    # random init is small; widen logits a bit so sampling is not uniform noise
    with torch.no_grad():
        m.talker.codec_head.weight.mul_(20)
        for h in m.talker.code_predictor.lm_head:
            h.weight.mul_(20)
        if text_gain != 1.0:  # random init leaves T() ~700x below C(); make the text side steer the output
            m.talker.model.text_embedding.weight.mul_(text_gain)
            m.talker.text_projection.linear_fc2.weight.mul_(text_gain)
    return m


@pytest.fixture(scope="session")
def tiny_variant():
    cache = {}

    def get(name: str):
        if name not in cache:
            cache[name] = build_tiny(tiny_variant_config(name), 1000 + VARIANTS.index(name), text_gain=30.0)
        return cache[name]
    return get


def make_prompt(text_ids, ref_text_ids=None, instruct_ids=None):
    """PromptIds from bare token ids. ref_text_ids None (every mode but icl) gives ref_ids None."""
    from lure_cuda.prompt import PromptIds

    ins = None if instruct_ids is None else [151644, 872, 198] + list(instruct_ids) + [151645, 198]
    ref = None if ref_text_ids is None else [151644, 77091, 198] + list(ref_text_ids) + [151645, 198]
    return PromptIds([151644, 77091, 198] + list(text_ids) + [151645, 198, 151644, 77091, 198], ref, ins)


class FakeCausalDecoder:
    """Causal toy codec: frame t -> 1920 samples depending on frames t-2..t (receptive field 3 < 25)."""

    def __call__(self, codes):  # [1, 16, T]
        import torch

        c = codes[0].double()  # [16, T]
        v = c.sum(0) / 30000.0
        prev = torch.cat([torch.zeros(1, dtype=v.dtype), v[:-1]])
        prev2 = torch.cat([torch.zeros(2, dtype=v.dtype), v[:-2]])
        y = v + 0.5 * prev + 0.25 * prev2
        ramp = torch.linspace(0, 1, UP, dtype=v.dtype)
        return (y.view(-1, 1) * ramp.view(1, -1)).reshape(1, 1, -1).float()


class FakeSpeechTokenizer:
    def __init__(self):
        import types

        self.model = types.SimpleNamespace(decoder=FakeCausalDecoder(), get_decode_upsample_rate=lambda: UP)

    def decode(self, items):
        c = items[0]["audio_codes"]
        return [self.model.decoder(c.t().unsqueeze(0)).reshape(-1)], 24000


def make_voice_tensors(n_ref=7, hidden=64, seed=0):
    import torch

    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 2048, (n_ref, 16), generator=g)
    spk = torch.randn(hidden, generator=g, dtype=torch.float64)
    return codes, spk
