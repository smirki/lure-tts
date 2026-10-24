"""lure_tts.model.inspect over the stock Qwen3-TTS configs and the Lure family markers."""
import json
import os

import pytest

from lure_tts.model import describe, inspect
from conftest import STOCK, TOKDIR, make_model_dir

LURE_JSON = {"format": "lure-model-v1", "family": "lure", "prompt_layout": "non_streaming", "inline_tags": True}


@pytest.mark.parametrize("key,variant,size,hidden,enc,layout,instruct,modes", [
    ("base06", "base", "0b6", 1024, 1024, "streaming", "none", ("icl", "xvec")),
    ("base17", "base", "1b7", 2048, 2048, "streaming", "none", ("icl", "xvec")),
    ("cv06", "custom_voice", "0b6", 1024, None, "non_streaming", "none", ("custom_voice",)),
    ("cv17", "custom_voice", "1b7", 2048, None, "non_streaming", "optional", ("custom_voice",)),
    ("vd17", "voice_design", "1b7", 2048, None, "non_streaming", "required", ("voice_design",)),
])
def test_stock_specs(tmp_path, key, variant, size, hidden, enc, layout, instruct, modes):
    s = inspect(make_model_dir(tmp_path, STOCK[key]))
    assert (s.family, s.variant, s.size, s.talker_hidden_size, s.speaker_encoder_dim) == ("qwen3_tts", variant, size, hidden, enc)
    assert s.code_predictor_hidden_size == 1024 and s.num_code_groups == 16
    assert (s.layout, s.instruct, s.modes, s.inline_tags, s.default_language) == (layout, instruct, modes, False, "auto")
    assert s.languages[0] == "auto" and "english" in s.languages and not any("dialect" in x for x in s.languages)
    assert s.turbo is None and s.name.startswith("Qwen3-TTS-12Hz-")
    assert s.sampling.repetition_penalty == 1.05 and s.max_new_tokens == 8192
    if variant == "custom_voice":
        assert "vivian" in s.speakers and s.speaker_dialects == {"eric": "sichuan_dialect", "dylan": "beijing_dialect"}
    else:
        assert s.speakers == () and s.speaker_dialects == {}
    assert describe(s).startswith(s.name)
    json.dumps(s.to_dict())


def test_lure_family(tmp_path):
    s = inspect(make_model_dir(tmp_path, STOCK["base06"], lure_json=dict(LURE_JSON, name="Lure Custom")))
    assert (s.family, s.name, s.layout, s.inline_tags, s.instruct, s.default_language) == (
        "lure", "Lure Custom", "non_streaming", True, "optional", "english")
    assert inspect(TOKDIR).family == "lure"


def test_turbo_json_alone_marks_lure(tmp_path):
    s = inspect(make_model_dir(tmp_path, STOCK["base06"], turbo=True))
    assert s.family == "lure" and s.name == "Lure Turbo" and s.turbo.n_passes == 3 and s.layout == "non_streaming"
    assert inspect(make_model_dir(tmp_path / "x", STOCK["base06"], turbo=True), turbo=0).turbo is None


def test_overrides(tmp_path):
    d = make_model_dir(tmp_path, STOCK["base06"])
    s = inspect(d, layout="non-streaming", inline_tags=True)
    assert s.layout == "non_streaming" and s.inline_tags
    assert inspect(TOKDIR, layout="streaming", inline_tags=False).layout == "streaming"
    with pytest.raises(ValueError):
        inspect(d, layout="chunky")


def test_generation_config_overrides_defaults(tmp_path):
    d = make_model_dir(tmp_path, STOCK["cv17"])
    with open(os.path.join(d, "generation_config.json"), "w") as f:
        json.dump({"temperature": 0.7, "subtalker_top_k": 20, "subtalker_dosample": False}, f)
    s = inspect(d).sampling
    assert (s.temperature, s.subtalker_top_k, s.subtalker_do_sample, s.top_k) == (0.7, 20, False, 50)
    os.remove(os.path.join(d, "generation_config.json"))
    assert inspect(d).sampling.temperature == 0.9


@pytest.mark.parametrize("bad", [
    {"format": "lure-model-v2"},
    {"format": "lure-model-v1", "family": "other"},
    {"format": "lure-model-v1", "prompt_layout": "zigzag"},
    {"format": "lure-model-v1", "inline_tags": "yes"},
])
def test_bad_lure_json(tmp_path, bad):
    with pytest.raises(ValueError):
        inspect(make_model_dir(tmp_path, STOCK["base06"], lure_json=bad))


def test_lure_json_on_non_base_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        inspect(make_model_dir(tmp_path, STOCK["cv17"], lure_json=LURE_JSON))


def test_rejects_other_tokenizers_and_missing_config(tmp_path):
    d = make_model_dir(tmp_path, STOCK["base06"])
    cfg = json.load(open(os.path.join(d, "config.json")))
    cfg["tokenizer_type"] = "qwen3_tts_tokenizer_25hz"
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    with pytest.raises(ValueError):
        inspect(d)
    with pytest.raises(FileNotFoundError):
        inspect(str(tmp_path / "nothing"))
