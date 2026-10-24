import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(ROOT, "conformance", "reference")
TOKDIR = os.path.join(REF, "tokenizer")
sys.path.insert(0, os.path.join(ROOT, "conformance"))


@pytest.fixture(scope="session")
def tokdir():
    return TOKDIR


@pytest.fixture(scope="session")
def tok():
    from lure_tts.tokenizer import QwenTokenizer
    return QwenTokenizer.from_dir(TOKDIR)


CONFIGS = os.path.join(REF, "configs")
STOCK = {
    "base06": "Qwen3-TTS-12Hz-0.6B-Base.json", "base17": "Qwen3-TTS-12Hz-1.7B-Base.json",
    "cv06": "Qwen3-TTS-12Hz-0.6B-CustomVoice.json", "cv17": "Qwen3-TTS-12Hz-1.7B-CustomVoice.json",
    "vd17": "Qwen3-TTS-12Hz-1.7B-VoiceDesign.json",
}


def make_model_dir(root, config_name, lure_json=None, turbo=False):
    """A weightless model folder: a stock config plus the reference tokenizer files."""
    import json
    import shutil
    d = os.path.join(str(root), os.path.splitext(config_name)[0])
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(CONFIGS, config_name), os.path.join(d, "config.json"))
    for f in ("vocab.json", "merges.txt", "tokenizer_config.json", "generation_config.json"):
        shutil.copy(os.path.join(TOKDIR, f), os.path.join(d, f))
    if lure_json is not None:
        with open(os.path.join(d, "lure.json"), "w") as fh:
            json.dump(lure_json, fh)
    if turbo:
        with open(os.path.join(d, "turbo_head.safetensors"), "wb") as fh:
            fh.write(b"x")
        with open(os.path.join(d, "turbo.json"), "w") as fh:
            json.dump({"format": "lure-turbo-v1", "groups": [[1, 3], [4, 7], [8, 15]]}, fh)
    return d


@pytest.fixture()
def stock_tts(tmp_path):
    """stock_tts("cv17") -> Lure on the fake backend over that stock config."""
    from lure_tts.api import Lure
    made = []

    def make(key, **kw):
        tts = Lure.load(make_model_dir(tmp_path / "models", STOCK[key]), backend="fake",
                          cache_dir=str(tmp_path / "cache"), **kw)
        made.append(tts)
        return tts
    yield make
    for t in made:
        t.close()


@pytest.fixture()
def fake_tts(tmp_path):
    from lure_tts.api import Lure
    tts = Lure.load(TOKDIR, backend="fake", cache_dir=str(tmp_path))
    yield tts
    tts.close()
