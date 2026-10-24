"""lure_tts.turbo: Turbo model-folder detection and the turbo / turbo_head option rules."""
import json

import pytest

from lure_tts import turbo as T


def _turbo_dir(tmp_path, **over):
    d = tmp_path / "m"
    d.mkdir()
    (d / "turbo_head.safetensors").write_bytes(b"x")
    cfg = {"format": "lure-turbo-v1", "groups": [[1, 3], [4, 7], [8, 15]], "head_file": "turbo_head.safetensors",
           "addon": {"hidden": 512, "layernorm_eps": 1e-5}, "sampling": {"temperature": 0.9, "top_k": 50, "top_p": 1.0}}
    cfg.update(over)
    (d / "turbo.json").write_text(json.dumps(cfg))
    return d


def test_plain_dir_is_lure(tmp_path):
    assert T.resolve(str(tmp_path)) is None
    assert T.resolve(str(tmp_path), turbo="auto") is None
    assert T.resolve(str(tmp_path), turbo=0) is None
    with pytest.raises(FileNotFoundError):
        T.resolve(str(tmp_path), turbo=1)


def test_turbo_dir(tmp_path):
    d = _turbo_dir(tmp_path)
    c = T.resolve(str(d))
    assert c.groups == ((1, 3), (4, 7), (8, 15)) and c.n_passes == 3
    assert c.head_path == str(d / "turbo_head.safetensors")
    assert (c.temperature, c.top_k, c.top_p, c.addon_hidden, c.layernorm_eps) == (0.9, 50, 1.0, 512, 1e-5)
    assert T.resolve(str(d), turbo="1") == c
    assert T.resolve(str(d), turbo=False) is None
    assert T.is_turbo_dir(str(d))


def test_head_override_and_validation(tmp_path):
    d = _turbo_dir(tmp_path)
    other = tmp_path / "head_s0060.safetensors"
    other.write_bytes(b"y")
    assert T.resolve(str(d), turbo_head=str(other)).head_path == str(other)
    with pytest.raises(FileNotFoundError):
        T.resolve(str(d), turbo_head=str(tmp_path / "missing.safetensors"))
    d2 = _turbo_dir(tmp_path / "x", groups=[[1, 3], [5, 15]]) if (tmp_path / "x").mkdir() is None else None
    with pytest.raises(ValueError):
        T.resolve(str(d2))
    with pytest.raises(ValueError):
        T.parse_flag("maybe")
