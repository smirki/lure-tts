"""Lure Turbo head on MLX vs a float64 reference implementation, tiny random weights only.

The fixture conformance/turbo/fixtures/tiny_turbo.safetensors (random head weights, inputs, expected logits and codes)
was computed in float64 by the reference implementation of the Turbo head and stored as float32.
Run: python -m pytest backends/mlx/tests/test_turbo_tiny.py
"""
import json
import os
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from lure_mlx.turbo import TurboCodePredictor, TurboHead
from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSTalkerCodePredictorConfig

from test_mlx_tiny import TOKDIR, codes_of, make_backend, tiny_talker
from lure_tts.tokenizer import QwenTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                   "conformance", "turbo", "fixtures")


@pytest.fixture(autouse=True)
def cpu_for_reference(request):
    """Reference parity runs on the mlx CPU device: Metal float32 matmuls are not full fp32 (logits differ ~1e-2 on the
    GPU, ~5e-5 on the CPU), which would hide real bugs behind a loose tolerance. The math is device-independent."""
    if "fx" not in request.fixturenames:
        yield
        return
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


@pytest.fixture(scope="module")
def fx():
    meta = json.load(open(os.path.join(FIX, "tiny_turbo.json")))
    w = mx.load(os.path.join(FIX, "tiny_turbo.safetensors"))
    cpc = Qwen3TTSTalkerCodePredictorConfig(hidden_size=meta["hidden"], intermediate_size=meta["inter"], num_hidden_layers=meta["layers"],
                                            num_attention_heads=meta["heads"], num_key_value_heads=meta["kv_heads"], head_dim=meta["head_dim"],
                                            vocab_size=meta["vocab"], rope_theta=meta["rope_theta"], rms_norm_eps=meta["rms_norm_eps"])
    head = TurboHead(cpc, meta["hidden"], meta["addon_hidden"], meta["layernorm_eps"], meta["groups"])
    head.load_weights([(k[5:], v) for k, v in w.items() if k.startswith("head.")], strict=True)
    cemb = []
    for i in range(15):
        e = nn.Embedding(meta["vocab"], meta["hidden"])
        e.weight = w[f"cemb.{i}.weight"]
        cemb.append(e)
    stub = types.SimpleNamespace(code_predictor=types.SimpleNamespace(codec_embedding=cemb),
                                 codec_head=types.SimpleNamespace(weight=mx.zeros((1,), dtype=mx.float32)))
    return meta, w, head, stub


def test_head_keys_cover_fixture(fx):
    meta, w, head, _ = fx
    from mlx.utils import tree_flatten
    names = {k for k, _ in tree_flatten(head.parameters())}
    assert names == {k[5:] for k in w if k.startswith("head.")} == set(k[5:] for k in meta["keys"])


def test_forced_logits_match_reference(fx):
    _, w, head, stub = fx
    tc = TurboCodePredictor(stub, head, 0.9, 50, 1.0)
    codes, logits = tc.codes(w["x2"], forced=w["forced_codes"])
    assert np.array(codes).tolist() == np.array(w["forced_codes"]).tolist()
    np.testing.assert_allclose(np.array(logits), np.array(w["forced_logits"]), atol=2e-4, rtol=1e-4)


def test_greedy_codes_match_reference(fx):
    _, w, head, stub = fx
    tc = TurboCodePredictor(stub, head, 0.9, 50, 1.0, do_sample=False)
    codes, logits = tc.codes(w["x2"])
    assert np.array(codes).tolist() == np.array(w["greedy_codes"]).tolist()
    np.testing.assert_allclose(np.array(logits), np.array(w["greedy_logits"]), atol=2e-4, rtol=1e-4)


def test_addon_matters(fx):
    """Sanity: zeroing the add-on output changes in-group logits but not group-first ones."""
    _, w, head, stub = fx
    tc = TurboCodePredictor(stub, head, 0.9, 50, 1.0)
    _, base = tc.codes(w["x2"], forced=w["forced_codes"])
    saved = head.addon.w2.weight
    head.addon.w2.weight = mx.zeros_like(saved)
    head.addon.w2.bias, sb = mx.zeros_like(head.addon.w2.bias), head.addon.w2.bias
    try:
        _, off = tc.codes(w["x2"], forced=w["forced_codes"])
    finally:
        head.addon.w2.weight, head.addon.w2.bias = saved, sb
    d = np.abs(np.array(base) - np.array(off)).max(-1)[0]
    firsts = [0, 3, 7]
    assert np.all(d[firsts] < 1e-5) and np.all(np.delete(d, firsts) > 1e-3)


# ---------------------------------------------------------------------------------------------------- with a tiny talker
def random_head(talker, seed=1):
    cpc = talker.code_predictor.config
    head = TurboHead(cpc, talker.config.hidden_size)
    mx.random.seed(seed)
    from mlx.utils import tree_flatten, tree_unflatten
    ps = []
    for k, v in tree_flatten(head.parameters()):
        s = 0.5 if k.startswith("lm_head") else 0.15
        is_norm = k.endswith("norm.weight") or k in ("addon.n1.weight", "addon.n2.weight", "addon.n3.weight")
        ps.append((k, (1.0 + 0.1 * mx.random.normal(v.shape)) if is_norm else s * mx.random.normal(v.shape)))
    head.update(tree_unflatten(ps))
    mx.eval(head.parameters())
    return head


@pytest.fixture(scope="module")
def talker():
    return tiny_talker()


@pytest.mark.parametrize("do_sample", [False, True])
def test_compiled_matches_eager(talker, do_sample):
    t, _ = talker
    head = random_head(t)
    eager = TurboCodePredictor(t, head, 0.9, 50, 1.0, do_sample)
    comp = TurboCodePredictor(t, head, 0.9, 50, 1.0, do_sample, compile=True)
    for i in range(3):
        mx.random.seed(20 + i)
        h = mx.random.normal((1, 1, 64))
        cb0 = mx.array([31 * (i + 1)], dtype=mx.int32)
        noise = mx.random.gumbel((15, 2048))
        a, ae = eager(h, cb0, noise)
        b, be_ = comp(h, cb0, noise)
        assert np.array(a).tolist() == np.array(b).tolist()
        np.testing.assert_allclose(np.array(ae), np.array(be_), atol=1e-4)
        assert a.shape == (15,) and ae.shape == (1, 1, 64)


def test_frame_equals_codes_path(talker):
    """__call__ (backend path) == codes() on x2 = [h, E0(cb0)] with the same noise; emb_sum = C(cb0) + sum cemb."""
    t, _ = talker
    head = random_head(t, 2)
    tc = TurboCodePredictor(t, head, 0.9, 50, 1.0)
    mx.random.seed(5)
    h = mx.random.normal((1, 1, 64))
    cb0 = mx.array([77], dtype=mx.int32)
    noise = mx.random.gumbel((15, 2048))
    a, emb = tc(h, cb0, noise)
    x2 = mx.concatenate([h, t.get_input_embeddings()(cb0[None])], axis=1)
    b, _ = tc.codes(x2, noise[:, None, :])
    assert np.array(a).tolist() == np.array(b[0]).tolist()
    want = t.get_input_embeddings()(cb0[None]).astype(mx.float32)
    for g in range(15):
        want = want + t.code_predictor.codec_embedding[g](a[g:g + 1][None]).astype(mx.float32)
    np.testing.assert_allclose(np.array(emb), np.array(want), atol=1e-5)


def turbo_dir(tmp_path, talker, with_json=True):
    d = tmp_path / ("turbo" if with_json else "plain")
    d.mkdir()
    for f in os.listdir(TOKDIR):
        os.symlink(os.path.join(TOKDIR, f), d / f)
    if with_json:
        head = random_head(talker)
        from mlx.utils import tree_flatten
        mx.save_safetensors(str(d / "turbo_head.safetensors"), dict(tree_flatten(head.parameters())))
        json.dump({"format": "lure-turbo-v1", "head_file": "turbo_head.safetensors", "groups": [[1, 3], [4, 7], [8, 15]],
                   "hidden_size": 64, "num_layers": 2, "addon": {"hidden": 512, "layernorm_eps": 1e-5},
                   "sampling": {"temperature": 0.9, "top_k": 50, "top_p": 1.0}}, open(d / "turbo.json", "w"))
    return str(d)


def test_backend_selects_turbo(tmp_path, talker):
    t, _ = talker
    tok = QwenTokenizer.from_dir(TOKDIR)
    plain_be, _ = make_backend(talker)
    plain_be.model_dir = turbo_dir(tmp_path, t, with_json=False)
    assert plain_be._setup_turbo() is None and not plain_be.turbo
    with pytest.raises(FileNotFoundError):
        plain_be._setup_turbo(turbo="1")
    ref_codes, _ = codes_of(plain_be, tok)

    d = turbo_dir(tmp_path, t)
    outs = {}
    for opts in ({}, {"compile_cp": True}, {"pipeline": True}):
        be, _ = make_backend(talker, **opts)
        be.model_dir = d
        assert be._setup_turbo() is not None and be.turbo
        got, out = codes_of(be, tok)
        assert out[-1].info.frames == 10 and out[-1].info.extra["turbo"]
        outs[tuple(opts)] = [c.tolist() for c in got]
        assert outs[tuple(opts)] != [c.tolist() for c in ref_codes]  # the Turbo head is actually used
    assert len({json.dumps(v) for v in outs.values()}) == 1  # eager == compiled == pipelined
    be, _ = make_backend(talker)
    be.model_dir = d
    assert be._setup_turbo(turbo="0") is None and not be.turbo
    got, _ = codes_of(be, tok)
    assert [c.tolist() for c in got] == [c.tolist() for c in ref_codes]
