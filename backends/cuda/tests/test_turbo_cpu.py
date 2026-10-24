"""Lure Turbo head on CPU with tiny random weights: the engine's head must equal the reference implementation.

The reference is conformance/turbo/ref_fastcp.py. It calls create_causal_mask(input_embeds=...), the transformers 4.57
keyword; under transformers 5 it runs through a keyword shim (renamed or dropped kwargs only, no math change), and under
4.57 it runs as written.

1. Live, float64: same logits at every position along forced codes, same greedy codes (temperature 0 = argmax), batch 3.
2. Fixture made by the reference on transformers 4.57.3 + qwen-tts 0.1.1: same codes and logits.
3. TurboPredictorGraph (the engine's batched step, per-row sampling) == TurboHead greedy, and sampled rows match a
   reference run fed the same Gumbel noise.
4. Whole engine with Turbo (greedy) == qwen-tts generate() with talker.code_predictor.generate patched to the reference
   head, on a tiny Qwen3-TTS whose CP hidden equals the talker hidden (as in the 0.6B).
5. Backend selection: turbo.json present -> Turbo, absent -> plain Lure; turbo=0 / turbo=1 rules.
"""
import json
import os
import sys
import types

import pytest
import torch

from conftest import REPO, make_prompt, make_voice_tensors, tiny_config

TURBO_REF = os.path.join(REPO, "conformance", "turbo")
sys.path.insert(0, TURBO_REF)
FIXTURE = os.path.join(TURBO_REF, "fixtures", "tiny_turbo.safetensors")
GROUPS = [(1, 3), (4, 7), (8, 15)]


@pytest.fixture(autouse=True)
def _mask_shim(monkeypatch):
    import inspect

    import transformers.masking_utils as mu
    orig = mu.create_causal_mask
    if "input_embeds" in inspect.signature(orig).parameters:  # transformers 4.57: the reference runs as written
        return

    def shim(*a, input_embeds=None, **kw):
        if input_embeds is not None:
            kw["inputs_embeds"] = input_embeds
        kw.pop("cache_position", None)  # transformers 5 derives it
        return orig(*a, **kw)
    monkeypatch.setattr(mu, "create_causal_mask", shim)


def _ref_and_engine_heads(dtype=torch.float64):
    import make_fixture as M
    from lure_cuda.turbo import TurboHead
    ref, cp, emb0 = M.build_reference(dtype)
    mine = TurboHead(cp, GROUPS).to(dtype).eval()
    mine.load_head(ref.state_dict())
    return ref, mine, cp, emb0


def _x2(emb0, B=3, seed=3):
    g = torch.Generator().manual_seed(seed)
    H = emb0.weight.shape[1]
    h = torch.randn(B, H, generator=g, dtype=emb0.weight.dtype)
    return torch.stack([h, emb0(torch.randint(0, 2048, (B,), generator=g))], 1).detach()


def _mine_logits(mine, x2, forced=None):
    logs = []

    def sample(logits, i):
        logs.append(logits.clone())
        return forced[:, i].clone() if forced is not None else logits.argmax(-1)
    with torch.no_grad():
        codes = mine.frame(x2, sample, mine.make_static(x2.device, x2.dtype))
    return codes, torch.stack(logs, 1)


def test_live_reference_parity_float64():
    import make_fixture as M
    ref, mine, cp, emb0 = _ref_and_engine_heads()
    x2 = _x2(emb0)
    rc, rl = M.run_reference(ref, x2)
    mc, ml = _mine_logits(mine, x2)
    assert torch.equal(rc, mc)
    assert torch.allclose(rl, ml, atol=1e-10, rtol=1e-9)
    forced = torch.randint(0, 512, (3, 15), generator=torch.Generator().manual_seed(9))
    _, rl2 = M.run_reference(ref, x2, forced)
    mc2, ml2 = _mine_logits(mine, x2, forced)
    assert torch.equal(mc2, forced)
    assert torch.allclose(rl2, ml2, atol=1e-10, rtol=1e-9)
    # in-group logits depend on the codes picked before them, so the add-on is exercised
    assert (rl2[:, 1] - rl[:, 1]).abs().max() > 1e-3


def test_shared_fixture_parity():
    from safetensors.torch import load_file
    from lure_cuda.turbo import TurboHead
    fx = load_file(FIXTURE)
    _, _, cp, _ = _ref_and_engine_heads(torch.float32)
    with torch.no_grad():
        for i, e in enumerate(cp.model.codec_embedding):
            e.weight.copy_(fx[f"cemb.{i}.weight"])
    mine = TurboHead(cp, GROUPS).float().eval()
    mine.load_head({k[5:]: v for k, v in fx.items() if k.startswith("head.")})
    gc, gl = _mine_logits(mine, fx["x2"])
    assert torch.equal(gc.int(), fx["greedy_codes"])
    assert torch.allclose(gl, fx["greedy_logits"], atol=2e-4, rtol=1e-4)
    fc, fl = _mine_logits(mine, fx["x2"], fx["forced_codes"].long())
    assert torch.allclose(fl, fx["forced_logits"], atol=2e-4, rtol=1e-4)


def test_predictor_graph_rows():
    """Engine step object: greedy rows == head greedy; sampled rows == reference fed the same noise (Gumbel-max)."""
    import make_fixture as M
    import ref_fastcp as R
    from lure_cuda.turbo import TurboPredictorGraph
    ref, mine, cp, emb0 = _ref_and_engine_heads()
    B = 4
    x2 = _x2(emb0, B=B, seed=21)
    pg = TurboPredictorGraph(mine, x2.shape[-1], B, device="cpu", dtype=torch.float64, use_graph=False)
    g = torch.Generator().manual_seed(5)
    pg.uniform.uniform_(generator=g)
    for b in range(B):
        pg.set_row_params(b, 0.9, 50, 1.0, do_sample=(b % 2 == 0))
    got = pg.run(x2)
    rc, _ = M.run_reference(ref, x2)
    assert torch.equal(got[1::2], rc[1::2])  # greedy rows
    # sampled rows: reference sample() with the same uniforms -> identical codes
    from lure_cuda.sampling import gumbel_from_uniform
    calls = []
    orig = R.sample

    def ref_sample(logits, temp, top_k=50):
        i = len(calls)
        calls.append(i)
        lg = logits.float() / temp
        v = lg.topk(top_k, -1).values[..., -1:]
        lg = lg.masked_fill(lg < v, float("-inf"))
        return (lg + gumbel_from_uniform(pg.uniform[:, i])).argmax(-1)
    R.sample = ref_sample
    try:
        rs = ref.sample_codes(x2, R.temp_fn("0.9"))
    finally:
        R.sample = orig
    assert torch.equal(got[0::2], rs[0::2])


# ------------------------------------------------------------------------------------------------- whole engine
@pytest.fixture(scope="module")
def tiny_turbo_model():
    from lure_cuda._compat import import_qwen_tts
    import_qwen_tts()
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
    cfg = tiny_config()
    cpc = cfg.talker_config.code_predictor_config
    cpc.hidden_size, cpc.num_attention_heads, cpc.num_key_value_heads, cpc.head_dim = 64, 4, 2, 16
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(77)
    m = Qwen3TTSForConditionalGeneration(cfg).to(torch.float64).eval()
    with torch.no_grad():
        m.talker.codec_head.weight.mul_(20)
    return m


def _turbo_heads_for(m):
    import make_fixture as M
    import ref_fastcp as R
    from lure_cuda.turbo import TurboHead
    cp, e0 = m.talker.code_predictor, m.talker.get_input_embeddings()
    ref = R.make_fastcp(cp, e0, R.HEADS["turbo"], addon=True).to(torch.float64).eval()
    M.randomize_head(ref, 123)
    mine = TurboHead(cp, GROUPS).to(torch.float64).eval()
    mine.load_head(ref.state_dict())
    return ref, mine


TEXTS = [([1001, 2002, 3003, 4004, 5005], [11, 12, 13, 14], None), ([7, 8, 9], [21, 22, 23, 24, 25, 26], [301, 302, 303])]


@pytest.mark.parametrize("case", TEXTS)
def test_engine_turbo_greedy_equals_reference_generate(tiny_turbo_model, case):
    import ref_fastcp as R
    from lure_cuda.engine import BatchEngine, Request, drain
    from lure_cuda.prompt import build_prefill
    from lure_cuda.sampling import SamplingParams, make_generator
    m = tiny_turbo_model
    ref, mine = _turbo_heads_for(m)
    text, rtext, ins = case
    codes, spk = make_voice_tensors(n_ref=7, hidden=64, seed=len(text))
    p = make_prompt(text, rtext, ins)
    emb, _tr, pad = build_prefill(m, p, codes, spk, "english")
    sp = SamplingParams(do_sample=False, subtalker_do_sample=False)
    eng = BatchEngine(m.talker, m.config.talker_config, max_batch=3, max_seq_len=256, device="cpu", dtype=torch.float64,
                      use_graphs=False, turbo_head=mine)
    assert eng.turbo
    r = Request(prefill=emb, tts_pad=pad, sampling=sp, max_new_tokens=20, generator=make_generator(0, "cpu"))
    eng.submit(r)
    eng.run_until_idle()
    got = list(drain(r, timeout=0))

    cp = m.talker.code_predictor
    orig = cp.generate
    cp.generate = lambda inputs_embeds=None, **kw: types.SimpleNamespace(
        sequences=ref.sample_codes(inputs_embeds, R.temp_fn("0")))
    try:
        vcp = {"ref_code": [codes], "ref_spk_embedding": [spk], "x_vector_only_mode": [False], "icl_mode": [True]}
        with torch.no_grad():
            out, _ = m.generate(input_ids=[torch.tensor([p.input_ids])], instruct_ids=[torch.tensor([p.instruct_ids]) if ins else None],
                                ref_ids=[torch.tensor([p.ref_ids])], voice_clone_prompt=vcp, languages=["english"],
                                non_streaming_mode=True, max_new_tokens=20, do_sample=False, subtalker_dosample=False,
                                repetition_penalty=1.05)
    finally:
        cp.generate = orig
    want = out[0].tolist()
    assert len(got) > 3
    assert got == want[:len(got)]
    # and Turbo really differs from the stock 15-pass CP on this model
    eng2 = BatchEngine(m.talker, m.config.talker_config, max_batch=3, max_seq_len=256, device="cpu", dtype=torch.float64,
                       use_graphs=False)
    r2 = Request(prefill=emb, tts_pad=pad, sampling=sp, max_new_tokens=20, generator=make_generator(0, "cpu"))
    eng2.submit(r2)
    eng2.run_until_idle()
    assert [f[1:] for f in drain(r2, timeout=0)][:3] != [f[1:] for f in got][:3]


def test_seeded_turbo_batch_invariance(tiny_turbo_model):
    from lure_cuda.engine import BatchEngine, Request, drain
    from lure_cuda.prompt import build_prefill
    from lure_cuda.sampling import SamplingParams, make_generator
    m = tiny_turbo_model
    _, mine = _turbo_heads_for(m)

    def req(case, seed):
        text, rtext, ins = case
        codes, spk = make_voice_tensors(n_ref=5, hidden=64, seed=len(text))
        emb, _tr, pad = build_prefill(m, make_prompt(text, rtext, ins), codes, spk, "english")
        return Request(prefill=emb, tts_pad=pad, sampling=SamplingParams(), max_new_tokens=12, generator=make_generator(seed, "cpu"))

    def run(reqs):
        eng = BatchEngine(m.talker, m.config.talker_config, max_batch=3, max_seq_len=256, device="cpu", dtype=torch.float64,
                          use_graphs=False, turbo_head=mine)
        for r in reqs:
            eng.submit(r)
        eng.run_until_idle()
        return [list(drain(r, timeout=0)) for r in reqs]
    solo = run([req(TEXTS[0], 11)])[0]
    batch = run([req(TEXTS[1], 5), req(TEXTS[0], 11), req(TEXTS[1], 6)])[1]
    assert solo == batch


# ------------------------------------------------------------------------------------------------- selection
def test_resolve_turbo(tmp_path):
    from lure_cuda.turbo import resolve_turbo
    d = tmp_path / "m"
    d.mkdir()
    assert resolve_turbo(str(d)) is None
    with pytest.raises(FileNotFoundError):
        resolve_turbo(str(d), turbo=1)
    (d / "turbo_head.safetensors").write_bytes(b"x")
    (d / "turbo.json").write_text(json.dumps({"format": "lure-turbo-v1", "groups": GROUPS, "head_file": "turbo_head.safetensors"}))
    c = resolve_turbo(str(d))
    assert c is not None and tuple(map(tuple, c.groups)) == tuple(GROUPS) and c.head_path.endswith("turbo_head.safetensors")
    assert resolve_turbo(str(d), turbo="0") is None
    assert resolve_turbo(str(d), turbo="auto", turbo_head=str(d / "turbo.json")).head_path.endswith("turbo.json")
