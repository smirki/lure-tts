"""MLX backend unit tests on a tiny random talker (no checkpoint, about 200 MB RSS). Run: python -m pytest backends/mlx/tests"""
import os
import types

import mlx.core as mx
import numpy as np
import pytest

from lure_tts.backends.base import GenerateRequest, Voice
from lure_tts.defaults import Sampling
from lure_tts.prompt import ModelIds, build_layout, build_prompt_ids
from lure_tts.tokenizer import QwenTokenizer
from lure_mlx.backend import MLXBackend
from lure_mlx.cp import CompiledCodePredictor, EagerCodePredictor
from lure_mlx.embed import embed_rows, ref_block
from lure_mlx.sampling import repetition_penalty, sample, top_k_filter, top_p_filter
from mlx_audio.tts.models.qwen3_tts import qwen3_tts as q3
from mlx_audio.tts.models.qwen3_tts.config import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig
from mlx_audio.tts.models.qwen3_tts.talker import Qwen3TTSTalkerForConditionalGeneration

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TOKDIR = os.path.join(ROOT, "conformance", "reference", "tokenizer")
REF_TEXT = "Yes, the weekend order can stay exactly as it is."


def tiny_talker(seed=0):
    mx.random.seed(seed)
    cpc = Qwen3TTSTalkerCodePredictorConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                            num_key_value_heads=2, head_dim=16)
    cfg = Qwen3TTSTalkerConfig(code_predictor_config=cpc, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=16, text_hidden_size=32,
                               rope_scaling={"interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"},
                               codec_language_id={"english": 2050})
    t = Qwen3TTSTalkerForConditionalGeneration(cfg)
    # random init is small; scale embeddings so logits are not flat
    t.model.codec_embedding.weight = mx.random.normal(t.model.codec_embedding.weight.shape) * 0.5
    mx.eval(t.parameters())
    return t, cfg


@pytest.fixture(scope="module")
def talker():
    return tiny_talker()


@pytest.fixture(scope="module")
def tok():
    return QwenTokenizer.from_dir(TOKDIR)


# ---------------------------------------------------------------------------------------------------- sampling
def test_top_k_and_top_p():
    x = mx.array([[0.1, 2.0, 1.0, -1.0, 3.0]])
    k = np.array(top_k_filter(x, 2))
    assert np.isfinite(k).sum() == 2 and np.isfinite(k[0, [1, 4]]).all()
    p = np.array(top_p_filter(x, 0.6))
    probs = (np.exp(np.array(x)) / np.exp(np.array(x)).sum())[0]
    keep = np.isfinite(p[0])
    assert keep[4] and probs[keep].sum() >= 0.6 and keep.sum() <= 2
    assert np.isfinite(np.array(top_p_filter(x, 1.0))).all()


def test_repetition_penalty_and_greedy():
    x = mx.array([[2.0, -2.0, 1.0]])
    seen = mx.array([[True, True, False]])
    np.testing.assert_allclose(np.array(repetition_penalty(x, seen, 2.0)), [[1.0, -4.0, 1.0]])
    assert int(sample(x, mx.zeros((1, 3)), 0.9, 50, 1.0, do_sample=False)[0]) == 0


def test_gumbel_sampling_distribution():
    mx.random.seed(3)
    logits = mx.log(mx.array([[0.7, 0.2, 0.1]]))
    draws = [int(sample(logits, mx.random.gumbel((1, 3)), 1.0, 0, 1.0)[0]) for _ in range(3000)]
    freq = np.bincount(draws, minlength=3) / 3000
    np.testing.assert_allclose(freq, [0.7, 0.2, 0.1], atol=0.04)


# ---------------------------------------------------------------------------------------------------- code predictor
@pytest.mark.parametrize("do_sample", [False, True])
def test_compiled_cp_matches_eager(talker, do_sample):
    t, _ = talker
    eager = EagerCodePredictor(t, 0.9, 50, 1.0, do_sample)
    comp = CompiledCodePredictor(t, 0.9, 50, 1.0, do_sample)
    for i in range(3):
        mx.random.seed(10 + i)
        h = mx.random.normal((1, 1, 64))
        cb0 = mx.array([17 * (i + 1)], dtype=mx.int32)
        noise = mx.random.gumbel((15, 2048))
        a_codes, a_emb = eager(h, cb0, noise)
        b_codes, b_emb = comp(h, cb0, noise)
        assert np.array(a_codes).tolist() == np.array(b_codes).tolist()
        np.testing.assert_allclose(np.array(a_emb), np.array(b_emb), atol=1e-4)


def test_eager_cp_matches_mlx_audio_greedy(talker):
    """Our eager CP (float32 logits) vs mlx-audio's own _predict_code_tokens with greedy sampling."""
    t, cfg = talker
    stub = types.SimpleNamespace(talker=t, config=types.SimpleNamespace(talker_config=cfg))
    stub._reset_code_cache = types.MethodType(q3.Model._reset_code_cache, stub)
    stub._sample_token_batch = types.MethodType(q3.Model._sample_token_batch, stub)
    h = mx.random.normal((1, 1, 64))
    cb0 = mx.array([[5]], dtype=mx.int32)
    toks, _ = q3.Model._predict_code_tokens(stub, cb0, h, temperature=0.0, top_k=50, top_p=1.0)
    ref = [int(x[0, 0]) for x in toks[1:]]
    ours, _ = EagerCodePredictor(t, 0.9, 50, 1.0, do_sample=False)(h, cb0[0], mx.zeros((15, 2048)))
    assert np.array(ours).tolist() == ref


# ---------------------------------------------------------------------------------------------------- prefill
def ids():
    return ModelIds.from_config(TOKDIR)


@pytest.mark.parametrize("direction", [None, "Tired and resigned, then a reluctant smile."])
def test_embed_rows_matches_mlx_audio_icl_prefill(talker, tok, direction):
    """mlx-audio's own ICL prefill (plus the instruct prefix the model expects) == embed_rows(core layout)."""
    t, cfg = talker
    T = 7
    codes = (np.arange(T * 16).reshape(T, 16) * 37 % 2048).astype(np.int32)
    spk = mx.random.normal((1, 64))
    m = ids()
    stub = types.SimpleNamespace(
        tokenizer=tok, talker=t, _icl_cache={}, speaker_encoder=object(),
        config=types.SimpleNamespace(tts_bos_token_id=m.tts_bos, tts_eos_token_id=m.tts_eos, tts_pad_token_id=m.tts_pad,
                                     talker_config=cfg),
        speech_tokenizer=types.SimpleNamespace(encode=lambda a: mx.array(codes.T[None])),
        extract_speaker_embedding=lambda a: spk,
    )
    text = "(sigh) I guess we're doing this again. (chuckle) Fine. I'm in."
    emb, trailing, pad, _ = q3.Model._prepare_icl_generation_inputs(stub, text, mx.zeros(4800), REF_TEXT, language="english")
    if direction:
        iid = mx.array(tok.encode(f"<|im_start|>user\n{direction}<|im_end|>\n"))[None]
        emb = mx.concatenate([t.text_projection(t.get_text_embeddings()(iid)), emb], axis=1)
    p = build_prompt_ids(tok, text, REF_TEXT, direction)
    rows = build_layout(p, m, T, "english")
    ours = embed_rows(t, rows, spk[0], ref_block(t, codes, mx.float32), mx.float32)
    assert ours.shape == emb.shape
    np.testing.assert_allclose(np.array(ours), np.array(emb), atol=2e-5)


# ---------------------------------------------------------------------------------------------------- generate loop
class StubDecoder:
    def __init__(self):
        self.calls = []

    def reset_streaming_state(self):
        self.calls.append("reset")

    def streaming_step(self, codes):
        self.calls.append(("step", codes.shape[-1]))
        return mx.zeros((1, 1, codes.shape[-1] * 1920))


def make_backend(talker, **opts):
    t, _ = talker
    dec = StubDecoder()
    st = types.SimpleNamespace(decoder=dec, decode_upsample_rate=1920,
                               decode=lambda c: (mx.zeros((1, c.shape[1] * 1920)), mx.array([c.shape[1] * 1920])))
    model = types.SimpleNamespace(talker=t, speech_tokenizer=st)
    be = MLXBackend(model, TOKDIR, dict(opts, dtype="float32"))
    return be, dec


def voice():
    T = 5
    return Voice("v", REF_TEXT, (np.arange(T * 16).reshape(T, 16) % 2048).astype(np.int32), np.ones(64, np.float32) * 0.01,
                 {"wav_sha256": "x"})


def run(be, tok, seed=7, max_new=10, stream=True, chunk=3, sampling=None):
    v = voice()
    text = "Hello there. (laugh) Okay."
    req = GenerateRequest(text=text, voice=v, prompt=build_prompt_ids(tok, text, v.ref_text, "Warm."), instruct="Warm.",
                          sampling=sampling or Sampling(), seed=seed, max_new_tokens=max_new, stream=stream, chunk_frames=chunk,
                          first_chunk_frames=2)
    return list(be.generate(req))


def codes_of(be, tok, **kw):
    """Capture generated codes by wrapping the decoder."""
    got = []
    dec = be.model.speech_tokenizer.decoder
    orig = dec.streaming_step

    def cap(codes):
        got.append(np.array(codes))
        return orig(codes)
    dec.streaming_step = cap
    out = run(be, tok, **kw)
    dec.streaming_step = orig
    return [g for g in got[1:]], out  # got[0] = decoder priming with ref codes


@pytest.mark.parametrize("opts", [{}, {"pipeline": True}, {"compile_cp": True}, {"compile_cp": True, "pipeline": True}])
def test_generate_paths_identical(talker, tok, opts):
    base_be, _ = make_backend(talker)
    ref_codes, ref_out = codes_of(base_be, tok)
    be, dec = make_backend(talker, **opts)
    got_codes, out = codes_of(be, tok)
    assert [c.tolist() for c in got_codes] == [c.tolist() for c in ref_codes]
    info = out[-1].info
    assert info is not None and info.frames == 10 and not info.ended_by_eos
    sizes = [len(c.pcm) for c in out[:-1]]
    assert sizes[0] == 2 * 1920 and all(s == 3 * 1920 for s in sizes[1:-1])
    assert sum(sizes) == 10 * 1920


def test_eos_and_min_new_tokens(talker, tok):
    """Leave only code 0 (far below) and EOS open: EOS is masked for exactly min_new_tokens frames, then wins."""
    be, _ = make_backend(talker)
    sup = np.full((1, 3072), -np.inf, np.float32)
    sup[0, 0] = -1e4
    sup[0, 2150] = 0.0
    be.suppress = mx.array(sup)
    out = run(be, tok, max_new=20)
    info = out[-1].info
    assert info.ended_by_eos and info.frames == 2


def test_non_stream_decodes_with_ref_and_cuts(talker, tok):
    be, _ = make_backend(talker)
    out = run(be, tok, stream=False, max_new=6)
    assert len(out) == 1 and out[0].info.frames == 6
    assert len(out[0].pcm) == 6 * 1920  # (5 ref + 6 gen) frames decoded, ref share cut


def test_seed_determinism(talker, tok):
    be, _ = make_backend(talker)
    a, _ = codes_of(be, tok, seed=1)
    b, _ = codes_of(be, tok, seed=1)
    c, _ = codes_of(be, tok, seed=2)
    assert [x.tolist() for x in a] == [x.tolist() for x in b]
    assert [x.tolist() for x in a] != [x.tolist() for x in c]


def test_dump_layout_matches_reference(talker):
    import json
    be, _ = make_backend(talker)
    ref = json.load(open(os.path.join(ROOT, "conformance", "reference", "prompt_ids.json")))
    for e in ref["entries"]:
        got = be.dump_layout(e["text"], e["ref_text"], e.get("direction"), e["n_ref_frames"], e.get("language", "english"))
        assert [list(r) for r in got["layout"]] == [list(r) for r in e["layout"]], e["id"]
        assert got["trailing"] == e["trailing_text_hidden"], e["id"]
