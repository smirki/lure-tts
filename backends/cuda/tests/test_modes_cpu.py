"""Every checkpoint variant, mode and layout on tiny random Qwen3-TTS models (CPU, float64, eager steps).

1. Greedy parity: engine codes == qwen-tts 0.1.1 `generate()` codes (talker + sub-talker greedy, repetition penalty
   1.05, suppress tokens, min_new_tokens) for icl / xvec (0.6B and 1.7B Base), custom_voice (0.6B and 1.7B
   CustomVoice, dialect speakers) and voice_design (1.7B VoiceDesign), each in the non_streaming and streaming layouts.
   The 1.7B-shaped models have talker hidden != code predictor hidden, so small_to_mtp_projection is exercised.
2. Mixed batches: prompts of different modes and layouts share one batch of 8 and each still equals its qwen-tts run;
   seeded sampled requests give the same frames alone and mixed, with slot reuse, late joins and KV rebases.
3. KV-length buckets: a request crossing the 256 bucket boundary keeps parity.
4. Backend: core GenerateRequest per mode, core-row length check, decode with the reference only in icl, mode / voice
   validation, spec and turbo.json checks.
"""
import json
import types

import numpy as np
import pytest
import torch

from conftest import UP, FakeSpeechTokenizer, _shrink, make_prompt, CONFIGS

LONG = [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010, 11011, 12012]
SHORT = [42, 43]
REF = [11, 12]
INS = [301, 302, 303]

# (variant, mode, layout, text ids, instruct ids, speaker, language)
CASES = [
    ("0.6B-Base", "icl", "non_streaming", LONG[:7], INS, None, "english"),
    ("0.6B-Base", "icl", "streaming", LONG, None, None, "english"),  # text longer than the codec column: trailing text
    ("0.6B-Base", "icl", "streaming", SHORT, None, None, "auto"),  # text fits: trailing is tts_pad
    ("0.6B-Base", "xvec", "non_streaming", LONG[:5], None, None, "english"),
    ("0.6B-Base", "xvec", "streaming", LONG, INS, None, "english"),
    ("1.7B-Base", "icl", "non_streaming", LONG[:6], None, None, "english"),
    ("1.7B-Base", "icl", "streaming", LONG, INS, None, "english"),
    ("1.7B-Base", "xvec", "non_streaming", LONG[:4], INS, None, "auto"),
    ("1.7B-Base", "xvec", "streaming", LONG, None, None, "english"),
    ("0.6B-CustomVoice", "custom_voice", "non_streaming", LONG[:6], None, "serena", "english"),
    ("0.6B-CustomVoice", "custom_voice", "streaming", LONG, None, "serena", "english"),
    ("1.7B-CustomVoice", "custom_voice", "non_streaming", LONG[:5], INS, "ryan", "english"),
    ("1.7B-CustomVoice", "custom_voice", "streaming", LONG, None, "eric", "chinese"),  # dialect id
    ("1.7B-CustomVoice", "custom_voice", "streaming", [7], INS, "dylan", "auto"),  # one text token: trailing [eos]
    ("1.7B-VoiceDesign", "voice_design", "non_streaming", LONG[:6], INS, None, "english"),
    ("1.7B-VoiceDesign", "voice_design", "streaming", LONG, INS, None, "auto"),
]
GREEDY = dict(do_sample=False, subtalker_do_sample=False)


def _case_id(c):
    return f"{c[0]}-{c[1]}-{c[2]}-{len(c[3])}"


def _voice(m, seed, n_ref=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2048, (n_ref, 16), generator=g), torch.randn(m.config.talker_config.hidden_size,
                                                                       generator=g, dtype=torch.float64)


def _setup(m, case, seed=0):
    _v, mode, layout, text, ins, speaker, lang = case
    codes, spk = _voice(m, seed + len(text))
    p = make_prompt(text, REF if mode == "icl" else None, ins)
    return p, codes, spk


def _request(m, case, sampling, max_new, seed=0):
    from lure_cuda.engine import Request
    from lure_cuda.prompt import build_prefill
    from lure_cuda.sampling import SamplingParams, make_generator

    _v, mode, layout, text, ins, speaker, lang = case
    p, codes, spk = _setup(m, case)
    emb, trailing, pad = build_prefill(m, p, codes if mode == "icl" else None,
                                       spk if mode in ("icl", "xvec") else None, lang, mode=mode, layout=layout,
                                       speaker=speaker)
    sp = SamplingParams(**sampling) if isinstance(sampling, dict) else sampling
    return Request(prefill=emb, tts_pad=pad, trailing=trailing, sampling=sp, max_new_tokens=max_new,
                   generator=make_generator(seed, "cpu"))


def _qwen_codes(m, case, max_new):
    """qwen-tts 0.1.1 generate() for the same prompt, the way its wrappers call it."""
    _v, mode, layout, text, ins, speaker, lang = case
    p, codes, spk = _setup(m, case)
    kw = dict(input_ids=[torch.tensor([p.input_ids])],
              instruct_ids=[torch.tensor([p.instruct_ids]) if p.instruct_ids else None], languages=[lang],
              non_streaming_mode=layout == "non_streaming", max_new_tokens=max_new, do_sample=False,
              subtalker_dosample=False, repetition_penalty=1.05)
    if mode == "icl":
        kw.update(ref_ids=[torch.tensor([p.ref_ids])], voice_clone_prompt={
            "ref_code": [codes], "ref_spk_embedding": [spk], "x_vector_only_mode": [False], "icl_mode": [True]})
    elif mode == "xvec":
        kw.update(ref_ids=[None], voice_clone_prompt={
            "ref_code": [None], "ref_spk_embedding": [spk], "x_vector_only_mode": [True], "icl_mode": [False]})
    elif mode == "custom_voice":
        kw.update(speakers=[speaker])
    with torch.no_grad():
        out, _ = m.generate(**kw)
    return out[0].tolist()


def _engine(m, B=4, max_seq_len=256):
    from lure_cuda.engine import BatchEngine

    return BatchEngine(m.talker, m.config.talker_config, max_batch=B, max_seq_len=max_seq_len, device="cpu",
                       dtype=torch.float64, use_graphs=False)


def _run(m, reqs, B=4, max_seq_len=256):
    from lure_cuda.engine import drain

    e = _engine(m, B, max_seq_len)
    for r in reqs:
        e.submit(r)
    e.run_until_idle()
    return [list(drain(r, timeout=0)) for r in reqs], e


def _assert_parity(got, want, r):
    assert len(got) > 0
    assert got == want[:len(got)]
    if r.ended_by_eos:
        assert len(got) == len(want)


# --------------------------------------------------------------------------------------------- 1. greedy parity
def test_tiny_shapes(tiny_variant):
    for name in ("0.6B-Base", "0.6B-CustomVoice"):
        cp = tiny_variant(name).talker.code_predictor
        assert isinstance(cp.small_to_mtp_projection, torch.nn.Identity)
    for name in ("1.7B-Base", "1.7B-CustomVoice", "1.7B-VoiceDesign"):
        m = tiny_variant(name)
        cp = m.talker.code_predictor
        assert isinstance(cp.small_to_mtp_projection, torch.nn.Linear)
        assert (cp.small_to_mtp_projection.in_features, cp.small_to_mtp_projection.out_features) == (64, 32)
        assert cp.model.codec_embedding[0].weight.shape[1] == 64  # CP_EMB lives in the talker space
    assert tiny_variant("1.7B-Base").speaker_encoder is not None
    assert tiny_variant("1.7B-CustomVoice").speaker_encoder is None
    assert tiny_variant("1.7B-VoiceDesign").speaker_encoder is None


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_greedy_parity_every_mode(tiny_variant, case):
    m = tiny_variant(case[0])
    max_new = 24
    r = _request(m, case, GREEDY, max_new)
    (got,), _ = _run(m, [r])
    want = _qwen_codes(m, case, max_new)
    _assert_parity(got, want, r)
    n_trailing = r.trailing.shape[1]
    if case[2] == "streaming" and n_trailing > 1:
        assert len(got) > n_trailing, "the run must outlast the trailing text to cover the switch to tts_pad"


# --------------------------------------------------------------------------------------------- 2. mixed batches
BASE_MIX = [c for c in CASES if c[0] == "1.7B-Base"] + [
    ("1.7B-Base", "icl", "non_streaming", LONG, INS, None, "auto"),
    ("1.7B-Base", "icl", "streaming", SHORT, None, None, "english"),
    ("1.7B-Base", "xvec", "non_streaming", LONG, None, None, "english"),
    ("1.7B-Base", "xvec", "streaming", [5, 6, 7], INS, None, "english"),
]


def test_mixed_modes_batch_of_8_greedy(tiny_variant):
    """icl and xvec, both layouts, with and without instruct, 8 at once: each equals its own qwen-tts run."""
    m = tiny_variant("1.7B-Base")
    assert len(BASE_MIX) == 8
    max_new = 20
    reqs = [_request(m, c, GREEDY, max_new) for c in BASE_MIX]
    got, e = _run(m, reqs, B=8, max_seq_len=512)
    for c, g, r in zip(BASE_MIX, got, reqs):
        _assert_parity(g, _qwen_codes(m, c, max_new), r)


def test_mixed_custom_voice_batch_greedy(tiny_variant):
    m = tiny_variant("1.7B-CustomVoice")
    cases = [c for c in CASES if c[0] == "1.7B-CustomVoice"]
    max_new = 20
    reqs = [_request(m, c, GREEDY, max_new) for c in cases]
    got, _ = _run(m, reqs, B=len(cases))
    for c, g, r in zip(cases, got, reqs):
        _assert_parity(g, _qwen_codes(m, c, max_new), r)


def test_mixed_modes_sampled_invariance_rebase_and_late_join(tiny_variant):
    from lure_cuda.sampling import SamplingParams

    m = tiny_variant("1.7B-Base")
    sp = SamplingParams(temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.05)
    max_new = 16
    alone = [_run(m, [_request(m, c, sp, max_new, seed=50 + i)], B=1)[0][0] for i, c in enumerate(BASE_MIX)]
    # batch of 3 for 8 requests (slot reuse) and a cache just big enough for the longest (forces rebases)
    longest = max(_request(m, c, sp, 1).prefill.shape[1] for c in BASE_MIX)
    got, e = _run(m, [_request(m, c, sp, max_new, seed=50 + i) for i, c in enumerate(BASE_MIX)], B=3,
                  max_seq_len=longest + max_new + 6)
    assert e.rebases > 0
    assert got == alone
    # late join: a streaming request admitted mid-run of a non_streaming one
    from lure_cuda.engine import drain

    e = _engine(m, B=4)
    a = _request(m, BASE_MIX[0], sp, max_new, seed=50)
    b = _request(m, BASE_MIX[3], sp, max_new, seed=53)
    e.submit(a)
    for _ in range(5):
        e._admit_pending()
        e.step()
    e.submit(b)
    e.run_until_idle()
    assert list(drain(a, timeout=0)) == alone[0]
    assert list(drain(b, timeout=0)) == alone[3]


# --------------------------------------------------------------------------------------------- 3. KV buckets
def test_parity_across_kv_bucket_boundary(tiny_variant):
    m = tiny_variant("1.7B-CustomVoice")
    case = ("1.7B-CustomVoice", "custom_voice", "non_streaming", list(range(1000, 1240)), None, "vivian", "english")
    max_new = 24
    r = _request(m, case, GREEDY, max_new)
    L = r.prefill.shape[1]
    assert L < 256 < L + 10
    (got,), e = _run(m, [r], B=2, max_seq_len=320)
    assert e.tgraph.buckets == [256, 320]
    assert L + len(got) > 256, "the run must cross into the 320 bucket"
    _assert_parity(got, _qwen_codes(m, case, max_new), r)


# --------------------------------------------------------------------------------------------- 4. backend
def _spec(tmp_path, name):
    """The core's ModelSpec for a tiny variant: its shrunk config.json written to a folder and inspected."""
    model = pytest.importorskip("lure_tts.model")
    with open(f"{CONFIGS}/Qwen3-TTS-12Hz-{name}.json") as f:
        cfg = _shrink(json.load(f))
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    return model.inspect(str(d))


def _backend(m, spec, B=4):
    from lure_cuda.backend import CudaBackend
    from lure_cuda.decode import CodecDecoder

    e = _engine(m, B, 512)
    e.start()
    return CudaBackend(types.SimpleNamespace(model=m, processor=None), e, CodecDecoder(FakeSpeechTokenizer(), "cpu"),
                       "tiny", "cpu", torch.float64, {"decode_context_frames": 25}, spec=spec)


def _core_req(case, voice, spec, stream, seed=3, max_new=12):
    base = pytest.importorskip("lure_tts.backends.base")
    from lure_tts.defaults import Sampling
    from lure_tts.prompt import PromptIds

    _v, mode, layout, text, ins, speaker, lang = case
    p = make_prompt(text, REF if mode == "icl" else None, ins)
    return base.GenerateRequest(
        text="x " * len(text), voice=voice if mode in ("icl", "xvec") else None,
        prompt=PromptIds(p.input_ids, p.ref_ids, p.instruct_ids), instruct="calm" if ins else None,
        sampling=Sampling(), seed=seed, max_new_tokens=max_new, stream=stream, chunk_frames=4, first_chunk_frames=2,
        language=lang, mode=mode, layout=layout, speaker=speaker,
        speaker_id=spec.ids.speaker_id(speaker) if speaker else None)


@pytest.mark.parametrize("name", ["0.6B-Base", "1.7B-Base", "1.7B-CustomVoice", "1.7B-VoiceDesign"])
def test_backend_core_requests_every_mode(tiny_variant, tmp_path, name, monkeypatch):
    base = pytest.importorskip("lure_tts.backends.base")
    m = tiny_variant(name)
    spec = _spec(tmp_path, name)
    be = _backend(m, spec)
    calls = []
    orig = be.codec.decode_full
    monkeypatch.setattr(be.codec, "decode_full", lambda ref, gen: calls.append((ref, gen)) or orig(ref, gen))
    try:
        codes, spk = _voice(m, 1, n_ref=5)
        voice = base.Voice("v", "a reference", codes.numpy().astype(np.int32), spk.numpy().astype(np.float32))
        for case in [c for c in CASES if c[0] == name]:
            if case[1] == "xvec":
                voice = base.Voice("x", "", voice.ref_codes, voice.spk_emb)
            elif case[1] == "icl":
                voice = base.Voice("v", "a reference", voice.ref_codes, voice.spk_emb)
            stream = list(be.generate(_core_req(case, voice, spec, True)))
            info = stream[-1].info
            assert info.extra["mode"] == case[1] and info.extra["layout"] == case[2]
            assert info.samples == info.frames * UP == sum(c.pcm.size for c in stream)
            calls.clear()
            off = list(be.generate(_core_req(case, voice, spec, False)))
            assert off[0].info.frames == info.frames
            ref, gen = calls[-1]
            if case[1] == "icl":  # decoded with the reference, then its share cut
                assert ref is not None and ref.shape[0] == 5
            else:  # the generated codes alone
                assert ref is None
                np.testing.assert_allclose(off[0].pcm, FakeSpeechTokenizer().decode([{"audio_codes": gen}])[0][0],
                                           atol=1e-6)
            # streamed windows use the same left context as the offline decode
            np.testing.assert_allclose(np.concatenate([c.pcm for c in stream]), off[0].pcm, atol=1e-6)
    finally:
        be.close()


def test_backend_rejects_wrong_mode_and_voice(tiny_variant, tmp_path):
    base = pytest.importorskip("lure_tts.backends.base")
    m = tiny_variant("1.7B-CustomVoice")
    be = _backend(m, _spec(tmp_path, "1.7B-CustomVoice"), B=1)
    try:
        with pytest.raises(ValueError, match="speaker encoder"):
            be.encode_voice(np.zeros(2400, np.float32), "x")
        codes, spk = _voice(m, 1)
        voice = base.Voice("v", "ref", codes.numpy().astype(np.int32), spk.numpy().astype(np.float32))
        with pytest.raises(ValueError, match="not available"):
            be.generate_segment("hi", voice, prompt=make_prompt([5, 6], REF), mode="icl")
        with pytest.raises(ValueError, match="unknown speaker"):
            be.generate_segment("hi", prompt=make_prompt([5, 6]), speaker="nobody")
    finally:
        be.close()
    m = tiny_variant("1.7B-Base")
    be = _backend(m, _spec(tmp_path, "1.7B-Base"), B=1)
    try:
        codes, _ = _voice(m, 1)
        small = base.Voice("v", "ref", codes.numpy().astype(np.int32), np.zeros(32, np.float32))  # a 0.6B-sized x-vector
        with pytest.raises(ValueError, match="x-vector has 32 dims"):
            be.generate_segment("hi", small, prompt=make_prompt([5, 6], REF))
        with pytest.raises(ValueError, match="reference transcript"):
            be.generate_segment("hi", base.Voice("x", "", small.ref_codes, small.spk_emb), mode="icl",
                                prompt=make_prompt([5, 6], REF))
    finally:
        be.close()


def test_backend_dump_layout_matches_core(tiny_variant, tmp_path):
    """dump_layout runs this backend's builder on a symbolic stub of the loaded config; it must equal the core rows."""
    from lure_tts.prompt import build_prompt, build_prompt_ids
    from lure_tts.tokenizer import QwenTokenizer

    from conftest import MODEL_DIR

    m = tiny_variant("1.7B-CustomVoice")
    spec = _spec(tmp_path, "1.7B-CustomVoice")
    be = _backend(m, spec, B=1)
    be.model_dir = MODEL_DIR  # tokenizer files
    try:
        tok = QwenTokenizer.from_dir(MODEL_DIR)
        for layout in ("non_streaming", "streaming"):
            got = be.dump_layout("Hello there, friend.", None, "Speak softly.", 0, "auto", mode="custom_voice",
                                 layout=layout, speaker="dylan")
            want = build_prompt(build_prompt_ids(tok, "Hello there, friend.", None, "Speak softly."), spec.ids,
                                mode="custom_voice", layout=layout, language="auto", speaker="dylan")
            assert got == {"layout": want.prefill, "trailing": want.trailing}
    finally:
        be.close()


def test_check_model_and_turbo(tiny_variant, tmp_path):
    import dataclasses

    from lure_cuda.backend import check_model, check_turbo

    m = tiny_variant("1.7B-Base")
    spec = _spec(tmp_path, "1.7B-Base")
    assert check_model(m, spec) == "base"
    with pytest.raises(ValueError, match="talker_hidden_size"):
        check_model(m, dataclasses.replace(spec, talker_hidden_size=1024))
    with pytest.raises(ValueError, match="variant"):
        check_model(m, dataclasses.replace(spec, variant="custom_voice"))
    assert check_model(tiny_variant("1.7B-VoiceDesign")) == "voice_design"
    # Turbo: 0.6B shape only, turbo.json hidden_size must equal the talker hidden
    t06 = tiny_variant("0.6B-Base").config.talker_config
    check_turbo(types.SimpleNamespace(hidden_size=32), t06)
    check_turbo(types.SimpleNamespace(hidden_size=None), t06)
    with pytest.raises(ValueError, match="hidden_size 1024"):
        check_turbo(types.SimpleNamespace(hidden_size=1024), t06)
    with pytest.raises(ValueError, match="0.6B checkpoints only"):
        check_turbo(types.SimpleNamespace(hidden_size=64), m.config.talker_config)
