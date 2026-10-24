"""Every variant, size shape, mode and layout on tiny random checkpoints (tiny.py), loaded through MLXBackend.load.

  - load: mlx-audio builds the right parts for each config (1.7B small_to_mtp_projection, speaker encoder on Base only)
  - prompt: dump_layout equals every conformance/reference/prompt_modes.json entry (prefill and trailing rows)
  - embedding: the prefill embedding equals T(text) + C(codec) | spk | ref computed in numpy from the weights
  - greedy parity: codes, prefill and trailing rows equal qwen-tts 0.1.1 on the same weights (fixtures/qwen_greedy.json,
    regenerate with make_qwen_fixture.py)
  - decode: only icl primes the streaming decoder and decodes with the reference and the proportional cut

Runs on the MLX CPU device (full float32, see test_turbo_tiny.py). Run: python -m pytest backends/mlx/tests
"""
import json
import os

import mlx.core as mx
import numpy as np
import pytest

import tiny
from lure_mlx.backend import MLXBackend, check_model
from lure_mlx.embed import embed_rows, ref_block
from lure_tts.backends.base import GenerateRequest, Voice
from lure_tts.defaults import Sampling
from lure_tts.model import inspect
from lure_tts.prompt import PromptIds

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = json.load(open(os.path.join(HERE, "fixtures", "qwen_greedy.json")))
MODE_ENTRIES = json.load(open(os.path.join(tiny.REF, "prompt_modes.json")))["entries"]
CASES = tiny.greedy_cases()
GREEDY = Sampling(do_sample=False, subtalker_do_sample=False)


@pytest.fixture(scope="module", autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


class StubDecoder:
    def __init__(self):
        self.calls = []

    def reset_streaming_state(self):
        self.calls.append(("reset",))

    def streaming_step(self, codes):
        self.calls.append(("step", np.array(codes)))
        return mx.zeros((1, 1, codes.shape[-1] * 1920))


class StubCodec:
    """Stands in for the speech tokenizer: records what is decoded and returns silence of the right length."""
    decode_upsample_rate = 1920

    def __init__(self):
        self.decoder = StubDecoder()
        self.decoded = []

    def decode(self, codes):
        self.decoded.append(np.array(codes[0]))
        return mx.zeros((1, codes.shape[1] * 1920)), None

    def encode(self, audio):
        n = max(1, audio.shape[-1] // 1920)
        return mx.array(tiny.ref_codes(n).T[None])  # [1, 16, T]


class Checkpoints:
    def __init__(self, root):
        self.root, self.backends, self.weights = root, {}, {}

    def dir(self, name):
        return tiny.make_model_dir(self.root, name)

    def backend(self, name) -> MLXBackend:
        if name not in self.backends:
            d = self.dir(name)
            be = MLXBackend.load(d, spec=inspect(d), dtype="float32", cache_limit_mb=64)
            be.model.speech_tokenizer = StubCodec()
            self.backends[name] = be
        be = self.backends[name]
        be.model.speech_tokenizer = StubCodec()
        return be

    def weight(self, name) -> dict:
        if name not in self.weights:
            w = mx.load(os.path.join(self.dir(name), "model.safetensors"))
            self.weights[name] = {k: np.array(v) for k, v in w.items() if k.startswith("talker.")}
        return self.weights[name]


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    return Checkpoints(str(tmp_path_factory.mktemp("tiny_ckpt")))


def voice_for(case):
    """icl and xvec voices both carry reference codes; xvec must ignore them."""
    if case["mode"] not in ("icl", "xvec"):
        return None
    return Voice(f"v-{case['id']}", case["ref_text"], tiny.ref_codes(6), np.array(case["x_vector"], np.float32),
                 {"wav_sha256": case["id"]})


def request(be, case, **kw):
    speaker = case["speaker"]
    args = dict(text="tiny", voice=voice_for(case), prompt=PromptIds(case["input_ids"], case["ref_ids"], case["instruct_ids"]),
                sampling=GREEDY, seed=0, max_new_tokens=case["max_new_tokens"], stream=False, language=case["language"],
                mode=case["mode"], layout=case["layout"], speaker=speaker)
    args.update(kw)
    if "speaker_id" not in kw:
        args["speaker_id"] = be.ids.speaker_id(speaker) if speaker else None
    return GenerateRequest(**args)


def generated_codes(be, req):
    out = list(be.generate(req))
    codec = be.model.speech_tokenizer
    if req.stream:
        steps = [c[1] for c in codec.decoder.calls if c[0] == "step"]
        if req.mode == "icl":
            steps = steps[1:]  # priming with the reference codes
        codes = np.concatenate([s[0].T for s in steps], axis=0) if steps else np.zeros((0, 16), np.int32)
    else:
        full = codec.decoded[-1]
        codes = full[req.voice.n_ref_frames:] if req.mode == "icl" else full
    return codes.astype(np.int64).tolist(), out


# ---------------------------------------------------------------------------------------------------- load
@pytest.mark.parametrize("name", list(tiny.VARIANTS))
def test_load_every_variant(ckpt, name):
    be = ckpt.backend(name)
    spec, t = be.spec, be.talker
    shape = tiny.SHAPES[tiny.VARIANTS[name][1]]
    assert spec.talker_hidden_size == t.config.hidden_size == shape["hidden"]
    assert spec.code_predictor_hidden_size == t.code_predictor.config.hidden_size == shape["cp_hidden"]
    proj = t.code_predictor.small_to_mtp_projection
    if shape["hidden"] != shape["cp_hidden"]:
        assert proj is not None and tuple(proj.weight.shape) == (shape["cp_hidden"], shape["hidden"]) and "bias" in proj
    else:
        assert proj is None
    assert all(tuple(e.weight.shape) == (2048, shape["hidden"]) for e in t.code_predictor.codec_embedding)
    if spec.variant == "base":
        assert be.model.speaker_encoder is not None and spec.speaker_encoder_dim == shape["hidden"]
        spk = be.model.extract_speaker_embedding(mx.zeros(24000))
        assert spk.shape == (1, shape["hidden"])
    else:
        assert be.model.speaker_encoder is None and spec.speaker_encoder_dim is None
    assert be.spec.modes == {"base": ("icl", "xvec"), "custom_voice": ("custom_voice",),
                             "voice_design": ("voice_design",)}[spec.variant]


def test_check_model_rejects_wrong_spec(ckpt):
    with pytest.raises(ValueError, match="hidden"):
        check_model(ckpt.backend("base06").model, inspect(ckpt.dir("base17")))
    with pytest.raises(ValueError, match="speaker encoder"):
        check_model(ckpt.backend("cv06").model, inspect(ckpt.dir("base06")))


def test_encode_voice_needs_base(ckpt):
    for name in ("cv06", "cv17", "vd17"):
        with pytest.raises(ValueError, match="speaker encoder"):
            ckpt.backend(name).encode_voice(np.zeros(24000, np.float32), "")


def test_turbo_head_is_0b6_only(ckpt):
    class Cfg:
        head_path, groups, addon_hidden, layernorm_eps = "unused", ((1, 3), (4, 7), (8, 15)), 512, 1e-5
    with pytest.raises(ValueError, match="0.6B"):
        ckpt.backend("base17")._use_turbo(Cfg())


# ---------------------------------------------------------------------------------------------------- prompt rows
@pytest.mark.parametrize("e", MODE_ENTRIES, ids=lambda e: e["id"])
def test_dump_layout_matches_prompt_modes(ckpt, e):
    be = ckpt.backend(tiny.CONFIG_VARIANT[e["config"]])
    got = be.dump_layout(e["text"], e["ref_text"], e["direction"], e["n_ref_frames"], e["language"], mode=e["mode"],
                         layout=e["prompt_layout"], speaker=e["speaker"])
    assert got["layout"] == e["layout"]
    assert got["trailing"] == e["trailing"]


def test_mode_must_match_variant(ckpt):
    be = ckpt.backend("base06")
    case = next(c for c in CASES if c["variant"] == "cv06")
    with pytest.raises(ValueError, match="runs modes"):
        list(be.generate(request(be, case, speaker_id=None)))
    be = ckpt.backend("cv17")
    case = next(c for c in CASES if c["id"] == "cv17.ns")
    with pytest.raises(ValueError, match="speaker_id"):
        list(be.generate(request(be, case, speaker_id=be.ids.speaker_id(case["speaker"]) + 1)))


def test_xvec_dim_must_match(ckpt):
    be = ckpt.backend("base17")
    case = next(c for c in CASES if c["id"] == "base17.icl.ns")
    v = voice_for(case)
    v.spk_emb = np.zeros(32, np.float32)  # a 0.6B-sized x-vector on the 1.7B shape
    with pytest.raises(ValueError, match="x-vector"):
        list(be.generate(request(be, case, voice=v)))


# ---------------------------------------------------------------------------------------------------- embeddings
def f64(w, key, idx=slice(None)):
    return np.asarray(w[key][idx], np.float64)


def np_text(w, ids):
    p = "talker.text_projection.linear_"
    h = f64(w, "talker.model.text_embedding.weight", ids) @ f64(w, p + "fc1.weight").T + f64(w, p + "fc1.bias")
    h = h / (1 + np.exp(-h))
    return h @ f64(w, p + "fc2.weight").T + f64(w, p + "fc2.bias")


def np_ref(w, codes):
    acc = f64(w, "talker.model.codec_embedding.weight", codes[:, 0])
    for g in range(1, 16):
        acc += f64(w, f"talker.code_predictor.model.codec_embedding.{g - 1}.weight", codes[:, g])
    return acc


def np_rows(w, rows, spk, codes):
    """Independent numpy embedding of the core rows: T(text) + C(codec) | spk | ref:t."""
    out = []
    for t, c in rows:
        v = np_text(w, [t])[0] if t is not None else 0.0
        if isinstance(c, int):
            v = v + f64(w, "talker.model.codec_embedding.weight", c)
        elif c == "spk":
            v = v + spk
        elif isinstance(c, str):
            v = v + np_ref(w, codes[int(c[4:])][None])[0]
        out.append(v)
    return np.stack(out)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_prefill_embedding_matches_core_rows(ckpt, case):
    be, w = ckpt.backend(case["variant"]), ckpt.weight(case["variant"])
    req = request(be, case)
    prompt = be.prompt(req)
    spk, refs = be._voice_tensors(req)
    got = np.array(embed_rows(be.talker, prompt.prefill, spk, refs, mx.float32)[0], np.float64)
    want = np_rows(w, prompt.prefill, np.array(case["x_vector"] or [0.0]), tiny.ref_codes(6))
    np.testing.assert_allclose(got, want, atol=2e-5, rtol=1e-5)
    if case["mode"] != "icl":
        assert refs is None and not any(isinstance(c, str) and c.startswith("ref:") for _, c in prompt.prefill)
    if case["mode"] in ("custom_voice", "voice_design"):
        assert spk is None and not any(c == "spk" for _, c in prompt.prefill)
    trailing = be._trailing(prompt)
    if case["layout"] == "non_streaming":
        assert prompt.trailing == [[be.ids.tts_pad, None]]
    if prompt.trailing == [[be.ids.tts_pad, None]]:  # also a streaming icl text that fits in the overlay
        assert trailing is None
    else:
        np.testing.assert_allclose(np.array(trailing[0], np.float64), np_text(w, [r[0] for r in prompt.trailing]), atol=2e-5)


def test_ref_block_in_talker_space(ckpt):
    be, w = ckpt.backend("base17"), ckpt.weight("base17")
    codes = tiny.ref_codes(6)
    got = np.array(ref_block(be.talker, codes, mx.float32), np.float64)
    assert got.shape == (6, 64)
    np.testing.assert_allclose(got, np_ref(w, codes), atol=2e-5)


# ---------------------------------------------------------------------------------------------------- qwen-tts parity
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_greedy_parity_with_qwen_tts(ckpt, case):
    ref = FIXTURE["results"][case["id"]]
    be = ckpt.backend(case["variant"])
    req = request(be, case)
    prompt = be.prompt(req)
    spk, refs = be._voice_tensors(req)
    proj = tiny.projection(be.talker.config.hidden_size).astype(np.float64)
    pre = np.array(embed_rows(be.talker, prompt.prefill, spk, refs, mx.float32)[0], np.float64) @ proj
    np.testing.assert_allclose(pre, ref["prefill_proj"], atol=2e-4, rtol=1e-5)
    trailing = be._trailing(prompt)
    tr = np.array(be.tts_pad_embed[0] if trailing is None else trailing[0], np.float64) @ proj
    np.testing.assert_allclose(tr, ref["trailing_proj"], atol=2e-4, rtol=1e-5)
    got, out = generated_codes(be, req)
    want = ref["codes"]
    assert len(want) > 0 and got[:len(want)] == want
    # qwen-tts returns max_new_tokens - 1 frames when it does not stop; a shorter result means it hit EOS
    if len(want) < case["max_new_tokens"] - 1:
        assert out[-1].info.ended_by_eos and len(got) == len(want)


@pytest.mark.parametrize("cid", ["base.icl.s", "base17.icl.s", "base.xvec.s", "cv17.s.instruct", "vd17.s"])
def test_streamed_codes_equal_whole(ckpt, cid):
    case = next(c for c in CASES if c["id"] == cid)
    be = ckpt.backend(case["variant"])
    whole, _ = generated_codes(be, request(be, case))
    be = ckpt.backend(case["variant"])
    streamed, out = generated_codes(be, request(be, case, stream=True, chunk_frames=3, first_chunk_frames=2))
    assert streamed == whole
    assert out[-1].info.extra["mode"] == case["mode"] and out[-1].info.extra["layout"] == case["layout"]


# ---------------------------------------------------------------------------------------------------- decode
@pytest.mark.parametrize("cid", ["base.icl.ns", "base17.icl.s", "base.xvec.ns", "cv06.ns", "cv17.s", "vd17.ns",
                                 "lure.c.icl", "lure.c.xvec"])
def test_decode_reference_only_in_icl(ckpt, cid):
    case = next(c for c in CASES if c["id"] == cid)
    icl = case["mode"] == "icl"
    be = ckpt.backend(case["variant"])
    out = list(be.generate(request(be, case, stream=False)))
    full = be.model.speech_tokenizer.decoded[-1]
    n_gen = out[-1].info.frames
    assert full.shape[0] == n_gen + (6 if icl else 0)
    if icl:
        assert full[:6].tolist() == tiny.ref_codes(6).tolist()
    assert len(out[-1].pcm) == n_gen * 1920  # the cut drops exactly the reference share
    be = ckpt.backend(case["variant"])
    out = list(be.generate(request(be, case, stream=True, chunk_frames=4, first_chunk_frames=4)))
    steps = [c for c in be.model.speech_tokenizer.decoder.calls if c[0] == "step"]
    primed = steps[0][1].shape[-1] == 6 and steps[0][1][0].T.tolist() == tiny.ref_codes(6).tolist()
    assert primed == icl
    assert sum(len(c.pcm) for c in out) == out[-1].info.frames * 1920


# ---------------------------------------------------------------------------------------------------- public API
@pytest.mark.parametrize("name", list(tiny.VARIANTS))
def test_api_runs_every_planned_mode(ckpt, tmp_path, name):
    """Lure.load(backend="mlx") + the run list of real_modes.py, on the tiny checkpoint of each variant."""
    from lure_tts import Lure
    from real_modes import plan
    tts = Lure.load(ckpt.dir(name), backend="mlx", cache_dir=str(tmp_path), dtype="float32", cache_limit_mb=64)
    try:
        tts.backend.model.speech_tokenizer = StubCodec()
        runs = plan(tts.spec)
        assert {r["mode"] for r in runs} == set(tts.spec.modes)
        assert {r["layout"] for r in runs} == {"non_streaming", "streaming"}
        voice = tts.make_voice((np.zeros(24000 * 2, np.float32), 24000), "Hello there.") if tts.spec.is_clone else None
        for r in runs:
            kw = {k: v for k, v in r.items() if k not in ("name", "mode")}
            infos = []
            audio = tts.speak("Hi.", voice, seed=3, infos=infos, **kw)
            assert infos and all(i.extra["mode"] == r["mode"] and i.extra["layout"] == r["layout"] for i in infos)
            assert len(audio) > 0
    finally:
        tts.close()
