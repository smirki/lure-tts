"""Batched engine on a tiny random Qwen3-TTS (CPU, float64, eager steps instead of CUDA graphs).

1. Greedy parity: engine codes == qwen-tts 0.1.1 `generate(non_streaming_mode=True)` codes (talker + sub-talker greedy,
   repetition penalty 1.05, suppress tokens, min_new_tokens) for ICL prompts with and without instruct.
2. Batch invariance: seeded sampled requests give the same frames alone and when sharing the batch, including slot
   reuse, late joins and KV rebases.
"""
import threading

import pytest
import torch

from conftest import make_prompt, make_voice_tensors

TEXTS = [
    ([1001, 2002, 3003, 4004, 5005], [11, 12, 13, 14], None),
    ([7, 8, 9], [21, 22, 23, 24, 25, 26], [301, 302, 303]),
    ([500, 600, 700, 800, 900, 1000, 1100], [31, 32], [401]),
    ([42], [51, 52, 53], None),
]


def _engine(m, B=4, max_seq_len=256):
    from lure_cuda.engine import BatchEngine

    return BatchEngine(m.talker, m.config.talker_config, max_batch=B, max_seq_len=max_seq_len, device="cpu",
                       dtype=torch.float64, use_graphs=False)


def _request(m, case, sampling, max_new, seed=0, n_ref=7):
    from lure_cuda.engine import Request
    from lure_cuda.prompt import build_prefill
    from lure_cuda.sampling import make_generator

    text, ref, ins = case
    codes, spk = make_voice_tensors(n_ref=n_ref, hidden=m.config.talker_config.hidden_size, seed=len(text))
    p = make_prompt(text, ref, ins)
    emb, _tr, pad = build_prefill(m, p, codes, spk, "english")
    return Request(prefill=emb, tts_pad=pad, sampling=sampling, max_new_tokens=max_new,
                   generator=make_generator(seed, "cpu")), (p, codes, spk)


def _frames(r):
    from lure_cuda.engine import drain

    return list(drain(r, timeout=0))


@pytest.mark.parametrize("case", TEXTS[:3])
def test_greedy_parity_with_qwen_tts(tiny_model, case):
    from lure_cuda.sampling import SamplingParams

    m = tiny_model
    max_new = 24
    sp = SamplingParams(do_sample=False, subtalker_do_sample=False)
    r, (p, codes, spk) = _request(m, case, sp, max_new)
    eng = _engine(m)
    eng.submit(r)
    eng.run_until_idle()
    got = _frames(r)

    import qwen_tts

    vcp = {"ref_code": [codes], "ref_spk_embedding": [spk], "x_vector_only_mode": [False], "icl_mode": [True]}
    ids = [torch.tensor([p.input_ids])]
    ref_ids = [torch.tensor([p.ref_ids])]
    ins = [torch.tensor([p.instruct_ids]) if p.instruct_ids else None]
    with torch.no_grad():
        out_codes, _ = m.generate(input_ids=ids, instruct_ids=ins, ref_ids=ref_ids, voice_clone_prompt=vcp,
                                  languages=["english"], non_streaming_mode=True, max_new_tokens=max_new,
                                  do_sample=False, subtalker_dosample=False, repetition_penalty=1.05)
    want = out_codes[0].tolist()
    assert len(got) > 0
    assert got == want[:len(got)]
    if r.ended_by_eos:
        assert len(got) == len(want)
    _ = qwen_tts


def _sampled(seed_case):
    from lure_cuda.sampling import SamplingParams

    return SamplingParams(temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.05)


def test_batch_invariance_and_rebase(tiny_model):
    m = tiny_model
    sp = _sampled(None)
    max_new = 20
    alone = []
    for i, case in enumerate(TEXTS):
        r, _ = _request(m, case, sp, max_new, seed=100 + i)
        e = _engine(m, B=1)
        e.submit(r)
        e.run_until_idle()
        alone.append(_frames(r))
    # all four at once in a batch of 2 (slot reuse) with a tight cache (forces rebases)
    e = _engine(m, B=2, max_seq_len=max(r.prefill.shape[1] for r in [_request(m, c, sp, 1)[0] for c in TEXTS]) + max_new + 6)
    reqs = []
    for i, case in enumerate(TEXTS):
        r, _ = _request(m, case, sp, max_new, seed=100 + i)
        reqs.append(e.submit(r))
    e.run_until_idle()
    assert e.rebases > 0
    for i, r in enumerate(reqs):
        assert _frames(r) == alone[i], i


def test_late_join_matches_alone(tiny_model):
    """A request admitted while another is mid-generation (different prefill length) is unaffected and unaffecting."""
    m = tiny_model
    sp = _sampled(None)
    a_alone, _ = _request(m, TEXTS[0], sp, 18, seed=5)
    b_alone, _ = _request(m, TEXTS[2], sp, 18, seed=6)
    for r in (a_alone, b_alone):
        e = _engine(m, B=1)
        e.submit(r)
        e.run_until_idle()
    fa, fb = _frames(a_alone), _frames(b_alone)

    e = _engine(m, B=4)
    a, _ = _request(m, TEXTS[0], sp, 18, seed=5)
    b, _ = _request(m, TEXTS[2], sp, 18, seed=6)
    e.submit(a)
    for _ in range(5):
        e._admit_pending()
        e.step()
    e.submit(b)
    e.run_until_idle()
    assert _frames(a) == fa
    assert _frames(b) == fb


def test_threaded_engine_and_cancel(tiny_model):
    from lure_cuda.engine import drain

    m = tiny_model
    sp = _sampled(None)
    e = _engine(m, B=2, max_seq_len=512)
    e.start()
    try:
        r1, _ = _request(m, TEXTS[0], sp, 15, seed=1)
        r2, _ = _request(m, TEXTS[1], sp, 400, seed=2)
        e.submit(r1)
        e.submit(r2)
        got = {}

        def consume(name, r, stop_after=None):
            out = []
            for f in drain(r, timeout=60):
                out.append(f)
                if stop_after and len(out) >= stop_after:
                    r.cancel()
            got[name] = out

        t1 = threading.Thread(target=consume, args=("a", r1))
        t2 = threading.Thread(target=consume, args=("b", r2, 3))
        t1.start(), t2.start()
        t1.join(60), t2.join(60)
        assert 1 <= len(got["a"]) <= 15
        assert r2.stop_reason in ("cancelled", "eos") and len(got["b"]) < 400
        assert all(len(f) == 16 for f in got["a"])
    finally:
        e.stop()


def test_submit_rejects_oversized(tiny_model):
    from lure_cuda.sampling import SamplingParams

    m = tiny_model
    e = _engine(m, B=1, max_seq_len=64)
    r, _ = _request(m, TEXTS[0], SamplingParams(), 60)
    with pytest.raises(ValueError):
        e.submit(r)
