"""Row-wise sampling == transformers logits processors (what qwen-tts generate() uses), on CPU."""
import pytest
import torch

from lure_cuda.sampling import (SamplingParams, apply_repetition_penalty_rows, mask_rows, resolve_sampling,
                                  sample_rows, suppress_mask, warp_rows)


def hf_process(logits, history, rep, temp, top_k, top_p):
    from transformers.generation.logits_process import (RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
                                                        TopKLogitsWarper, TopPLogitsWarper)

    x = logits.clone()
    if history is not None and history.numel():
        x = RepetitionPenaltyLogitsProcessor(rep)(history.view(1, -1), x)
    x = TemperatureLogitsWarper(temp)(None, x)
    if top_k:
        x = TopKLogitsWarper(top_k)(None, x)
    if top_p < 1.0:
        x = TopPLogitsWarper(top_p)(None, x)
    return x


@pytest.mark.parametrize("temp,top_k,top_p,rep", [(0.9, 50, 1.0, 1.05), (0.7, 10, 0.8, 1.3), (1.3, 0, 0.95, 1.0),
                                                  (0.5, 1, 1.0, 1.05)])
def test_rows_match_hf(temp, top_k, top_p, rep):
    g = torch.Generator().manual_seed(0)
    V = 3072
    for _ in range(5):
        logits = torch.randn(1, V, generator=g) * 4
        hist = torch.randint(0, 2048, (37,), generator=g)
        ref = hf_process(logits, hist, rep, temp, top_k, top_p)
        seen = torch.zeros(1, V, dtype=torch.bool)
        seen[0, hist] = True
        x = apply_repetition_penalty_rows(logits, seen, torch.tensor([rep]))
        got = warp_rows(x, torch.tensor([temp]), torch.tensor([top_k]), torch.tensor([top_p]))
        assert torch.equal(torch.isinf(ref), torch.isinf(got))
        fin = ~torch.isinf(ref)
        torch.testing.assert_close(got[fin], ref[fin])


def test_mixed_rows_are_independent():
    g = torch.Generator().manual_seed(1)
    V = 2048
    logits = torch.randn(3, V, generator=g) * 3
    params = [(0.9, 50, 1.0), (0.6, 5, 0.7), (1.0, 0, 0.9)]
    batch = warp_rows(logits, torch.tensor([p[0] for p in params]), torch.tensor([p[1] for p in params]),
                      torch.tensor([p[2] for p in params]))
    for i, (t, k, p) in enumerate(params):
        one = warp_rows(logits[i:i + 1], torch.tensor([t]), torch.tensor([k]), torch.tensor([p]))
        assert torch.equal(batch[i:i + 1], one)


def test_gumbel_sampling_distribution():
    """Gumbel-max over the warped logits draws from softmax(warped)."""
    V, N = 8, 40000
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0, -2.0, 3.0, 0.1]])
    temp, k = 0.9, 5
    w = warp_rows(logits, torch.tensor([temp]), torch.tensor([k]), torch.tensor([1.0]))
    probs = w.softmax(-1)[0]
    g = torch.Generator().manual_seed(7)
    u = torch.rand(N, V, generator=g)
    toks = sample_rows(logits.expand(N, V), temperature=torch.full((N,), temp), top_k=torch.full((N,), k),
                       top_p=torch.ones(N), do_sample=torch.ones(N, dtype=torch.bool), uniform=u)
    freq = torch.bincount(toks, minlength=V).double() / N
    assert torch.all(freq[probs == 0] == 0)
    assert torch.allclose(freq, probs.double(), atol=0.01)


def test_greedy_rows_and_masks():
    V, eos = 3072, 2150
    logits = torch.zeros(2, V)
    logits[:, eos] = 10.0
    logits[:, 2500] = 20.0  # suppressed region
    logits[:, 7] = 5.0
    sup = suppress_mask(V, eos)
    assert sup[2048] and sup[3071] and not sup[eos] and not sup[2047]
    x = mask_rows(logits, sup, eos, torch.tensor([True, False]))
    tok = sample_rows(x, temperature=torch.ones(2), top_k=torch.zeros(2, dtype=torch.long), top_p=torch.ones(2),
                      do_sample=torch.zeros(2, dtype=torch.bool), uniform=torch.rand(2, V))
    assert tok.tolist() == [7, eos]


def test_resolve_sampling():
    assert resolve_sampling(None) == SamplingParams()
    sp = resolve_sampling({"temperature": 0.7, "subtalker_dosample": False, "max_tokens": 99})
    assert sp.temperature == 0.7 and sp.subtalker_do_sample is False and sp.max_new_tokens == 99
    assert sp.top_k == 50 and sp.top_p == 1.0 and sp.repetition_penalty == 1.05
    with pytest.raises(ValueError):
        resolve_sampling({"bogus": 1})
    with pytest.raises(ValueError):
        resolve_sampling({"top_p": 0})
    core = pytest.importorskip("lure_tts.defaults")
    assert resolve_sampling(core.Sampling()) == SamplingParams()
    assert resolve_sampling(core.Sampling().with_(top_k=20)).top_k == 20
