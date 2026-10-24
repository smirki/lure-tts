"""CudaBackend plumbing on CPU: core GenerateRequest in, Chunks + SegmentInfo out; streaming decode windows; using the
tiny random model, eager engine and a fake causal codec (no real weights, no GPU)."""
import types

import numpy as np
import pytest
import torch

from conftest import UP, FakeSpeechTokenizer, make_voice_tensors


def test_stream_windows_equal_full_decode():
    from lure_cuda.decode import CodecDecoder, StreamDecoder, stream_audio

    codec = CodecDecoder(FakeSpeechTokenizer(), "cpu")
    g = torch.Generator().manual_seed(0)
    ref = torch.randint(0, 2048, (9, 16), generator=g)
    gen = torch.randint(0, 2048, (23, 16), generator=g)
    full = codec.decode_full(ref, gen)
    assert full.size == 23 * UP
    for first, chunk in ((4, 4), (1, 4), (2, 3), (7, 7)):
        dec = StreamDecoder(codec, ref, chunk_frames=chunk, first_chunk_frames=first, context_frames=25)
        parts = list(stream_audio(gen.tolist(), dec))
        assert parts[0].size == first * UP
        np.testing.assert_allclose(np.concatenate(parts), full, rtol=0, atol=1e-6)


@pytest.fixture()
def cpu_backend(tiny_model):
    from lure_cuda.backend import CudaBackend
    from lure_cuda.decode import CodecDecoder
    from lure_cuda.engine import BatchEngine

    m = tiny_model
    eng = BatchEngine(m.talker, m.config.talker_config, max_batch=2, max_seq_len=512, device="cpu",
                      dtype=torch.float64, use_graphs=False)
    eng.start()
    base = types.SimpleNamespace(model=m, processor=None)
    be = CudaBackend(base, eng, CodecDecoder(FakeSpeechTokenizer(), "cpu"), "tiny", "cpu", torch.float64,
                     {"decode_context_frames": 25})
    yield be
    be.close()


def _core_request(voice, text_ids, stream, seed=3, max_new=12, instruct_ids=None):
    base = pytest.importorskip("lure_tts.backends.base")
    from lure_tts.defaults import Sampling
    from lure_tts.prompt import PromptIds

    ins = None if instruct_ids is None else [151644, 872, 198] + instruct_ids + [151645, 198]
    p = PromptIds([151644, 77091, 198] + text_ids + [151645, 198, 151644, 77091, 198],
                  [151644, 77091, 198, 11, 12, 13, 151645, 198], ins)
    return base.GenerateRequest(text="x " * len(text_ids), voice=voice, prompt=p,
                                instruct="calm" if instruct_ids else None, sampling=Sampling(), seed=seed,
                                max_new_tokens=max_new, stream=stream, chunk_frames=4, first_chunk_frames=2)


def test_generate_core_request_stream_and_offline(cpu_backend):
    base = pytest.importorskip("lure_tts.backends.base")
    codes, spk = make_voice_tensors(n_ref=5, hidden=64)
    voice = base.Voice("v", "ref", codes.numpy().astype(np.int32), spk.numpy().astype(np.float32))
    chunks = list(cpu_backend.generate(_core_request(voice, [5, 6, 7, 8], True, instruct_ids=[9, 10])))
    info = chunks[-1].info
    assert info is not None and all(c.info is None for c in chunks[:-1])
    assert 1 <= info.frames <= 11 and info.samples == info.frames * UP
    assert chunks[0].pcm.size == 2 * UP
    assert sum(c.pcm.size for c in chunks) == info.samples
    assert info.ttfa_s is not None and info.extra["stop_reason"] in ("eos", "max_new_tokens")
    stream_pcm = np.concatenate([c.pcm for c in chunks])

    off = list(cpu_backend.generate(_core_request(voice, [5, 6, 7, 8], False, instruct_ids=[9, 10])))
    assert len(off) == 1 and off[0].info.frames == info.frames
    np.testing.assert_allclose(off[0].pcm, stream_pcm, atol=1e-6)  # same seed -> same frames -> same audio


def test_concurrent_generates_share_the_batch(cpu_backend):
    import threading

    base = pytest.importorskip("lure_tts.backends.base")
    codes, spk = make_voice_tensors(n_ref=5, hidden=64)
    voice = base.Voice("v", "ref", codes.numpy().astype(np.int32), spk.numpy().astype(np.float32))
    solo = {}
    for i in range(3):
        solo[i] = np.concatenate([c.pcm for c in cpu_backend.generate(_core_request(voice, [5 + i, 6], True, seed=i))])
    out = {}

    def run(i):
        out[i] = np.concatenate([c.pcm for c in cpu_backend.generate(_core_request(voice, [5 + i, 6], True, seed=i))])

    ts = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    [t.start() for t in ts]
    [t.join(120) for t in ts]
    for i in range(3):
        np.testing.assert_allclose(out[i], solo[i], atol=1e-6)


def test_quant_rejected():
    from lure_cuda.backend import CudaBackend

    with pytest.raises(NotImplementedError):
        CudaBackend.load("/nonexistent", quant="q4")
