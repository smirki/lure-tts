import numpy as np

from lure_tts.defaults import max_frames_for


def test_stream_segments_requests_and_pauses(fake_tts, tmp_path):
    v = fake_tts.make_voice((np.zeros(48000, np.float32), 48000), "Ref text.", name="r")
    assert v.n_ref_frames == 13  # 24000 samples after resample -> ceil(24000/1920)
    text = " ".join(f"Sentence number {i} has quite a few words in it." for i in range(6))
    chunks = list(fake_tts.stream(text, "r", instruct="Calm.", seed=10))
    reqs = fake_tts.backend.requests
    assert len(reqs) == len(fake_tts.last_infos) >= 2
    assert [r.seed for r in reqs] == [10 + i for i in range(len(reqs))]
    assert all(r.prompt.instruct_ids and r.instruct == "Calm." for r in reqs)
    assert all(r.max_new_tokens == max_frames_for(r.text) for r in reqs)
    assert " ".join(r.text for r in reqs) == text
    assert reqs[0].chunk_frames == 4 and all(len(c) for c in chunks)


def test_voice_cache_reuse_and_invalidate(fake_tts):
    wav = (np.random.default_rng(0).standard_normal(24000) * 0.1).astype(np.float32)
    a = fake_tts.make_voice((wav, 24000), "Ref.", name="x")
    b = fake_tts.make_voice((wav, 24000), "Ref.", name="x")
    assert a is b or a.meta["created"] == b.meta["created"]
    c = fake_tts.make_voice((wav, 24000), "Other ref.", name="x")
    assert c.ref_text == "Other ref."
    fake_tts.voices._mem.clear()
    d = fake_tts.voice("x")
    assert d.ref_text == "Other ref." and d.ref_codes.shape == (13, 16) and d.ref_codes.dtype == np.int32


def test_live_session_incremental(fake_tts):
    fake_tts.make_voice((np.zeros(24000, np.float32), 24000), "Ref.", name="r")
    live = fake_tts.live("r", seed=1)
    out = []
    for tok in "The storm had been building all afternoon, and by the time she arrived it was dark. (sigh) Then rain.".split(" "):
        out += list(live.feed(tok + " "))
    out += list(live.flush())
    texts = [r.text for r in fake_tts.backend.requests]
    assert texts[0] == "The storm had been building all afternoon,"
    assert all(t.count("(") == t.count(")") for t in texts)


def test_cancel_midway(fake_tts):
    fake_tts.make_voice((np.zeros(24000, np.float32), 24000), "Ref.", name="r")
    g = fake_tts.stream("One two three four five. " * 20, "r")
    next(g)
    g.close()  # must not hang; worker stays usable
    assert len(list(fake_tts.stream("Short one here.", "r"))) > 0
