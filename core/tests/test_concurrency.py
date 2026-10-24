"""Backends that set `concurrent = True` get one producer thread per stream; others are serialized on one worker."""
import threading

import numpy as np
import pytest

from conftest import TOKDIR


def run_two(tts, voice):
    out = {}

    def go(k):
        infos = []
        out[k] = (np.concatenate(list(tts.stream("one two three four five six. Seven eight nine ten.", voice, seed=k, infos=infos))), infos)

    ts = [threading.Thread(target=go, args=(k,)) for k in (1, 2)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    return out


@pytest.mark.parametrize("concurrent", [False, True])
def test_streams_overlap_only_when_backend_is_concurrent(tmp_path, concurrent):
    from lure_tts.api import Lure
    tts = Lure.load(TOKDIR, backend="fake", cache_dir=str(tmp_path), concurrent=concurrent, delay_s=0.02)
    try:
        assert tts.concurrent is concurrent
        v = tts.make_voice((np.zeros(24000, np.float32), 24000), "ref text", name="v")
        out = run_two(tts, v)
        assert set(out) == {1, 2}
        for audio, infos in out.values():
            assert len(audio) > 0 and len(infos) == 1 and infos[0].samples == len(audio)  # per-call infos, not mixed
        assert tts.backend.max_active == (2 if concurrent else 1)
    finally:
        tts.close()
