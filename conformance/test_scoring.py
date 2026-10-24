"""Unit tests for the pure scoring helpers (no models). Run: pytest conformance"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scoring  # noqa: E402


def test_norm_and_wer():
    assert scoring.norm("(laugh) You can't, it's forty!") == ["you", "can", "not", "it", "is", "40"]
    assert scoring.wer(["a", "b", "c"], ["a", "x", "c"]) == 1 / 3


def test_tag_read_detected_and_gate():
    row = {"file": "t1.wav", "case": "t1", "kind": "tag", "text": "(chuckle) Okay, okay, you got me there.", "ended_by_eos": [True]}
    good = scoring.score_row(row, "Okay, okay, you got me there.", 0.7)
    bad = scoring.score_row(row, "Chuckle. Okay, okay, you got me there.", 0.7)
    assert good["tags_read"] == [] and good["wer"] == 0
    assert bad["tags_read"] == ["chuckle"]
    ref = scoring.summarize([good])
    assert scoring.gate(ref, ref) == []
    fails = scoring.gate(scoring.summarize([bad]), ref)
    assert any("tag words" in f for f in fails)
