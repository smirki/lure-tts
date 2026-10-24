"""Core prompt builder vs the qwen-tts reference dump (conformance/reference)."""
import json
import os

import pytest

from lure_tts.prompt import ModelIds, build_layout, build_prompt_ids, layout_spans
from lure_tts.text import segment_text
from conftest import REF

PROMPTS = json.load(open(os.path.join(REF, "prompt_ids.json")))
TOKS = json.load(open(os.path.join(REF, "tokenizer_ids.json")))


@pytest.fixture(scope="module")
def mid(tokdir):
    return ModelIds.from_config(tokdir)


@pytest.mark.parametrize("e", TOKS["entries"], ids=lambda e: e["text"][:30])
def test_tokenizer_matches_qwen_processor(tok, e):
    assert tok.encode(e["text"]) == e["ids"]


@pytest.mark.parametrize("e", PROMPTS["entries"], ids=lambda e: e["id"])
def test_prompt_ids_and_layout(tok, mid, e):
    p = build_prompt_ids(tok, e["text"], e["ref_text"], e["direction"])
    assert p.input_ids == e["input_ids"]
    assert p.ref_ids == e["ref_ids"]
    assert p.instruct_ids == e["instruct_ids"]
    lay = build_layout(p, mid, e["n_ref_frames"], e["language"])
    assert lay == e["layout"]
    sp = layout_spans(p, mid, e["n_ref_frames"], e["language"])
    assert sp["total"] == len(lay)
    assert e["trailing_text_hidden"] == [[mid.tts_pad, None]]
    assert e["talker_kwargs"]["repetition_penalty"] == 1.05
    lo, hi, _, keep = e["suppress_tokens_range"]
    assert set(range(lo, hi)) - set(keep) == set(mid.suppress_tokens())


def test_reference_segments_still_match_segmenter():
    """The e.segNN entries were cut by the core segmenter; if the segmenter changes, regenerate the reference."""
    from cases import CASES
    segs = [s.text for s in segment_text(CASES["e"][0])]
    ref = [e["text"] for e in PROMPTS["entries"] if e["id"].startswith("e.seg")]
    assert segs == ref


def test_special_tokens_single_ids(tok):
    ids = tok.encode("<|im_start|>user\nX<|im_end|>\n")
    assert ids[0] == 151644 and ids[-2] == 151645
    assert tok.encode("<tts_text_bos_single>") == [151674]


def test_decode_roundtrip(tok):
    s = "Café (laugh) <|im_end|> 3.5%"
    assert tok.decode(tok.encode(s)) == s


def test_layout_no_direction_vs_direction(tok, mid):
    a = build_prompt_ids(tok, "Hi.", "Ref.", None)
    b = build_prompt_ids(tok, "Hi.", "Ref.", "   ")
    assert a == b and a.instruct_ids is None
    c = build_prompt_ids(tok, "Hi.", "Ref.", "Warm.")
    la, lc = build_layout(a, mid, 3), build_layout(c, mid, 3)
    assert lc[len(c.instruct_ids):] == la
