"""Core prompt builder vs the qwen-tts reference dump for every mode and layout (conformance/reference/prompt_modes.json)."""
import json
import os

import pytest

from lure_tts.prompt import ModelIds, build_layout, build_prompt, build_prompt_ids
from conftest import REF

MODES = json.load(open(os.path.join(REF, "prompt_modes.json")))
CONFIGS = os.path.join(REF, "configs")


def ids_for(cfg):
    return ModelIds.from_config(os.path.join(CONFIGS, cfg))


@pytest.mark.parametrize("e", MODES["entries"], ids=lambda e: e["id"])
def test_mode_prompt_matches_qwen_tts(tok, e):
    p = build_prompt_ids(tok, e["text"], e["ref_text"], e["direction"])
    assert p.input_ids == e["input_ids"]
    assert p.ref_ids == e["ref_ids"]
    assert p.instruct_ids == e["instruct_ids"]
    pr = build_prompt(p, ids_for(e["config"]), mode=e["mode"], layout=e["prompt_layout"], language=e["language"],
                      n_ref_frames=e["n_ref_frames"], speaker=e["speaker"])
    assert pr.prefill == e["layout"]
    assert pr.trailing == e["trailing"]
    assert pr.spans["total"] == len(pr.prefill)
    assert build_layout(p, ids_for(e["config"]), e["n_ref_frames"], e["language"], mode=e["mode"],
                        layout=e["prompt_layout"], speaker_name=e["speaker"]) == e["layout"]


def test_reference_covers_every_mode_and_layout():
    combos = {(e["mode"], e["prompt_layout"]) for e in MODES["entries"]}
    assert combos == {(m, lay) for m in ("icl", "xvec", "custom_voice", "voice_design")
                      for lay in ("non_streaming", "streaming")}


def test_dialect_speakers():
    mid = ids_for("Qwen3-TTS-12Hz-1.7B-CustomVoice.json")
    assert mid.language_id("chinese", "eric") == mid.language_ids["sichuan_dialect"]
    assert mid.language_id("auto", "Dylan") == mid.language_ids["beijing_dialect"]
    assert mid.language_id("english", "eric") == mid.language_ids["english"]
    assert mid.language_id("auto", "ryan") is None
    assert mid.speaker_id("VIVIAN") == 3065
    with pytest.raises(ValueError):
        mid.speaker_id("nobody")


def test_mode_argument_errors(tok):
    mid = ids_for("Qwen3-TTS-12Hz-1.7B-CustomVoice.json")
    p = build_prompt_ids(tok, "Hi.")
    with pytest.raises(ValueError):
        build_prompt(p, mid, mode="icl")  # no ref_ids
    with pytest.raises(ValueError):
        build_prompt(p, mid, mode="custom_voice")  # no speaker
    with pytest.raises(ValueError):
        build_prompt(p, mid, mode="voice_design", speaker="vivian")
    with pytest.raises(ValueError):
        build_prompt(p, mid, mode="voice_design", layout="sideways")
    with pytest.raises(ValueError):
        build_prompt(p, mid, mode="voice_design", language="klingon")


def test_streaming_icl_trailing_is_text_tail(tok, tokdir):
    mid = ModelIds.from_config(tokdir)
    p = build_prompt_ids(tok, "A fairly long target sentence for the overlay.", "Reference words here.")
    pr = build_prompt(p, mid, mode="icl", layout="streaming", n_ref_frames=3)
    x = p.ref_text_ids + p.text_ids + [mid.tts_eos]
    overlay = pr.prefill[pr.spans["icl_overlay"][0]:]
    assert [r[0] for r in overlay] == x[:4] and [r[1] for r in overlay] == [mid.codec_bos, "ref:0", "ref:1", "ref:2"]
    assert [r[0] for r in pr.trailing] == x[4:]
