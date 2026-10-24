"""Prefill conformance: lure_cuda.prompt.build_prefill vs the qwen-tts 0.1.1 dumps in conformance/reference.

prompt_ids.json holds the icl non_streaming prompts (with and without instruct); prompt_modes.json holds every mode and
layout for the stock 0.6B / 1.7B Base, CustomVoice and VoiceDesign configs. A symbolic stub model makes every embedding
row decode back to an exact [text, codec] pair, so equality is token for token, trailing rows included. The same rows
are also checked against the core's builder (lure_tts.prompt.build_prompt)."""
import json
import os

import pytest

from conftest import CONFIGS, MODEL_DIR, MODES_REFERENCE, REFERENCE


def _load(path):
    if not os.path.exists(path):
        pytest.skip(f"{path} not found")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def reference():
    return _load(REFERENCE)


@pytest.fixture(scope="module")
def modes_reference():
    return _load(MODES_REFERENCE)


def _config(path):
    from lure_cuda._compat import import_qwen_tts

    import_qwen_tts()
    from qwen_tts.core.models import Qwen3TTSConfig

    return Qwen3TTSConfig.from_pretrained(path)


@pytest.fixture(scope="module")
def full_config():
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        pytest.skip("set LURE_MODEL_DIR to run")
    return _config(MODEL_DIR)


def test_prefill_rows_match_reference(reference, full_config):
    from lure_cuda.prompt import PromptIds, symbolic_rows

    for e in reference["entries"]:
        p = PromptIds(e["input_ids"], e["ref_ids"], e["instruct_ids"])
        got = symbolic_rows(full_config, p, e["n_ref_frames"], e["language"])
        assert got["layout"] == e["layout"], e["id"]
        assert got["trailing"] == e["trailing_text_hidden"], e["id"]
        assert got["tts_pad"] == e["tts_pad_embed"], e["id"]
    assert len(reference["entries"]) > 0


def test_prefill_rows_match_every_mode_and_layout(modes_reference):
    """All 40 prompt_modes.json entries: icl, xvec, custom_voice, voice_design x non_streaming, streaming."""
    from lure_cuda.prompt import PromptIds, symbolic_rows

    configs = {}
    seen = set()
    for e in modes_reference["entries"]:
        if e["config"] not in configs:
            configs[e["config"]] = _config(os.path.join(CONFIGS, e["config"]))
        p = PromptIds(e["input_ids"], e["ref_ids"], e["instruct_ids"])
        got = symbolic_rows(configs[e["config"]], p, e["n_ref_frames"], e["language"], e["mode"], e["prompt_layout"],
                            e["speaker"])
        assert got["layout"] == e["layout"], e["id"]
        assert got["trailing"] == e["trailing"], e["id"]
        seen.add((e["mode"], e["prompt_layout"]))
    assert len(modes_reference["entries"]) == 40
    assert len(seen) == 8


def test_prefill_rows_match_core_builder(modes_reference):
    pytest.importorskip("lure_tts.prompt")
    from lure_cuda.prompt import PromptIds, layout_from_core, symbolic_rows

    configs = {}
    for e in modes_reference["entries"]:
        path = os.path.join(CONFIGS, e["config"])
        if path not in configs:
            configs[path] = _config(path)
        p = PromptIds(e["input_ids"], e["ref_ids"], e["instruct_ids"])
        prefill, trailing = layout_from_core(p, path, e["n_ref_frames"], e["language"], e["mode"], e["prompt_layout"],
                                             e["speaker"].lower() if e["speaker"] else None)
        got = symbolic_rows(configs[path], p, e["n_ref_frames"], e["language"], e["mode"], e["prompt_layout"],
                            e["speaker"])
        assert got["layout"] == prefill, e["id"]
        assert got["trailing"] == trailing, e["id"]


def test_prefill_matches_core_layout(reference):
    pytest.importorskip("lure_tts.prompt")
    from lure_cuda.prompt import PromptIds, layout_from_core

    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        pytest.skip("set LURE_MODEL_DIR to run")
    for e in reference["entries"]:
        p = PromptIds(e["input_ids"], e["ref_ids"], e["instruct_ids"])
        assert layout_from_core(p, MODEL_DIR, e["n_ref_frames"], e["language"])[0] == e["layout"], e["id"]


def test_sampling_kwargs_in_reference_match_defaults(reference):
    from lure_cuda.sampling import SamplingParams

    sp = SamplingParams()
    for e in reference["entries"]:
        kw = e["talker_kwargs"]
        assert (kw["temperature"], kw["top_k"], kw["top_p"], kw["repetition_penalty"]) == (
            sp.temperature, sp.top_k, sp.top_p, sp.repetition_penalty)
        assert (kw["subtalker_temperature"], kw["subtalker_top_k"], kw["subtalker_top_p"]) == (
            sp.subtalker_temperature, sp.subtalker_top_k, sp.subtalker_top_p)
        assert kw["do_sample"] and kw["subtalker_dosample"] and kw["min_new_tokens"] == sp.min_new_tokens == 2


def test_token_ids_match_reference(reference, modes_reference):
    """qwen-tts processor (what this backend uses when the core passes no ids) gives the reference ids."""
    if not os.path.exists(os.path.join(MODEL_DIR, "vocab.json")):
        pytest.skip("set LURE_MODEL_DIR to run")
    from lure_cuda._compat import import_qwen_tts

    import_qwen_tts()
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor
    from transformers import AutoConfig, AutoProcessor

    from lure_cuda.prompt import PromptIds, build_prompt_ids

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL_DIR, fix_mistral_regex=True)
    for e in reference["entries"]:
        got = build_prompt_ids(proc, e["text"], e["ref_text"], e["direction"])
        assert got == PromptIds(e["input_ids"], e["ref_ids"], e["instruct_ids"]), e["id"]
    for e in modes_reference["entries"]:
        got = build_prompt_ids(proc, e["text"], e["ref_text"] if e["mode"] == "icl" else None, e["direction"])
        assert got == PromptIds(e["input_ids"], e["ref_ids"], e["instruct_ids"]), e["id"]
