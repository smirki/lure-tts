"""Binding checks that need the built library but no model weights."""
import ctypes
import json
import os

import pytest

import lure_cpp

REFERENCE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "conformance", "reference")
TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")


@pytest.fixture(scope="module")
def lib():
    try:
        path = lure_cpp._find_lib()
    except FileNotFoundError:
        pytest.skip("liblure_tts not built")
    return lure_cpp._bind(ctypes.CDLL(path))


@pytest.fixture(scope="module")
def stock(tmp_path_factory):
    """Prompt-only folders (reference tokenizer + one stock config.json) keyed by checkpoint name."""
    root = tmp_path_factory.mktemp("stock")
    out = {}
    for config in sorted(os.listdir(os.path.join(REFERENCE, "configs"))):
        name = os.path.splitext(config)[0].replace("Qwen3-TTS-12Hz-", "")
        d = root / name
        d.mkdir()
        for f in TOKENIZER_FILES:
            os.symlink(os.path.abspath(os.path.join(REFERENCE, "tokenizer", f)), d / f)
        os.symlink(os.path.abspath(os.path.join(REFERENCE, "configs", config)), d / "config.json")
        out[name] = str(d)
    return out


def _take(lib, ptr):
    assert ptr, lib.lure_last_error().decode()
    try:
        return json.loads(ctypes.string_at(ptr).decode("utf-8"))
    finally:
        lib.lure_string_free(ptr)


def _dump(lib, model_dir, text, ref_text=None, ref_frames=0, **kw):
    r = lure_cpp.make_request(lib, text, **kw)
    return lib.lure_dump_prefill_offline(model_dir.encode(), ctypes.byref(r),
                                           None if ref_text is None else ref_text.encode(), ref_frames)


def test_abi_version(lib):
    assert lib.lure_abi_version() == lure_cpp.ABI_VERSION == 2


def test_params_layout_matches_c(lib):
    params = lure_cpp._Params()
    lib.lure_default_params(ctypes.byref(params))
    assert params.struct_size == ctypes.sizeof(lure_cpp._Params)
    assert (params.temperature, params.top_k, params.top_p) == (pytest.approx(0.9), 50, pytest.approx(1.0))
    assert params.repetition_penalty == pytest.approx(1.05)
    assert params.chunk_grow == 1


def test_request_layout_matches_c(lib):
    r = lure_cpp._Request()
    lib.lure_default_request(ctypes.byref(r))
    assert r.struct_size == ctypes.sizeof(lure_cpp._Request)
    assert (r.mode, r.layout, r.speaker_id) == (0, -1, -1)
    assert not r.text and not r.voice


def test_build_info_names_backends(lib):
    assert lib.lure_build_info().decode()


def test_load_failure_sets_error(lib):
    assert not lib.lure_load(b"/nonexistent/model", b"cpu", 0, 1, None)
    assert lib.lure_last_error().decode()


@pytest.mark.parametrize("name,variant,hidden,modes,instruct", [
    ("0.6B-Base", "base", 1024, ["icl", "xvec"], "none"),
    ("1.7B-Base", "base", 2048, ["icl", "xvec"], "none"),
    ("0.6B-CustomVoice", "custom_voice", 1024, ["custom_voice"], "none"),
    ("1.7B-CustomVoice", "custom_voice", 2048, ["custom_voice"], "optional"),
    ("1.7B-VoiceDesign", "voice_design", 2048, ["voice_design"], "required"),
])
def test_model_info_offline(lib, stock, name, variant, hidden, modes, instruct):
    info = _take(lib, lib.lure_model_info_offline(stock[name].encode()))
    assert info["variant"] == variant
    assert info["talker_hidden_size"] == hidden
    assert info["code_predictor_hidden_size"] == 1024
    assert info["modes"] == modes
    assert info["instruct"] == instruct
    assert bool(info["speakers"]) == (variant == "custom_voice")
    assert info["default_layout"] == ("streaming" if variant == "base" else "non_streaming")


def test_model_info_lure(lib):
    info = _take(lib, lib.lure_model_info_offline(os.path.abspath(os.path.join(REFERENCE, "tokenizer")).encode()))
    assert info["family"] == "lure"
    assert info["default_layout"] == "non_streaming"
    assert info["default_language"] == "english"


def test_custom_voice_speaker_row(lib, stock):
    rec = _take(lib, _dump(lib, stock["1.7B-CustomVoice"], "Hi there.", mode="custom_voice", speaker="Ryan",
                           language="english", layout="non_streaming"))
    rows = lure_cpp.recipe_rows(rec)
    assert [None, 3061] not in rows and [151671, 3061] in rows
    assert lure_cpp.recipe_rows(rec, "trailing_") == [[151671, None]]
    assert (rec["mode"], rec["layout"]) == ("custom_voice", "non_streaming")


def test_streaming_trailing_ends_with_tts_eos(lib, stock):
    rec = _take(lib, _dump(lib, stock["1.7B-VoiceDesign"], "Hi there.", mode="voice_design",
                           instruct="A calm voice.", layout="streaming"))
    trailing = lure_cpp.recipe_rows(rec, "trailing_")
    assert trailing[-1] == [151673, None]
    assert rec["instruct_ids"]


def test_xvec_has_speaker_row_and_no_ref(lib, stock):
    rec = _take(lib, _dump(lib, stock["1.7B-Base"], "Hi there.", mode="xvec", layout="non_streaming"))
    rows = lure_cpp.recipe_rows(rec)
    assert [151671, "spk"] in rows
    assert not any(isinstance(c, str) and c.startswith("ref:") for _, c in rows)
    assert not rec["ref_ids"]


@pytest.mark.parametrize("name,kw,needle", [
    ("0.6B-CustomVoice", dict(mode="custom_voice", speaker="ryan", instruct="Happy."), "instruct"),
    ("1.7B-CustomVoice", dict(mode="custom_voice", speaker="nobody"), "unknown speaker"),
    ("1.7B-CustomVoice", dict(mode="icl"), "base"),
    ("0.6B-Base", dict(mode="custom_voice", speaker="ryan"), "custom_voice"),
    ("1.7B-VoiceDesign", dict(mode="voice_design", language="klingon"), "language"),
])
def test_mode_errors(lib, stock, name, kw, needle):
    ref_text = "ref" if kw.get("mode") == "icl" else None
    assert not _dump(lib, stock[name], "Hi.", ref_text=ref_text, ref_frames=2, **kw)
    assert needle in lib.lure_last_error().decode().lower()


def test_make_request_rejects_bad_values(lib):
    with pytest.raises(ValueError):
        lure_cpp.make_request(lib, "x", mode="clone")
    with pytest.raises(ValueError):
        lure_cpp.make_request(lib, "x", layout="chunked")
