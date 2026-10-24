"""Lure API / CLI / server over every model variant, on the fake backend (no weights)."""
import base64
import io
import json

import numpy as np
import pytest

from lure_tts.api import ModeError
from lure_tts.cli import main
from conftest import STOCK, make_model_dir

SILENCE = (np.zeros(24000, np.float32), 24000)


def test_base_clone_defaults_follow_qwen_tts(stock_tts):
    tts = stock_tts("base06")
    tts.make_voice(SILENCE, "Ref text.", name="r")
    list(tts.stream("Hello (laugh) there. Second sentence here.", "r"))
    req, pr = tts.backend.requests[-1], tts.backend.prompts[-1]
    assert (req.mode, req.layout, req.language, req.speaker) == ("icl", "streaming", "auto", None)
    assert req.prompt.ref_ids is not None and "icl_overlay" in pr.spans


def test_base_xvec(stock_tts):
    tts = stock_tts("base17")
    v = tts.make_voice(SILENCE, None, name="x")
    assert v.ref_text == "" and v.spk_emb.size == 2048
    list(tts.stream("Hello there.", "x", layout="non_streaming"))
    req = tts.backend.requests[-1]
    assert (req.mode, req.layout) == ("xvec", "non_streaming") and req.prompt.ref_ids is None
    tts.make_voice(SILENCE, "Ref.", name="y")
    list(tts.stream("Hello there.", "y", x_vector_only=True))
    assert tts.backend.requests[-1].mode == "xvec"


def test_custom_voice(stock_tts):
    tts = stock_tts("cv17")
    list(tts.stream("Hi there, friend.", speaker="Vivian", instruct="Cheerful.", language="English"))
    req = tts.backend.requests[-1]
    assert (req.mode, req.speaker, req.speaker_id, req.layout, req.language) == (
        "custom_voice", "vivian", 3065, "non_streaming", "english")
    assert req.voice is None and req.prompt.instruct_ids
    assert [3065 == r[1] for r in tts.backend.prompts[-1].prefill].count(True) == 1


def test_voice_design(stock_tts):
    tts = stock_tts("vd17")
    list(tts.stream("Hi there.", instruct="A deep calm voice."))
    req = tts.backend.requests[-1]
    assert (req.mode, req.voice, req.speaker) == ("voice_design", None, None)


@pytest.mark.parametrize("key,kw,msg", [
    ("base06", {"speaker": "vivian"}, "no preset speakers"),
    ("base06", {}, "needs a voice"),
    ("base06", {"voice": "r", "instruct": "Warm."}, "does not accept instruct"),
    ("base06", {"voice": "r", "language": "klingon"}, "not supported"),
    ("cv17", {}, "needs a preset speaker"),
    ("cv17", {"speaker": "nobody"}, "unknown speaker"),
    ("cv17", {"voice": "r"}, "does not clone"),
    ("cv06", {"speaker": "vivian", "instruct": "Warm."}, "drops instruct"),
    ("vd17", {}, "needs instruct"),
    ("vd17", {"instruct": "   "}, "needs instruct"),
    ("vd17", {"speaker": "vivian", "instruct": "Warm."}, "VoiceDesign"),
])
def test_mode_errors(stock_tts, key, kw, msg):
    tts = stock_tts(key)
    if key == "base06":
        tts.make_voice(SILENCE, "Ref.", name="r")
    with pytest.raises(ModeError, match=msg):
        list(tts.stream("Hello.", **kw))


def test_voice_and_speaker_together(stock_tts):
    with pytest.raises(ModeError, match="not both"):
        list(stock_tts("cv17").stream("Hello.", voice="a", speaker="vivian"))


def test_make_voice_needs_base(stock_tts):
    with pytest.raises(ModeError, match="cannot clone"):
        stock_tts("cv17").make_voice(SILENCE, "Ref.")


def test_voice_from_other_model_size(stock_tts, tmp_path):
    a = stock_tts("base06")
    v = a.make_voice(SILENCE, "Ref.", name="small")
    with pytest.raises(ModeError, match="1024-d"):
        list(stock_tts("base17").stream("Hello.", v))


def test_stock_model_segments_without_tag_rules(stock_tts):
    text = "He paused for a long moment by the window (he said it twice. Then he left the house) and the door closed behind him slowly, without a single sound at all."
    vd = stock_tts("vd17")
    list(vd.stream(text, instruct="Calm."))
    texts = [r.text for r in vd.backend.requests]
    assert texts == ["He paused for a long moment by the window (he said it twice.",
            "Then he left the house) and the door closed behind him slowly, without a single sound at all."]


def test_inline_tags_override(stock_tts):
    tts = stock_tts("base06", inline_tags=True, layout="non_streaming")
    assert tts.spec.inline_tags and tts.spec.layout == "non_streaming"


def test_cli_info(tmp_path, capsys):
    d = make_model_dir(tmp_path, STOCK["cv17"])
    assert main(["info", d]) == 0
    out = capsys.readouterr().out
    assert "custom_voice" in out and "eric (sichuan_dialect)" in out
    assert main(["info", d, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["speakers"][0] == "aiden"


def test_cli_prompt_modes(tmp_path, capsys):
    d = make_model_dir(tmp_path, STOCK["cv17"])
    assert main(["prompt", "Hi there.", "--model", d, "--speaker", "eric", "--language", "chinese", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "custom_voice" and [151671, 2062] in out["layout"] and [151671, 2875] in out["layout"]
    d = make_model_dir(tmp_path, STOCK["base06"])
    assert main(["prompt", "Hi there.", "--model", d, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["layout_mode"] == "streaming" and out["mode"] == "icl"


def test_cli_say_speaker_and_design(tmp_path):
    import soundfile as sf
    cv = make_model_dir(tmp_path, STOCK["cv17"])
    out = tmp_path / "o.wav"
    assert main(["say", "Hello there.", "--speaker", "ryan", "-o", str(out), "--backend", "fake", "--model", cv,
                 "--cache", str(tmp_path)]) == 0
    assert len(sf.read(out)[0]) > 0
    vd = make_model_dir(tmp_path, STOCK["vd17"])
    assert main(["say", "Hello there.", "--instruct", "Deep voice.", "-o", str(out), "--backend", "fake", "--model", vd,
                 "--cache", str(tmp_path)]) == 0


def wav_b64():
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(24000, np.float32), 24000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def client_for(tts):
    from fastapi.testclient import TestClient
    from lure_tts.server import create_app
    return TestClient(create_app(tts))


def test_server_custom_voice(stock_tts):
    tts = stock_tts("cv17")
    c = client_for(tts)
    v = c.get("/v1/voices").json()
    assert v["kind"] == "speaker" and "vivian" in v["voices"]
    m = c.get("/v1/models").json()["data"][0]
    assert m["variant"] == "custom_voice" and "vivian" in m["speakers"] and m["layout"] == "non_streaming"
    r = c.post("/v1/audio/speech", json={"input": "Hi there.", "voice": "Serena", "instructions": "Warm.", "stream": False})
    assert r.status_code == 200 and tts.backend.requests[-1].speaker == "serena"
    r = c.post("/v1/audio/speech", json={"input": "Hi.", "voice": "alloy"})
    assert r.status_code == 400 and "preset speakers" in r.json()["error"]["message"]
    r = c.post("/v1/audio/speech", json={"input": "Hi.", "voice": {"ref_audio": wav_b64(), "ref_text": "x"}})
    assert r.status_code == 400
    assert c.post("/v1/voices", json={"name": "me", "ref_audio": wav_b64(), "ref_text": "Ref."}).status_code == 400


def test_server_voice_design(stock_tts):
    tts = stock_tts("vd17")
    c = client_for(tts)
    assert c.get("/v1/voices").json()["kind"] == "none"
    r = c.post("/v1/audio/speech", json={"input": "Hi there.", "instructions": "Deep voice.", "stream": False})
    assert r.status_code == 200 and tts.backend.requests[-1].mode == "voice_design"
    r = c.post("/v1/audio/speech", json={"input": "Hi.", "voice": "alloy", "instructions": "Deep.", "language": "English"})
    assert r.status_code == 200
    r = c.post("/v1/audio/speech", json={"input": "Hi.", "voice": "alloy"})
    assert r.status_code == 400 and "needs instruct" in r.json()["error"]["message"]


def test_server_base_xvec_and_instruct_error(stock_tts):
    tts = stock_tts("base06")
    c = client_for(tts)
    r = c.post("/v1/audio/speech", json={"input": "Hi there.", "voice": {"ref_audio": wav_b64()}, "stream": False})
    assert r.status_code == 200 and tts.backend.requests[-1].mode == "xvec"
    c.post("/v1/voices", json={"name": "me", "ref_audio": wav_b64(), "ref_text": "Ref."})
    r = c.post("/v1/audio/speech", json={"input": "Hi.", "voice": "me", "instructions": "Warm."})
    assert r.status_code == 400 and "instruct" in r.json()["error"]["message"]
