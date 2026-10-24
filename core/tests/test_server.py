import base64
import io
import json
import struct

import numpy as np
import pytest

from lure_tts.server import InlineVoice, RequestError, create_app, parse_speech_request


def wav_b64(n=24000, sr=24000):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(n, np.float32), sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def test_minimal_openai_body():
    r = parse_speech_request({"model": "tts-1", "input": "Hello.", "voice": "willow"})
    assert r.voice == "willow" and r.response_format == "wav" and r.stream and r.stream_format == "audio"
    assert r.sampling.repetition_penalty == 1.05 and r.instruct is None


def test_extra_fields():
    r = parse_speech_request({"input": "Hi (laugh).", "voice": "v", "instruct": "Warm.", "seed": 3, "stream": False,
                              "response_format": "pcm", "sampling": {"temperature": 0.8, "top_k": 40}})
    assert r.instruct == "Warm." and r.seed == 3 and not r.stream and r.response_format == "pcm"
    assert r.sampling.temperature == 0.8 and r.sampling.top_k == 40 and r.sampling.top_p == 1.0


def test_instructions_alias_and_inline_voice():
    r = parse_speech_request({"input": "Hi.", "instructions": "Calm.", "voice": {"ref_audio": wav_b64(), "ref_text": "Ref text."}})
    assert r.instruct == "Calm." and isinstance(r.voice, InlineVoice) and r.voice.ref_text == "Ref text."


def test_sse_forces_stream_and_flac_forces_whole():
    assert parse_speech_request({"input": "a", "voice": "v", "stream_format": "sse", "stream": False}).stream
    assert not parse_speech_request({"input": "a", "voice": "v", "response_format": "flac"}).stream


@pytest.mark.parametrize("body", [
    {"voice": "v"},
    {"input": "", "voice": "v"},
    {"input": "a"},
    {"input": "a", "voice": "v", "speed": 1.5},
    {"input": "a", "voice": "v", "response_format": "mp3"},
    {"input": "a", "voice": "v", "response_format": "ogg"},
    {"input": "a", "voice": "v", "stream_format": "ws"},
    {"input": "a", "voice": "v", "sampling": {"temperature": -1}},
    {"input": "a", "voice": "v", "sampling": {"top_p": 0}},
    {"input": "a", "voice": "v", "sampling": {"repetition_penalty": 0.5}},
    {"input": "a", "voice": "v", "sampling": {"bogus": 1}},
    {"input": "a", "voice": "v", "sampling": {"top_k": 1.5}},
    {"input": "a", "voice": "v", "seed": -1},
    {"input": "a", "voice": "v", "seed": True},
    {"input": "a", "voice": "v", "instructions": "A", "instruct": "B"},
    {"input": "a", "voice": "v", "unknown": 1},
    {"input": "a", "voice": {"ref_audio": "!!!", "ref_text": "x"}},
    {"input": "a", "voice": {"ref_audio": wav_b64()}},
    {"input": "a", "voice": 3},
    {"input": "a", "voice": "v", "stream": "yes"},
    {"input": "x" * 20001, "voice": "v"},
])
def test_rejects(body):
    with pytest.raises(RequestError):
        parse_speech_request(body)


@pytest.fixture()
def client(fake_tts):
    from fastapi.testclient import TestClient
    return TestClient(create_app(fake_tts))


def test_voice_upload_and_stream_wav(client, fake_tts):
    r = client.post("/v1/voices", json={"name": "me", "ref_audio": wav_b64(), "ref_text": "Ref."})
    assert r.status_code == 200 and r.json()["voice"] == "me"
    assert "me" in client.get("/v1/voices").json()["voices"]
    r = client.post("/v1/audio/speech", json={"input": "Hello there. Second sentence, longer one.", "voice": "me",
                                               "instructions": "Warm.", "seed": 5})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/wav")
    data = r.content
    assert data[:4] == b"RIFF" and struct.unpack("<I", data[40:44])[0] == 0xFFFFFFFF
    assert len(data) > 44 + 1000
    req = fake_tts.backend.requests[-1]
    assert req.instruct == "Warm." and req.prompt.instruct_ids[0] == 151644 and req.seed == 5
    assert req.sampling.repetition_penalty == 1.05


def test_whole_wav_pcm_and_sse(client):
    client.post("/v1/voices", json={"name": "me", "ref_audio": wav_b64(), "ref_text": "Ref."})
    r = client.post("/v1/audio/speech", json={"input": "Hi there you.", "voice": "me", "stream": False})
    assert r.status_code == 200 and struct.unpack("<I", r.content[40:44])[0] == len(r.content) - 44
    r = client.post("/v1/audio/speech", json={"input": "Hi there you.", "voice": "me", "response_format": "pcm"})
    assert r.status_code == 200 and len(r.content) % 2 == 0 and len(r.content) > 0
    r = client.post("/v1/audio/speech", json={"input": "Hi there you.", "voice": "me", "stream_format": "sse"})
    events = [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1]["type"] == "speech.audio.done" and events[0]["type"] == "speech.audio.delta"
    assert events[-1]["segments"][0]["ended_by_eos"] is True


def test_inline_voice_and_errors(client):
    r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": {"ref_audio": wav_b64(), "ref_text": "Ref."}, "stream": False})
    assert r.status_code == 200
    r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "nobody"})
    assert r.status_code == 400 and "unknown voice" in r.json()["error"]["message"]
    r = client.post("/v1/audio/speech", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert client.get("/v1/models").json()["data"][0]["id"] == "lure"
