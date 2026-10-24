"""OpenAI-compatible speech server.

POST /v1/audio/speech   OpenAI fields: model, input, voice, instructions, response_format (wav | pcm | flac), speed (1.0 only),
                        stream_format (audio | sse). Extra fields: instruct (alias of instructions), language, sampling
                        {temperature, top_k, top_p, repetition_penalty, subtalker_*, max_new_tokens}, seed, stream
                        (default true).
                        voice, by model variant:
                          Base          cached voice id, or {"ref_audio": <base64 wav>, "ref_text": "...", "name": optional}
                                        (ref_text may be omitted for an x-vector-only clone)
                          CustomVoice   preset speaker name (GET /v1/voices lists them)
                          VoiceDesign   optional and ignored; the voice comes from instructions
GET  /v1/voices, POST /v1/voices {name, ref_audio (base64), ref_text} (Base only), GET /v1/models, GET /health.
Streaming: wav = streaming WAV header + PCM16 chunks as they are generated; pcm = raw s16le 24 kHz mono;
sse = `data: {"type": "speech.audio.delta", "audio": <base64 pcm16>}` ... `{"type": "speech.audio.done", ...}`.
"""

import base64
import binascii
import io
import json
from dataclasses import dataclass, field
from typing import Optional, Union

from .audio import SAMPLE_RATE, load_audio, to_pcm16, wav_bytes, wav_header
from .defaults import Sampling

MAX_INPUT_CHARS = 20000
FORMATS = {"wav": "audio/wav", "pcm": "audio/pcm", "flac": "audio/flac"}
UNSUPPORTED = {"mp3", "opus", "aac"}


class RequestError(ValueError):
    pass


@dataclass
class InlineVoice:
    ref_audio: bytes
    ref_text: Optional[str]  # None: x-vector-only clone
    name: Optional[str] = None


@dataclass
class SpeechRequest:
    input: str
    voice: Union[str, InlineVoice, None]
    instruct: Optional[str] = None
    response_format: str = "wav"
    stream: bool = True
    stream_format: str = "audio"
    sampling: Sampling = field(default_factory=Sampling)
    seed: Optional[int] = None
    model: Optional[str] = None
    language: Optional[str] = None
    sampling_overrides: dict = field(default_factory=dict)  # only the fields sent; applied over the model's defaults


def _b64(s: str, what: str) -> bytes:
    if not isinstance(s, str) or not s:
        raise RequestError(f"{what} must be a base64 string")
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise RequestError(f"{what} is not valid base64")


def parse_voice(v, require_text: bool = True) -> Union[str, InlineVoice]:
    if isinstance(v, str):
        if not v.strip():
            raise RequestError("voice must not be empty")
        return v.strip()
    if isinstance(v, dict):
        extra = set(v) - {"ref_audio", "ref_text", "name"}
        if extra:
            raise RequestError(f"unknown voice fields: {sorted(extra)}")
        text = v.get("ref_text")
        if text is None and not require_text:
            pass
        elif not isinstance(text, str) or not text.strip():
            raise RequestError("voice.ref_text is required for an inline voice (ICL clone)")
        else:
            text = text.strip()
        name = v.get("name")
        if name is not None and not isinstance(name, str):
            raise RequestError("voice.name must be a string")
        return InlineVoice(_b64(v.get("ref_audio"), "voice.ref_audio"), text, name)
    raise RequestError("voice must be a voice id or an object {ref_audio, ref_text}")


def parse_speech_request(body: dict, require_voice: bool = True, xvec_ok: bool = False) -> SpeechRequest:
    """require_voice=False for VoiceDesign models; xvec_ok lets an inline voice omit ref_text (x-vector-only clone)."""
    if not isinstance(body, dict):
        raise RequestError("body must be a JSON object")
    known = {"model", "input", "voice", "instructions", "instruct", "response_format", "speed", "stream_format", "stream",
             "sampling", "seed", "language"}
    extra = set(body) - known
    if extra:
        raise RequestError(f"unknown fields: {sorted(extra)}")
    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        raise RequestError("input must be a non-empty string")
    if len(text) > MAX_INPUT_CHARS:
        raise RequestError(f"input longer than {MAX_INPUT_CHARS} characters")
    if require_voice and body.get("voice") is None:
        raise RequestError("voice is required")
    language = body.get("language")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise RequestError("language must be a non-empty string")
    ins_a, ins_b = body.get("instructions"), body.get("instruct")
    for x in (ins_a, ins_b):
        if x is not None and not isinstance(x, str):
            raise RequestError("instructions must be a string")
    if ins_a and ins_b and ins_a != ins_b:
        raise RequestError("instructions and instruct differ; send one")
    instruct = (ins_a or ins_b or None)
    instruct = instruct.strip() if instruct and instruct.strip() else None
    fmt = body.get("response_format", "wav")
    if fmt in UNSUPPORTED:
        raise RequestError(f"response_format {fmt!r} not supported; use wav, pcm or flac")
    if fmt not in FORMATS:
        raise RequestError(f"unknown response_format {fmt!r}")
    speed = body.get("speed", 1.0)
    if not isinstance(speed, (int, float)) or abs(float(speed) - 1.0) > 1e-6:
        raise RequestError("speed must be 1.0 (pace is controlled with instructions)")
    sf = body.get("stream_format", "audio")
    if sf not in ("audio", "sse"):
        raise RequestError("stream_format must be 'audio' or 'sse'")
    stream = body.get("stream", True)
    if not isinstance(stream, bool):
        raise RequestError("stream must be a boolean")
    if sf == "sse":
        stream = True
    elif fmt == "flac" and stream:
        stream = False  # flac is only produced whole
    samp = body.get("sampling") or {}
    if not isinstance(samp, dict):
        raise RequestError("sampling must be an object")
    try:
        sampling = Sampling.from_dict(samp)
    except (ValueError, TypeError) as e:
        raise RequestError(str(e))
    _check_sampling(sampling)
    seed = body.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise RequestError("seed must be a non-negative integer")
    voice = body.get("voice")
    voice = parse_voice(voice, require_text=not xvec_ok) if voice is not None else None
    return SpeechRequest(text, voice, instruct, fmt, stream, sf, sampling, seed, body.get("model"), language, dict(samp))


def _check_sampling(s: Sampling):
    for name in ("temperature", "subtalker_temperature"):
        v = getattr(s, name)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 5:
            raise RequestError(f"sampling.{name} must be in [0, 5]")
    for name in ("top_p", "subtalker_top_p"):
        v = getattr(s, name)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 < v <= 1:
            raise RequestError(f"sampling.{name} must be in (0, 1]")
    for name in ("top_k", "subtalker_top_k"):
        v = getattr(s, name)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise RequestError(f"sampling.{name} must be an integer >= 0")
    v = s.repetition_penalty
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not 1.0 <= v <= 3.0:
        raise RequestError("sampling.repetition_penalty must be in [1, 3]")
    if s.max_new_tokens is not None and (not isinstance(s.max_new_tokens, int) or not 1 <= s.max_new_tokens <= 8192):
        raise RequestError("sampling.max_new_tokens must be in [1, 8192]")


def parse_voice_upload(body: dict) -> InlineVoice:
    if not isinstance(body, dict) or not isinstance(body.get("name"), str) or not body["name"].strip():
        raise RequestError("name is required")
    return parse_voice({k: body.get(k) for k in ("ref_audio", "ref_text", "name") if k in body}, require_text=False)


# ---------------------------------------------------------------------------------------------------- app
def make_inline_voice(tts, v: InlineVoice):
    try:
        wav = load_audio(v.ref_audio)
    except Exception as e:
        raise RequestError(f"could not decode ref_audio: {e}")
    return tts.make_voice((wav, SAMPLE_RATE), v.ref_text, name=v.name)


def resolve_target(tts, v) -> dict:
    """Map the request's `voice` onto the loaded model: {"voice": Voice} (Base), {"speaker": name} (CustomVoice) or {}
    (VoiceDesign)."""
    spec = tts.spec
    if spec.variant == "base":
        if v is None:
            raise RequestError("voice is required: a cached voice id or an inline {ref_audio, ref_text}")
        if isinstance(v, InlineVoice):
            return {"voice": make_inline_voice(tts, v)}
        try:
            return {"voice": tts.voice(v)}
        except KeyError:
            raise RequestError(f"unknown voice {v!r}; register it with POST /v1/voices")
    if isinstance(v, InlineVoice):
        raise RequestError(f"{spec.name} cannot clone a voice; "
                           + ("use a preset speaker name" if spec.variant == "custom_voice" else "describe it in instructions"))
    if spec.variant == "custom_voice":
        if v is None or v.lower() not in spec.speakers:
            raise RequestError(f"voice must be one of the preset speakers {list(spec.speakers)}, got {v!r}")
        return {"speaker": v.lower()}
    return {}


def check_request(tts, req: SpeechRequest, target: dict):
    from .api import ModeError
    try:
        tts.resolve_mode(target.get("voice"), target.get("speaker"), req.instruct, req.language)
    except ModeError as e:
        raise RequestError(str(e))


def synth_kwargs(tts, req: SpeechRequest, target: dict) -> dict:
    return dict(instruct=req.instruct, sampling=tts.spec.sampling.with_(**req.sampling_overrides), seed=req.seed,
                language=req.language, **target)


def voices_listing(tts) -> dict:
    spec = tts.spec
    if spec.variant == "base":
        return {"kind": "clone", "voices": tts.voices.list(), "speakers": []}
    if spec.variant == "custom_voice":
        return {"kind": "speaker", "voices": list(spec.speakers), "speakers": list(spec.speakers)}
    return {"kind": "none", "voices": [], "speakers": [], "note": "VoiceDesign: describe the voice in instructions"}


def audio_iter(tts, req: SpeechRequest, target: dict):
    infos: list = []  # per request: concurrent backends run several streams at once
    pcm_iter = tts.stream(req.input, infos=infos, **synth_kwargs(tts, req, target))
    if req.stream_format == "sse":
        n = 0
        for pcm in pcm_iter:
            n += len(pcm)
            yield ("data: " + json.dumps({"type": "speech.audio.delta", "audio": base64.b64encode(to_pcm16(pcm)).decode()}) + "\n\n").encode()
        infos = [i.__dict__ for i in infos]
        done = {"type": "speech.audio.done", "samples": n, "sample_rate": SAMPLE_RATE,
                "segments": [{k: v for k, v in i.items() if k != "extra"} for i in infos]}
        yield ("data: " + json.dumps(done) + "\n\n").encode()
        return
    if req.response_format == "wav":
        yield wav_header(SAMPLE_RATE)
    for pcm in pcm_iter:
        yield to_pcm16(pcm)


def create_app(tts, model_name: str = "lure"):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="Lure TTS", version="0.1.0")

    def err(msg, code=400):
        return JSONResponse({"error": {"message": msg, "type": "invalid_request_error"}}, status_code=code)

    async def body_json(request: Request):
        try:
            return await request.json()
        except Exception:
            raise RequestError("body is not valid JSON")

    @app.get("/health")
    def health():
        return {"ok": True, "backend": tts.backend_name}

    @app.get("/v1/models")
    def models():
        spec = tts.spec.to_dict()
        spec.pop("model_dir", None)
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "lure",
                                            "backend": tts.backend_name, **spec}]}

    @app.get("/v1/voices")
    def voices():
        return voices_listing(tts)

    @app.post("/v1/voices")
    async def add_voice(request: Request):
        try:
            if not tts.spec.is_clone:
                raise RequestError(f"{tts.spec.name} is a {tts.spec.variant} model and cannot clone voices")
            v = parse_voice_upload(await body_json(request))
            voice = await run_in_threadpool(make_inline_voice, tts, v)
        except RequestError as e:
            return err(str(e))
        return {"voice": voice.voice_id, "n_ref_frames": voice.n_ref_frames}

    @app.post("/v1/audio/speech")
    async def speech(request: Request):
        try:
            req = parse_speech_request(await body_json(request), require_voice=tts.spec.variant != "voice_design",
                                       xvec_ok=tts.spec.is_clone)
            target = await run_in_threadpool(resolve_target, tts, req.voice)
            check_request(tts, req, target)
        except RequestError as e:
            return err(str(e))
        media = "text/event-stream" if req.stream_format == "sse" else FORMATS[req.response_format]
        if req.stream:
            return StreamingResponse(audio_iter(tts, req, target), media_type=media)
        # blocking work off the event loop; StreamingResponse already iterates sync generators in the threadpool
        audio = await run_in_threadpool(lambda: tts.speak(req.input, **synth_kwargs(tts, req, target)))
        if req.response_format == "wav":
            data = wav_bytes(audio)
        elif req.response_format == "pcm":
            data = to_pcm16(audio)
        else:
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, audio, SAMPLE_RATE, format="FLAC")
            data = buf.getvalue()
        return Response(data, media_type=FORMATS[req.response_format])

    return app


def serve(tts, host: str = "127.0.0.1", port: int = 8880):
    import uvicorn
    uvicorn.run(create_app(tts), host=host, port=port, log_level="info")
