# Getting started

## 1. Get the model

Download Lure from [smirki/lure-tts-0.6b-preview](https://huggingface.co/smirki/lure-tts-0.6b-preview) or Lure Turbo
from [smirki/lure-turbo-tts-0.6b-preview](https://huggingface.co/smirki/lure-turbo-tts-0.6b-preview) (both 0.6B):

```bash
hf download smirki/lure-tts-0.6b-preview --local-dir models/lure
```

Lure TTS loads Qwen3-TTS 12Hz checkpoints in their Hugging Face layout: 0.6B and 1.7B Base, CustomVoice and
VoiceDesign, plus the Lure family (Lure, Lure Turbo). A Lure folder looks like this:

```
lure/
  config.json  generation_config.json  preprocessor_config.json
  model.safetensors                      talker, code predictor, speaker encoder (bf16)
  vocab.json  merges.txt  tokenizer_config.json
  speech_tokenizer/                      12.5 Hz codec: config.json, model.safetensors, ...
  lure.json                              marks the Lure family (layout, inline tags)
  turbo.json  turbo_head.safetensors     Turbo folders only, see turbo.md
```

Stock Qwen3-TTS folders are used as downloaded, with no `lure.json`. Check what a folder is with `lure info
/path/to/model`, which reads only the JSON files.

Point Lure TTS at a folder with `Lure.load("/path/to/lure")`, `--model /path/to/lure`, or `export LURE_MODEL=/path/to/lure`.

## 2. Install

Use Python 3.10 or newer and a virtual environment. Run everything from the repository root.

| Platform | Install |
|---|---|
| Mac, Apple silicon | `pip install ./backends/mlx ".[server]"` |
| Linux + NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cu128`, then `pip install -r backends/cuda/requirements.txt`, `pip install --no-deps qwen-tts-hf==0.1.1.post1`, `pip install ".[server]"` and `pip install --no-deps ./backends/cuda` |
| Any (C++ library) | build `backends/cpp` with a CMake preset, then `pip install . ./backends/cpp/python` and set `LURE_CPP_LIB` to the built library |
| Core only (tests, fake backend) | `pip install -e ".[dev]"` |

The core package is `lure-tts` (import name `lure_tts`, command `lure`). Extras: `server` (FastAPI and uvicorn),
`play` (live playback with `--play`), `dev` (pytest). Each backend is a separate package that registers itself with the
core. Backend-specific steps and options:
[cuda](backends/cuda.md), [mlx](backends/mlx.md), [cpp](backends/cpp.md).

Check what got installed:

```bash
lure backends
```

## 3. First audio

Voice cloning needs a reference clip (a few seconds of clean speech; any sample rate, resampled to 24 kHz) and its exact
transcript.

```bash
lure say "Hello there. (laugh) Okay." --ref-wav ref.wav --ref-text "Transcript of ref.wav." --save-voice me -o hello.wav
lure say "I sound the same, and nothing was re-encoded." --voice me -o again.wav --stats
```

On a CustomVoice checkpoint pass a preset speaker instead of a clip, and on VoiceDesign describe the voice:

```bash
lure say "Good morning." --model /models/Qwen3-TTS-12Hz-1.7B-CustomVoice --speaker vivian -o cv.wav
lure say "Good morning." --model /models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --instruct "A bright young woman, fast." -o vd.wav
```

`--save-voice me` stores the encoded voice under `~/.cache/lure/voices/me` (change the location with `--cache` or
`LURE_CACHE`). `--stats` prints frames, EOS and time to first audio per segment.

The same in Python:

```python
from lure_tts import Lure

with Lure.load("/path/to/lure") as tts:
    voice = tts.make_voice("ref.wav", "Transcript of ref.wav.", name="me")
    for pcm in tts.stream("Hello there. This streams.", voice):
        ...  # float32 mono, 24 kHz, about 0.32 s per chunk
```

## 4. Serve it

```bash
pip install ".[server]"     # if you skipped the extra
lure serve --port 8880
curl -s http://127.0.0.1:8880/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model": "lure", "voice": "me", "input": "Hello from the server."}' -o server.wav
```

Any OpenAI client works against `http://127.0.0.1:8880/v1`; see [api.md](api.md#server).

## Next

- [prompting.md](prompting.md): directions, speakers, inline tags, long text
- [api.md](api.md): every API call, CLI flag and endpoint
- [../examples/](../examples/): runnable snippets
