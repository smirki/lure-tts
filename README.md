# Lure TTS

Lure TTS runs Qwen3-TTS 12Hz checkpoints and the Lure family of expressive voices on a Mac, on an NVIDIA GPU, or on
any CPU, with one Python API, one CLI and one OpenAI-compatible server on top. Every backend gets the same prompt token
for token, the same text segmentation and the same sampling defaults, and a shared conformance suite checks the output
against the qwen-tts reference implementation.

```python
from lure_tts import Lure

tts = Lure.load("/path/to/lure")                       # picks mlx, cuda or cpp for this machine
voice = tts.make_voice("ref.wav", "Transcript of ref.wav.", name="willow")
tts.speak("(sigh) I guess we're doing this again. (chuckle) Fine. I'm in.", voice,
          instruct="Tired and resigned, then a reluctant smile.", out="out.wav")
```

## Supported models

| Model | Modes | Default layout | Inline tags | Instruct |
|---|---|---|---|---|
| Qwen3-TTS-12Hz-0.6B-Base, 1.7B-Base | voice clone (`icl`), x-vector clone (`xvec`) | streaming | no | no |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | preset speaker (`custom_voice`) | non-streaming | no | no |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | preset speaker (`custom_voice`) | non-streaming | no | optional |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | voice from a description (`voice_design`) | non-streaming | no | required |
| Lure, Lure Turbo | voice clone (`icl`), x-vector clone (`xvec`) | non-streaming | yes | optional |

Qwen3-TTS has no 0.6B VoiceDesign checkpoint. The stock defaults match qwen-tts 0.1.1. The Lure family folders carry
a `lure.json` file that selects their layout and turns on inline tags. `lure info MODEL` prints what a folder
supports, and [docs/architecture.md](docs/architecture.md) has the exact prompt for every mode.

```python
tts = Lure.load("/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
tts.speak("Good morning.", speaker="vivian", instruct="Warm and unhurried.", out="cv.wav")

tts = Lure.load("/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
tts.speak("Good morning.", instruct="A gravelly older man, amused.", out="vd.wav")
```

### Backend status

Every backend runs every model and mode in the table above, in both layouts and both sizes.

| | mlx | cuda | cpp |
|---|---|---|---|
| Lure, Lure Turbo (`icl`, 0.6B) | verified with real weights (conformance suite below) | verified with real weights | verified with real weights |
| Stock Qwen3-TTS, all variants and modes, 0.6B and 1.7B | supported, tested on fixtures | supported, tested on fixtures | supported, tested with prompt dumps and fixtures |

"Tested on fixtures" means each mode and layout produces the same prompt, and the same greedy codes, as qwen-tts 0.1.1
on small random-weight checkpoints with the real configs. The stock checkpoints have not been run with real weights in
this repository yet; reports are welcome as GitHub issues. Turbo heads apply to 0.6B checkpoints only.

## Highlights

- Audio streams in chunks of about 0.32 s while the rest is still being generated. Long text is cut into sentences and
  streamed one segment at a time.
- With the Lure family, inline tags such as `(laugh)`, `(sigh)` or `(gasp)` are performed and never read aloud.
  Across every backend and every conformance take, the tag-read check found 0 tags spoken as words. On stock
  checkpoints parentheses are plain text.
- A direction (instruct), a free-text note such as `"Frantic panic, breathless."`, sets the delivery for a request.
- Voices are cloned from a short clip plus its transcript (or from the clip alone), encoded once and cached on disk by
  name.
- When the model folder ships `turbo.json`, the Turbo head replaces the 15 sequential code-predictor passes per frame
  with 3. On an L4 the code predictor step gets 3.4x faster (8-stream batch).
- The cuda backend batches up to 8 concurrent requests in one CUDA-graph decode loop.
- The built-in server speaks the OpenAI `POST /v1/audio/speech` API with streaming WAV, PCM or SSE, so existing OpenAI
  clients work unchanged.

## Quickstart

Download a model folder from Hugging Face:

| Model | Size | Link |
|---|---|---|
| Lure | 0.6B | [smirki/lure-tts-0.6b-preview](https://huggingface.co/smirki/lure-tts-0.6b-preview) |
| Lure Turbo | 0.6B | [smirki/lure-turbo-tts-0.6b-preview](https://huggingface.co/smirki/lure-turbo-tts-0.6b-preview) |

```bash
git clone https://github.com/smirki/lure-tts && cd lure-tts
hf download smirki/lure-tts-0.6b-preview --local-dir models/lure
```

Point `--model` or `LURE_MODEL` at the downloaded folder. Stock Qwen3-TTS 12Hz checkpoints work the same way.

All commands run from the repository root, in a fresh virtual environment (`python3 -m venv .venv && source .venv/bin/activate`).

### Mac (Apple silicon, mlx)

```bash
pip install ./backends/mlx ".[server]"
export LURE_MODEL=/path/to/lure
lure say "Hello there. (laugh) Okay." --ref-wav ref.wav --ref-text "Transcript of ref.wav." --save-voice me -o hello.wav
lure say "Same voice, no re-encoding." --voice me -o again.wav
```

### NVIDIA GPU (Linux, cuda)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r backends/cuda/requirements.txt
pip install --no-deps qwen-tts-hf==0.1.1.post1
pip install ".[server]" && pip install --no-deps ./backends/cuda
export LURE_MODEL=/path/to/lure
lure serve --port 8880          # OpenAI-compatible server, batches concurrent requests
```

The `--no-deps` steps skip packages the backend does not need. Details: [docs/backends/cuda.md](docs/backends/cuda.md).

### C++ binary (CPU, Metal, CUDA, Vulkan)

```bash
cd backends/cpp
cmake --preset macos-metal -G Ninja        # or linux-cpu, linux-cuda, linux-vulkan, windows-cpu, windows-cuda, windows-vulkan
cmake --build build/macos-metal -j4 --target lure_tts lure_cli
build/macos-metal/bin/lure_cli generate --model /path/to/lure \
    --ref-wav ref.wav --ref-text "Transcript of ref.wav." --text "Hello from the single binary." --out hello.wav
```

To drive the library from Python instead, run `pip install . ./backends/cpp/python` from the repository root, set
`LURE_CPP_LIB` to the built `liblure_tts` library and use `backend="cpp"`. Details:
[docs/backends/cpp.md](docs/backends/cpp.md).

More in [docs/getting-started.md](docs/getting-started.md), and runnable snippets in [examples/](examples/).

## Choosing a backend

| Your machine | Backend | Notes |
|---|---|---|
| Mac with Apple silicon | `mlx` | Picked automatically on Apple silicon. RTF 0.79 on an M5 with 16 GB. |
| Linux with an NVIDIA GPU (Ampere or newer) | `cuda` | The only backend that batches concurrent requests (up to 8 per CUDA graph). |
| NVIDIA GPU without the Python stack, or a single binary to ship | `cpp` (CUDA build) | Lowest time to first audio through the API: 0.17 s on an L4 (0.139 s with Turbo). |
| AMD / Intel GPU | `cpp` (Vulkan build) | Builds from a preset; not measured yet. |
| CPU only | `cpp` (CPU build) | RTF 6.6 on 16 vCPUs, so offline jobs only. |
| Windows | `cuda` or `cpp` | Presets and install notes exist, but nothing has been tested on Windows yet. |

`backend="auto"` (the default) picks mlx on Apple silicon, then cuda if PyTorch sees a GPU, else cpp. Override with
`backend=`, `--backend` or `LURE_BACKEND`. `lure backends` shows what is installed.

## Performance

RTF is wall time divided by audio duration (below 1.0 is faster than real time). TTFA is time to first audio. Sources and
more tables: [docs/performance.md](docs/performance.md). The performance and quality numbers in this README were
measured with an earlier reference clip; the clip now bundled in `conformance/assets/` is a synthetic voice.

cuda backend measured below the core API (no core segmenting), Lure (not Turbo), CUDA graphs on, warm runs:

| GPU | TTFA median | 1 stream (frames/s) | 8 streams (aggregate frames/s) | RTF, short lines |
|---|---|---|---|---|
| H100 | 0.302 s | 41.5 | 171.0 | 0.33 |
| L40S | 0.282 s | 41.7 | 141.9 | 0.31 |
| A10G | 0.415 s | 27.4 | 92.3 | 0.45 |
| L4 | 0.486 s | 24.0 | 83.4 | 0.50 |
| T4 | 2.66 s | 4.3 | 22.7 | 2.85 |

12.5 frames/s is one real-time stream, so 8 streams on an H100 run at about 13.7x real time in total. These runs used the
older 25-frame streaming decode context; the default is now 72.

H100, warm, current defaults (decode context 72), one request at a time. `first chunk` is how many frames the first
audio chunk waits for (`first_chunk_frames`, default 4):

| Model | Batch slots | TTFA, first chunk 4 | TTFA, first chunk 1 | ms per frame after the first | RTF |
|---|---|---|---|---|---|
| Lure | 1 | 0.283 s | 0.229 s | 18.4 | 0.23 |
| Lure | 8 | 0.331 s | 0.266 s | 19.6 | 0.25 |
| Turbo | 1 | 0.168 s | 0.149 s | 11.4 | 0.14 |
| Turbo | 8 | 0.200 s | 0.167 s | 12.4 | 0.15 |

With the shorter 25-frame decode context and a 1-frame first chunk, Turbo reaches 0.135 s median (0.121 s best).

Every backend through the public API (the conformance runner: segmentation, voice prompt and chunked decode included):

| Backend | Hardware | Model | RTF | TTFA median |
|---|---|---|---|---|
| cuda | L4 | Lure | 0.66 | 1.06 s |
| cuda | L4 | Turbo | 0.48 | 0.64 s |
| cpp CUDA | L4 | Lure | 0.51 | 0.175 s |
| cpp CUDA | L4 | Turbo | 0.40 | 0.139 s |
| mlx | Apple M5, 16 GB | Lure | 0.79 | 0.83 s |
| cpp Metal | Apple M5, 16 GB | Lure | 1.05 | 0.36 s |
| cpp CPU | 16 vCPU | Lure | 6.64 | 2.36 s |

The cuda and cpp CUDA rows come from the Lure-versus-Turbo comparison on L4; the mlx, cpp Metal and cpp CPU rows show
RTF on short lines. Turbo has not been measured on a Mac yet. Windows is untested.

## Quality parity

Each backend renders the same 13 cases twice (short lines, tagged lines, a 97 s passage, and lines with and without a
direction), cloning the same voice prompt with the same seeds. Scoring uses Qwen3-ASR-1.7B for word error rate and
ECAPA speaker similarity against the voice prompt. The reference is the qwen-tts implementation of the architecture.
Gates: WER at most 0.045 (reference + 0.03), ECAPA at least 0.545 (reference - 0.03), no tags read, every segment ends on
EOS.

| Backend (streaming) | Model | WER | ECAPA | Tags read | EOS rate | Gates |
|---|---|---|---|---|---|---|
| reference (qwen-tts, L4) | Lure | 0.0150 | 0.575 | 0 | 1.0 | reference |
| cuda (L4) | Lure | 0.0223 | 0.582 | 0 | 1.0 | pass |
| cuda (L4) | Turbo | 0.0130 | 0.570 | 0 | 1.0 | pass |
| cpp CUDA (L4) | Lure | 0.0150 | 0.562 | 0 | 1.0 | pass |
| cpp CUDA (L4) | Turbo | 0.0001 | 0.566 | 0 | 1.0 | pass |
| cpp CPU (16 vCPU) | Lure, older build | 0.0098 | 0.544 | 0 | 1.0 | ECAPA 0.001 under the gate |
| cpp CPU (16 vCPU) | Turbo | 0.0141 | 0.579 | 0 | 1.0 | pass |
| mlx (Apple M5) | Lure | 0.0073 | 0.533 | 0 | 1.0 | ECAPA fails |
| cpp Metal (Apple M5) | Lure | 0.0112 | 0.527 | 0 | 1.0 | ECAPA fails |

The cpp CPU Lure run used a build from before the streaming decoder fix; the same fix raised cpp CUDA from 0.546 to
0.562. How the suite works: [docs/conformance.md](docs/conformance.md).

## Known limitations

- On both Mac backends, speaker similarity sits about 0.04 to 0.05 below the reference (mlx 0.533, cpp Metal 0.527 vs
  0.575) while WER is fine and the cloned speaker embedding matches exactly. The cause is not found yet; Metal decode
  precision or sampling numerics are the suspects.
- cpp Metal streams slower than real time on an M5 with 16 GB (RTF 1.05 on short lines). Use mlx on a Mac.
- T4 GPUs have no bf16 tensor cores, and the cuda backend runs at 4.3 frames/s there, slower than real time.
- The cpp CPU build runs at RTF 6.6 on 16 vCPUs, which suits offline jobs. Its seeded output also differs between
  machines with different CPUs.
- Directions change the audio on every backend, but with 2 takes per case the measured effect (duration, pitch) did not
  point the same way on every backend. More takes are needed before claiming direction parity.
- Seeded output with 8 concurrent requests differs from a single request on H100 (bf16 kernel choice depends on batch
  size). It matches on L40S, A10G, L4 and T4.
- Windows has build presets and install notes but has not been run.

## How it fits together

```
            your code / lure CLI / OpenAI-compatible server
                                |
  core (Python package `lure_tts`): model inspection, text segmenter, prompts,
                                      sampling defaults, voice cache, streaming scheduler
                                |
                 one segment + prompt ids + cached voice at a time
          +---------------------+----------------------+
          |                     |                      |
     backends/mlx          backends/cuda          backends/cpp
     MLX, Apple silicon    PyTorch + CUDA graphs  ggml: CPU, Metal, CUDA, Vulkan
          |                     |                      |   (C API, CLI, own HTTP server)
          +---------- float32 PCM chunks, 24 kHz ------+
```

The core owns everything that decides what the model hears, and backends only turn a prompt into audio. Backends can
then differ only in numerics, which the prompt check and the conformance gates measure. See
[docs/architecture.md](docs/architecture.md).

## Documentation

- [Getting started](docs/getting-started.md): install, model folder, first audio
- [API](docs/api.md): Python API, CLI, server endpoints
- [Prompting](docs/prompting.md): directions, speakers, inline tags, long text, sampling defaults
- [Turbo](docs/turbo.md): the grouped code-predictor head
- [Architecture](docs/architecture.md): supported checkpoints, the prompt contract and the backend contract
- [Conformance](docs/conformance.md): the cross-backend test suite
- [Performance](docs/performance.md): every measured number, with hardware
- [FAQ](docs/faq.md)
- Backends: [cuda](docs/backends/cuda.md), [mlx](docs/backends/mlx.md), [cpp](docs/backends/cpp.md)

## License

Apache-2.0, see [LICENSE](LICENSE). The backends build on faster-qwen3-tts (MIT), mlx-audio (MIT) and audio.cpp
(Apache-2.0); [NOTICE](NOTICE) lists these and the third-party code they include. The model weights are distributed
separately under their own terms.
