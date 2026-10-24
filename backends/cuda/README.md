# lure-cuda

Lure backend for NVIDIA GPUs. It runs every Qwen3-TTS 12Hz checkpoint and the Lure family with CUDA graphs,
continuous batching of up to 8 concurrent requests (any mix of modes and layouts), streaming codec decode and the
optional Turbo head. The core loads it as backend `cuda`.

| Checkpoint | Modes | Status |
|---|---|---|
| Lure, Lure Turbo (0.6B) | `icl`, non-streaming layout | Verified with the conformance suite on L4 |
| Lure, Lure Turbo | `xvec`; `icl` in the streaming layout | Supported (tested on CPU fixtures) |
| Qwen3-TTS-12Hz-0.6B-Base, 1.7B-Base | `icl`, `xvec` | Supported (tested on CPU fixtures; real-weights reports welcome via GitHub issues) |
| Qwen3-TTS-12Hz-0.6B-CustomVoice, 1.7B-CustomVoice | `custom_voice` | Supported (tested on CPU fixtures; real-weights reports welcome via GitHub issues) |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | `voice_design` | Supported (tested on CPU fixtures; real-weights reports welcome via GitHub issues) |

Every mode runs in both the `non_streaming` and `streaming` prompt layouts. Only `icl` decodes with the reference
codes; the other modes decode the generated codes alone. The Turbo head is 0.6B only. Windows is implemented but
untested. The backend contract is in [docs/architecture.md](../../docs/architecture.md#backend-contract).

Full documentation: [docs/backends/cuda.md](../../docs/backends/cuda.md). General setup and the Python API:
[docs/getting-started.md](../../docs/getting-started.md), [docs/api.md](../../docs/api.md).

The CUDA graph code is derived from [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) by Andres
Marafioti (MIT). Its license is kept in [LICENSE-faster-qwen3-tts](LICENSE-faster-qwen3-tts).

## Install (Linux)

Python 3.10 or newer and an NVIDIA GPU. Ampere or newer is recommended, since the backend runs in bf16 (see the T4
numbers below). From the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps qwen-tts-hf==0.1.1.post1
pip install ./backends/cuda ".[server]"
lure backends          # should list cuda
```

Install qwen-tts-hf with `--no-deps`: its declared dependencies (gradio, sox, onnxruntime, torchaudio) are only used by
the 25 Hz tokenizer, which Lure does not use. The backend stubs them during import when they are missing. If
torchaudio is installed, it must match your torch CUDA build.

Windows is untested; see the Windows section in the docs.

## Usage

```bash
lure say "Hello there. (laugh) Really?" --backend cuda --model /path/to/lure \
    --ref-wav ref.wav --ref-text "Transcript of ref.wav." -o out.wav --opt max_batch=4
lure serve --backend cuda --model /path/to/lure
lure say "Hello there." --backend cuda --model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice --speaker vivian -o out.wav
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `max_batch` | 8 | Concurrent segments decoded in one graph step. |
| `max_seq_len` | 4096 | KV slots per batch row (prefill plus frames). |
| `use_graphs` | true | `false` runs the same step eagerly, for debugging. |
| `decode_context_frames` | 72 | Left context for each streamed decode window. |
| `turbo` | auto | Turbo head if the model folder has `turbo.json`; `1` requires it, `0` disables it. |
| `turbo_head` | | Path to a different head file. |
| `dtype` | bfloat16 | Talker, code predictor and speaker encoder weights. |
| `codec_dtype` | float32 | Speech tokenizer weights. |

## Measured (Lure, streaming, graphs on, B = 8 graphs)

12.5 frames/s is real time. Full tables, including Turbo, are in the docs.

| GPU | TTFA median | B1 frames/s | B8 aggregate frames/s |
|---|---|---|---|
| H100 | 0.30 s | 41.5 | 171 |
| L40S | 0.28 s | 41.7 | 142 |
| A10G | 0.42 s | 27.4 | 92 |
| L4 | 0.49 s | 24.0 | 83 |
| T4 | 2.66 s | 4.3 | 23 |

Turbo on L4 (same container, 72-frame decode context): TTFA 0.36 s vs 0.54 s, B1 24.6 vs 18.0 frames/s.

## Known limitations

- T4 has no bf16 tensor cores and runs slower than real time. There is no fp16 path.
- Seeded output is batch-invariant on L40S, A10G, L4 and T4, but not on H100, where bf16 kernel choice depends on
  the batch size.
- The stock Qwen3-TTS checkpoints and the 1.7B shapes are tested on tiny CPU fixtures against qwen-tts, not yet with
  real weights on a GPU.
- Greedy output matches qwen-tts eager only for a short prefix of frames. This is bf16 drift; qwen-tts eager itself
  differs across GPUs.
- No weight quantization.

## Tests

```bash
pip install pytest
pytest backends/cuda/tests
```

The CPU tests (60) use tiny random models, one per stock config (0.6B and 1.7B Base, 0.6B and 1.7B CustomVoice,
1.7B VoiceDesign), and need no GPU or weights. They check the prompt rows of every mode and layout against the qwen-tts
dumps in `conformance/reference`, greedy codes against qwen-tts 0.1.1 `generate()` for every mode and layout, mixed-mode
batches of 8, KV rebases and buckets, and the decode rules. The prompt checks read config and tokenizer files from
`LURE_MODEL_DIR` when set, else from `conformance/reference/tokenizer`. The GPU scripts in `tests/gpu/`
(`e2e_gpu.py`, `turbo_gpu.py`, `ttfa_gpu.py`, `modes_gpu.py`) need CUDA and real weights; run them directly with
`--help` for options.
