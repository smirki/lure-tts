# cuda backend (NVIDIA GPUs)

The cuda backend runs Qwen3-TTS 12Hz checkpoints, including the Lure family, on NVIDIA GPUs with PyTorch. It lives in `backends/cuda` as the `lure-cuda` package
(import `lure_cuda`) and registers itself as backend `cuda`. For general setup and the Python API, see
[getting-started.md](../getting-started.md) and [api.md](../api.md).

## Supported checkpoints and modes

| Checkpoint | Modes | Layouts | Status |
|---|---|---|---|
| Lure, Lure Turbo (0.6B) | `icl` | `non_streaming` | Verified with the conformance suite on L4 |
| Lure, Lure Turbo | `xvec`, and `icl` in the `streaming` layout | both | Supported (tested on CPU fixtures) |
| Qwen3-TTS-12Hz-0.6B-Base, 1.7B-Base | `icl`, `xvec` | both | Supported (tested on CPU fixtures; real-weights reports welcome via GitHub issues) |
| Qwen3-TTS-12Hz-0.6B-CustomVoice, 1.7B-CustomVoice | `custom_voice` | both | Supported (tested on CPU fixtures; real-weights reports welcome via GitHub issues) |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | `voice_design` | both | Supported (tested on CPU fixtures; real-weights reports welcome via GitHub issues) |

The core picks the mode and layout ([architecture.md](../architecture.md#choosing-the-mode)); the backend follows the
contract in [architecture.md](../architecture.md#backend-contract). The Turbo head exists only for the 0.6B Lure
family. Windows is implemented but untested.

The CUDA graph code (static KV cache, side-stream warmup and capture) is derived from
[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) by Andres Marafioti (MIT). The model, processor,
speech tokenizer and prompt builder come from qwen-tts-hf 0.1.1.post1.

## How it works

- Load: `load` takes the core's `spec=` and checks the weights against it (variant, talker and code predictor hidden
  sizes, speaker encoder size). Any variant and size loads; only Base checkpoints have a speaker encoder, so
  `encode_voice` works on Base only. An empty reference transcript gives an x-vector-only voice for `xvec`.
- Prompt: the backend uses the core's prompt ids, or builds the same ids with the qwen-tts processor when called
  directly. The prefill is a batch-1 copy of the qwen-tts `generate()` prompt code for every mode and layout, with the
  same op grouping so bf16 numerics match upstream. The speaker row is the x-vector in `icl` and `xvec`, the preset
  speaker's codec row in `custom_voice` (dialect speakers switch the language row as in qwen-tts), and absent in
  `voice_design`. It is checked row by row against all of `conformance/reference/prompt_ids.json` and
  `prompt_modes.json`, and on every core request its row and trailing counts are checked against the core's
  `build_prompt`. `dump_layout` runs the same builder on a symbolic stub, so the conformance prompt check covers the
  real code path. Voice prompts are loaded with librosa at 24 kHz mono, with no appended silence.
- Trailing text: each request carries its trailing text rows. In the streaming layouts generated frame k is fed with
  the text of `trailing[k]`, then `tts_pad`; in the non-streaming layouts it is `tts_pad` throughout.
- Batching: one engine thread steps every active request together. The talker step and the code predictor are each
  captured as a CUDA graph at batch `max_batch`, and requests join or leave between steps. A new request's prefill runs
  eagerly at batch 1 and its KV cache is copied into a free slot of the shared static cache; each row has its own start
  mask and RoPE offset, so positions match a single-request run. The engine does not depend on the mode, so requests
  of different modes and layouts share the batch.
- 1.7B shapes: the talker runs at hidden 2048, and every code predictor input (the talker hidden plus `C(cb0)`, then
  each code embedding, all 2048-d) goes through `small_to_mtp_projection` (2048 to 1024) before the 1024-d code
  predictor, in the eager and the CUDA-graph step alike. The x-vector is 2048-d.
- KV-length buckets: the talker has one graph per cache length (256, 512, 1024, 2048, `max_seq_len`). A step attends
  only to the smallest bucket that covers the live keys. On an L4 this took the step time from 113 ms to 33 ms.
- Sampling: logits are processed in float32 in the order transformers `generate()` uses for qwen-tts (repetition
  penalty, token suppression, EOS blocked below `min_new_tokens`, temperature, top-k, top-p). Each request has its own
  `torch.Generator` and samples with the Gumbel-max trick, so a seeded request gives the same frames alone or inside a
  batch (see the H100 exception below). Sampling parameters are per request.
- Streaming decode: every chunk decodes the window `[reference ; generated][a - ctx : b]` and keeps the samples of
  `[a, b)`. The default context is 72 frames, the codec decoder's sliding window, so in `icl` the first chunk is primed
  with the tail of the reference codes. Non-streaming mode decodes `[reference ; generated]` once and cuts the
  reference share, as qwen-tts does. Only `icl` decodes with the reference: `xvec`, `custom_voice` and
  `voice_design` decode the generated codes alone, with left context from earlier generated frames.
- Turbo: when the model folder has `turbo.json`, the 15 sequential code-predictor passes per frame are replaced by the
  Turbo head: 3 causal passes over codebook groups (1-3, 4-7, 8-15) with the head's own 5 layers, plus a small add-on
  MLP for the later codebooks of each group. The whole fixed-shape frame is one CUDA graph. Talker graphs, batching and
  seeding are unchanged. `turbo.json` `hidden_size` must equal the talker hidden size, and the head needs the code
  predictor hidden to equal the talker hidden, which holds on 0.6B only; anything else is rejected at load.

Weights run in bf16; the speech tokenizer runs in float32 (`codec_dtype`). Sampled audio does not match qwen-tts bit
for bit, because of bf16 weights and the different random stream, but the sampling distribution is the same.

## Install

### Linux

Python 3.10 or newer and an NVIDIA GPU. Ampere or newer is recommended (T4 is slower than real time, see below). From
the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128     # or cu126 to match your driver
pip install --no-deps qwen-tts-hf==0.1.1.post1
pip install ./backends/cuda ".[server]"
lure backends          # should list cuda
```

qwen-tts-hf declares gradio, sox, onnxruntime and torchaudio as dependencies. They are only used by the 25 Hz
tokenizer, which Lure does not use, so install it with `--no-deps`. When those modules are missing, the backend
registers inert stubs while `qwen_tts` imports and removes them afterwards. If torchaudio is installed, it must match
the torch CUDA build or the import fails.

The backend uses SDPA attention and no `torch.compile`, so it needs neither flash-attn nor triton.

### Windows (untested)

Nothing in the backend depends on Linux, but it has not been run on Windows. Expected steps:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps qwen-tts-hf==0.1.1.post1
pip install .\backends\cuda ".[server]"
```

- A plain `pip install torch` on Windows gives a CPU build. Use the CUDA index URL above.
- CUDA graphs work with PyTorch CUDA builds on Windows. Kernel launches cost more under WDDM, so graphs should help
  at least as much as on Linux.
- Hugging Face cache paths can exceed MAX_PATH. Enable long paths or set `HF_HOME` to a short directory.

## Usage

```bash
lure say "Hello there. (laugh) Really?" --backend cuda --model /path/to/lure \
    --ref-wav ref.wav --ref-text "Transcript of ref.wav." -o out.wav
lure serve --backend cuda --model /path/to/lure --opt max_batch=4
```

```python
from lure_tts import Lure

tts = Lure.load("/path/to/lure", backend="cuda", max_batch=4)
```

With `backend="auto"`, the core picks cuda when the package is installed and `torch.cuda.is_available()` is true
(mlx comes first on Apple silicon). The backend sets `concurrent = True`, so the core and its server send overlapping
requests to it from separate threads and they share the batch.

## Options

Pass options as `--opt key=value` on the command line or as keyword arguments to `Lure.load`.

| Option | Default | Meaning |
|---|---|---|
| `max_batch` | 8 | Batch size of the CUDA graphs: how many segments decode together. Every step computes all rows. |
| `max_seq_len` | 4096 | KV slots per batch row, covering prefill plus frames. Requests that cannot fit are rejected. |
| `use_graphs` | true | `false` runs the same step eagerly (debugging, parity checks). |
| `warmup` | true | Capture the graphs at load time. |
| `decode_context_frames` | 72 | Left context of each streamed decode window. 25 is cheaper but gives worse chunk joins (see Performance). |
| `turbo` | auto | `auto`: Turbo head if the model folder has `turbo.json`. `1`: require it. `0`: use the 15-pass code predictor. |
| `turbo_head` | | Path to another Turbo head file; the groups still come from `turbo.json`. |
| `dtype` | bfloat16 | Talker, code predictor and speaker encoder weights and compute. |
| `codec_dtype` | float32 | Speech tokenizer (encoder and decoder) weights. |
| `attn_implementation` | sdpa | Passed to transformers. |
| `device` | cuda | Torch device, for example `cuda:1`. |
| `verify_prompt_ids` | false | Cross-check the core's prompt ids against the qwen-tts processor on every request. |

`quant` accepts only `None` or `bf16`; this backend has no weight quantization.

Memory: the bf16 KV cache costs 114,688 bytes per token per row, so `max_batch=8` with `max_seq_len=4096` is about
3.8 GB, plus about 1.9 GB of weights (codec in float32) and the graph pools. The 1.7B checkpoints have the same KV
size per token (same layer count, KV heads and head dim) and roughly 2 GB more weights. Peak allocated memory measured 6.6 to
6.9 GB at batch 8. On 8 to 12 GB cards, lower `max_batch` or `max_seq_len`.

## Performance

All numbers are from cloud GPUs measured on 2026-10-23. Frames are 80 ms codec frames, so 12.5 frames/s is real time.
TTFA is time to first audio for a streamed short case with a 4-frame first chunk.

### Lure, by GPU

Graphs on, `max_batch=8`, streaming decode with a 25-frame context (the default is now 72, which costs more; see the
Turbo table for L4 numbers at 72). B8 is the aggregate of 8 concurrent requests. "ng" means `use_graphs=false`.

| GPU | TTFA median (p90) | TTFA ng | B1 frames/s | B8 frames/s | ms/step B1 / B8 | B1 frames/s ng | B8 frames/s ng | Short-case RTF | Peak alloc / reserved (B8) |
|---|---|---|---|---|---|---|---|---|---|
| H100 | 0.302 s (0.307) | 0.465 s | 41.5 | 171.0 | 24.1 / 46.8 | 9.6 | 47.9 | 0.33 | 6.94 / 8.15 GB |
| L40S | 0.282 s (0.287) | 0.634 s | 41.7 | 141.9 | 24.0 / 56.4 | 6.6 | 38.5 | 0.31 | 6.64 / 8.61 GB |
| A10G | 0.415 s (0.421) | 0.529 s | 27.4 | 92.3 | 36.5 / 86.7 | 8.6 | 47.1 | 0.45 | 6.70 / 8.10 GB |
| L4 | 0.486 s (0.503) | 0.897 s | 24.0 | 83.4 | 41.8 / 95.9 | 4.9 | 29.9 | 0.50 | 6.70 / 8.26 GB |
| T4 | 2.66 s (2.69) | 1.01 s | 4.3 | 22.7 | 235 / 352 | not finished | not finished | 2.85 | 6.91 / 9.39 GB |

CUDA graphs speed up batch 1 by 4 to 6 times on every bf16 GPU. All 26 conformance clips ended on EOS on every GPU,
and the cancel test (8 streams, half closed after the first chunk) passed on all five.

### Turbo vs Lure, L4

End to end with the 72-frame streaming decode, same container:

| | Lure | Turbo |
|---|---|---|
| TTFA median (p90) | 0.543 s (0.642) | 0.358 s (0.411) |
| B1 frames/s | 18.0 | 24.6 |
| B2 aggregate frames/s | 26.0 | 32.2 |
| B4 aggregate frames/s | 33.6 | 38.2 |
| B8 aggregate frames/s | 39.0 | 42.2 |
| Seeded batch invariance | yes | yes |
| Peak allocated | 6.35 GB | 6.56 GB |

At this context length the streaming decode limits throughput. Engine only, without decode, with the two code
predictors swapped inside one loaded engine:

| | 15-pass code predictor | Turbo head |
|---|---|---|
| Code predictor per frame, B8 graph | 21.2 ms | 6.25 ms |
| Engine B1 | 28.8 frames/s (34.8 ms/step) | 50.6 frames/s (19.8 ms/step) |
| Engine B8 aggregate | 219 frames/s (36.5 ms/step) | 374 frames/s (21.4 ms/step) |

Turbo has not been measured on other GPUs.

### Through the core (conformance runner, L4)

The runner goes through core segmenting and the voice prompt path on every call, so its TTFA is higher than the
engine numbers above. 26 clips (13 cases, 2 takes); WER from Qwen3-ASR-1.7B, speaker similarity from ECAPA against the
voice prompt. These numbers were measured with an earlier reference clip; the clip now bundled in `conformance/assets/` is a synthetic voice.

| Mode | WER | ECAPA | Tags read aloud | EOS rate | RTF | TTFA median |
|---|---|---|---|---|---|---|
| Lure, streaming | 0.0223 | 0.582 | 0 | 1.0 | 0.69 short, 0.65 long | 1.06 s |
| Lure, non-streaming | 0.0223 | 0.581 | 0 | 1.0 | 0.47 short, 0.44 long | 1.78 s |
| Turbo, streaming | 0.0130 | 0.570 | 0 | 1.0 | 0.48 | 0.64 s |
| Turbo, non-streaming | 0.0130 | 0.570 | 0 | 1.0 | 0.26 | |

All four pass the conformance gates. Streamed and non-streamed renders produce identical frames (24 of 24 clips), so
they differ only at decoder joins.

Stream joins against the full decode of the same frames: a 25-frame context gives 9 to 15 dB SNR at chunk joins; 72
frames gives 22 to 30 dB (median 28) and the same click count as the full decode.

## Known limitations

- T4 has no bf16 tensor cores. It runs at 4.3 frames/s at batch 1, slower than real time, and there is no fp16 path.
- Seeded batch invariance (the same seed alone or inside a batch of 8) holds on L40S, A10G, L4 and T4 but not on H100,
  with graphs on or off. The cause is bf16 kernel selection by batch size, not the graphs.
- Greedy output matches qwen-tts eager on the same bf16 weights only for a short prefix (0 to 63 codebook-0 frames
  across the test cases). qwen-tts eager itself differs across GPUs, so this is bf16 drift, and there is no exact
  parity guarantee.
- The per-GPU table was measured with the older 25-frame decode context. TTFA and RTF at the default 72 have been
  measured on L4 only.
- Windows is untested.
- No weight quantization.

## Tests

```bash
pip install pytest
pytest backends/cuda/tests
```

The CPU suite (60 tests) runs on tiny random Qwen3-TTS models in float64, with no GPU and no weights. Besides the
Base model used by the older tests, there is a tiny model for each stock config in `conformance/reference/configs`
(0.6B and 1.7B Base, 0.6B and 1.7B CustomVoice, 1.7B VoiceDesign). The 1.7B-shaped ones keep talker hidden different
from code predictor hidden, so `small_to_mtp_projection` is a real projection.

- prompt rows against all entries of `conformance/reference/prompt_ids.json` and all 40 mode and layout entries of
  `prompt_modes.json`, using a symbolic stub model, and against the core's `build_prompt`;
- greedy engine codes equal to qwen-tts 0.1.1 `generate()` for `icl` and `xvec` (0.6B and 1.7B), `custom_voice`
  (0.6B and 1.7B, including dialect speakers) and `voice_design` (1.7B), each in both layouts;
- mixed batches: 8 requests of different modes and layouts in one batch, each equal to its own qwen-tts run; seeded
  sampled requests of mixed modes identical alone and batched, with slot reuse, late joins and KV rebases; parity
  across the 256 KV-length bucket boundary;
- backend requests of every mode from a core `GenerateRequest`, decode with the reference only in `icl`, mode and
  voice validation, spec and `turbo.json` checks;
- row-wise sampling against the transformers logits processors;
- engine greedy codes equal to qwen-tts `generate(non_streaming_mode=True)`, with and without instruct;
- seeded batch invariance with slot reuse, late joins and KV rebases; threaded engine with cancel;
- backend plumbing from a core `GenerateRequest` to chunks and `SegmentInfo`; streamed and offline audio identical;
  windowed decode equal to full decode on a causal fake codec;
- the Turbo head against the reference implementation in `conformance/turbo/` (float64 logits within 1e-10, a stored
  fixture, CUDA-graph predictor rows, whole-engine greedy frames, batch invariance) and the `turbo.json` rules.

The prompt checks read `config.json` and tokenizer files from `LURE_MODEL_DIR` when set, and otherwise from the
copies in `conformance/reference/tokenizer`.

GPU scripts, run directly on a CUDA machine with real weights:

```bash
python backends/cuda/tests/gpu/e2e_gpu.py --model /path/to/lure --out results/ all   # parity, stream, ttfa, render, batch, cancel
python backends/cuda/tests/gpu/turbo_gpu.py --model /path/to/lure_turbo --out results/            # Turbo head parity
python backends/cuda/tests/gpu/turbo_gpu.py --model /path/to/lure_turbo --out results/ --bench    # engine-only speed
python backends/cuda/tests/gpu/ttfa_gpu.py --model /path/to/lure --out results/                   # TTFA breakdown
python backends/cuda/tests/gpu/modes_gpu.py --models /path/to/model ... --out results/            # every mode and layout
```

`e2e_gpu.py` and `ttfa_gpu.py` use the voice and cases in `conformance/cases.py`. `e2e_gpu.py` exits 1 if a hard check
fails.
