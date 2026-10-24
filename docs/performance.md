# Performance

All numbers below were measured on 2026-10-23 with an earlier reference clip; the clip now bundled in
`conformance/assets/` is a synthetic voice. Definitions:

- RTF: wall time divided by audio duration. Below 1.0 is faster than real time.
- TTFA: time from request to the first audio chunk.
- frames/s: codec frames generated per second. One frame is 80 ms of audio, so 12.5 frames/s is one real-time stream.

Windows has not been measured. Turbo has not been measured on a Mac.

## H100 time to first audio

cuda backend at engine level, CUDA graphs on, warm (3 warmup calls, then 15 timed streamed calls of a directed line), one
request at a time. "Slots" is the engine's `max_batch`. Each cell is the median TTFA with the best run in brackets. The
default is decode context 72 with a 4-frame first chunk.

| Model | Slots | Decode context | First chunk 1 | First chunk 2 | First chunk 4 | ms per frame after the first | RTF |
|---|---|---|---|---|---|---|---|
| Lure | 1 | 72 | 0.229 s (0.223) | 0.257 s (0.240) | 0.283 s (0.278) | 18.4 | 0.23 |
| Lure | 1 | 25 | 0.221 s (0.200) | 0.239 s (0.223) | 0.276 s (0.257) | 17.0 | 0.21 |
| Lure | 8 | 72 | 0.266 s (0.225) | 0.269 s (0.261) | 0.331 s (0.306) | 19.6 | 0.25 |
| Lure | 8 | 25 | 0.239 s (0.219) | 0.261 s (0.242) | 0.303 s (0.282) | 18.6 | 0.23 |
| Turbo | 1 | 72 | 0.149 s (0.137) | 0.157 s (0.148) | 0.168 s (0.164) | 11.4 | 0.14 |
| Turbo | 1 | 25 | 0.135 s (0.121) | 0.143 s (0.135) | 0.164 s (0.155) | 10.1 | 0.13 |
| Turbo | 8 | 72 | 0.167 s (0.152) | 0.173 s (0.156) | 0.200 s (0.177) | 12.4 | 0.15 |
| Turbo | 8 | 25 | 0.152 s (0.141) | 0.161 s (0.149) | 0.191 s (0.174) | 11.0 | 0.14 |

Prompt building and prefill take about 2 ms of this. The first decoded window takes about 16 ms.

## cuda backend across GPUs

Engine level (the backend's own GPU test script, no core segmenting), Lure, bf16, 8-slot engine. "ng" = CUDA graphs
off. These runs used the older 25-frame streaming decode context; the default is now 72, which costs more decode time.

| GPU | TTFA median (p90) | TTFA ng | 1 stream frames/s | 8 streams frames/s | ms/step 1 / 8 | 1 stream ng | 8 streams ng | RTF short | Peak alloc / reserved (8 streams) |
|---|---|---|---|---|---|---|---|---|---|
| H100 | 0.302 s (0.307) | 0.465 s | 41.5 | 171.0 | 24.1 / 46.8 | 9.6 | 47.9 | 0.33 | 6.94 / 8.15 GB |
| L40S | 0.282 s (0.287) | 0.634 s | 41.7 | 141.9 | 24.0 / 56.4 | 6.6 | 38.5 | 0.31 | 6.64 / 8.61 GB |
| A10G | 0.415 s (0.421) | 0.529 s | 27.4 | 92.3 | 36.5 / 86.7 | 8.6 | 47.1 | 0.45 | 6.70 / 8.10 GB |
| L4 | 0.486 s (0.503) | 0.897 s | 24.0 | 83.4 | 41.8 / 95.9 | 4.9 | 29.9 | 0.50 | 6.70 / 8.26 GB |
| T4 | 2.66 s (2.69) | 1.01 s | 4.3 | 22.7 | 235 / 352 | not finished | not finished | 2.85 | 6.91 / 9.39 GB |

CUDA graphs speed up a single stream 4 to 6x on every bf16 GPU. The T4 has no bf16 tensor cores and is slower than real
time with one stream. All 26 clips ended on EOS on every GPU, and the cancel test passed on all five.

## Every backend through the public API

Measured by the conformance runner, so segmentation, the voice prompt path and chunked decode are included. Lure,
streaming unless noted.

| Backend | Hardware | RTF short | RTF long | TTFA median | Peak memory |
|---|---|---|---|---|---|
| reference (qwen-tts, no streaming) | L4 | 2.54 | 2.42 | 10.6 s | |
| cuda | L4 | 0.69 | 0.65 | 1.06 s | |
| cuda, no streaming | L4 | 0.47 | 0.44 | 1.78 s | |
| cpp CUDA | L4 | 0.53 | 0.50 | 0.17 s | |
| cpp CUDA, no streaming | L4 | 0.31 | 0.28 | 1.17 s | |
| cpp CPU | 16 vCPU | 6.64 | 6.26 | 2.36 s | |
| mlx | Apple M5, 16 GB | 0.79 | 0.67 | 0.83 s | 4.99 GB |
| mlx, no streaming | Apple M5, 16 GB | 0.93 | 0.71 | 3.56 s | 5.71 GB |
| mlx, `compile_cp=1 pipeline=1` | Apple M5, 16 GB | 0.74 | 0.63 | 0.78 s | 4.97 GB |
| cpp Metal | Apple M5, 16 GB | 1.05 | 0.92 | 0.36 s | 4.04 GB |

Notes:

- cuda TTFA through the API (1.06 s on L4) is higher than at engine level (0.49 s) because each call goes through core
  segmenting and the voice-prompt path.
- mlx ran with `cache_limit_mb=128 memory_limit_gb=3.5` to keep memory low on the 16 GB machine.
- The cpp CPU run used a build with the older 25-frame stream decoder; the current 72-frame decoder costs more on CPU.

## Turbo

On an L4. Engine only, the stock code predictor and the Turbo head swapped inside one engine:

| | Stock 15-pass | Turbo head |
|---|---|---|
| code predictor per frame, 8-stream graph | 21.2 ms | 6.25 ms |
| 1 stream | 28.8 frames/s (34.8 ms/step) | 50.6 frames/s (19.8 ms/step) |
| 8 streams, aggregate | 219 frames/s (36.5 ms/step) | 374 frames/s (21.4 ms/step) |

cuda backend end to end with streaming decode, same container:

| | Lure | Turbo |
|---|---|---|
| TTFA median (p90) | 0.543 s (0.642) | 0.358 s (0.411) |
| 1 stream | 18.0 frames/s | 24.6 frames/s |
| 2 streams, aggregate | 26.0 | 32.2 |
| 4 streams, aggregate | 33.6 | 38.2 |
| 8 streams, aggregate | 39.0 | 42.2 |
| peak allocated | 6.35 GB | 6.56 GB |

End to end, the 72-frame streaming decode dominates, which is why the Turbo gain is smaller here than at engine level
and why these frames/s are lower than the engine-level table above.

Conformance runner, streaming:

| Backend | Hardware | Lure RTF / TTFA | Turbo RTF / TTFA |
|---|---|---|---|
| cuda | L4 | 0.66 / 1.06 s | 0.48 / 0.64 s |
| cpp CUDA | L4 | 0.51 / 0.175 s | 0.40 / 0.139 s |

Non-streaming: cuda 0.45 to 0.26, cpp CUDA 0.28 to 0.17. On cpp CPU (same build, 4 cases), non-streaming RTF goes
from 6.72 to 3.70; streaming stays decode-bound (12.66 to 10.26).

## Quality

| Backend (streaming) | Hardware | Model | WER | ECAPA | Tags read | EOS rate |
|---|---|---|---|---|---|---|
| reference (qwen-tts) | L4 | Lure | 0.0150 | 0.575 | 0 | 1.0 |
| cuda | L4 | Lure | 0.0223 | 0.582 | 0 | 1.0 |
| cuda, no streaming | L4 | Lure | 0.0223 | 0.581 | 0 | 1.0 |
| cuda | L4 | Turbo | 0.0130 | 0.570 | 0 | 1.0 |
| cpp CUDA | L4 | Lure | 0.0150 | 0.562 | 0 | 1.0 |
| cpp CUDA, no streaming | L4 | Lure | 0.0150 | 0.561 | 0 | 1.0 |
| cpp CUDA | L4 | Turbo | 0.0001 | 0.566 | 0 | 1.0 |
| cpp CPU (older build) | 16 vCPU | Lure | 0.0098 | 0.544 | 0 | 1.0 |
| cpp CPU | 16 vCPU | Turbo | 0.0141 | 0.579 | 0 | 1.0 |
| mlx | Apple M5 | Lure | 0.0073 | 0.533 | 0 | 1.0 |
| cpp Metal | Apple M5 | Lure | 0.0112 | 0.527 | 0 | 1.0 |

26 clips per row, scored with Qwen3-ASR-1.7B and ECAPA. The 97 s passage ended every segment on EOS on every backend,
with WER 0.000 to 0.007. Within a backend, streaming and non-streaming produce identical frames; they differ only at
decoder chunk joins (median join SNR vs full decode: cuda 28.1 dB, cpp CUDA 28.8 dB, mlx 57.6 dB).

## Reproducing

```bash
python benchmarks/ttfa.py  --model /path/to/lure --backend cuda
python benchmarks/rtf.py   --model /path/to/lure --backend cuda
python benchmarks/batch.py --model /path/to/lure --backend cuda
```

Quality: [conformance.md](conformance.md). Engine-level cuda scripts: `backends/cuda/tests/gpu/`.
