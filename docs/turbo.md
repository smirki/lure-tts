# Turbo

Lure Turbo is a model folder with a second code-predictor head. All three backends support it, and the core picks it
automatically.

## What it changes

For every 80 ms frame, the talker predicts codebook 0, then a code predictor fills codebooks 1-15. The standard code
predictor does that in 15 sequential passes, one per codebook. On an L4 at batch 8 those passes take 21.2 ms of a
36.5 ms step.

The Turbo head splits codebooks 1-15 into three groups, (1-3), (4-7) and (8-15), and runs one causal pass per group with
its own 5 transformer layers (sequence lengths 4, 8 and 16). The first codebook of each group is sampled straight from
that pass. The later codebooks in the group are sampled one by one through a small MLP add-on that conditions on the
codebooks already picked.

Every Turbo position samples with the subtalker settings (default temperature 0.9, top-k 50).

## Model folder

A Turbo folder is a normal Lure folder (with the Turbo talker weights in `model.safetensors`) plus:

| File | Contents |
|---|---|
| `turbo_head.safetensors` | the head weights, bf16, about 224 MB. Keys: `layers.{0-4}.*` (same names as `code_predictor.model.layers.*`), `norm.weight`, `lm_head.{0-14}.weight`, `mask` [14, 1024], `addon.{n1,n2,n3}.{weight,bias}` (LayerNorm, eps 1e-5), `addon.w1` [512, 3072] + bias, `addon.w2` [1024, 512] + bias, `addon_pos` [15, 512] |
| `turbo.json` | `"format": "lure-turbo-v1"`, `groups`, `hidden_size`, `num_layers`, add-on sizes, default sampling, `head_file` |

A folder with `turbo.json` counts as the Lure family even without `lure.json`. The Turbo head exists for the
0.6B Lure family only.

## Selecting it

Every backend calls `lure_tts.turbo.resolve(model_dir, turbo=..., turbo_head=...)` at load time:

| Setting | Effect |
|---|---|
| `turbo=auto` (default) | Turbo if `turbo.json` exists in the model folder, otherwise plain Lure |
| `turbo=1` | require Turbo; loading fails without `turbo.json` |
| `turbo=0` | run the original 15-pass code predictor. On a Turbo folder this is for A/B comparisons only: the Turbo talker expects its head, so the pairing is unsupported (a warning is logged) |
| `turbo_head=PATH` | load a different head file; groups and sizes still come from `turbo.json` |

```python
tts = Lure.load("/path/to/lure_turbo")             # Turbo
print(tts.turbo)                                        # True
tts = Lure.load("/path/to/lure_turbo", turbo=0)         # original code predictor, for comparison
```

```bash
lure say "..." --voice me -o out.wav --opt turbo=0
```

`Lure.turbo` tells you whether the head is active. The cuda and mlx backends also report it per segment in
`SegmentInfo.extra["turbo"]`.

## Measured

All numbers on an NVIDIA L4. Source runs are listed in [performance.md](performance.md).

Engine only (no decode), the stock code predictor and the Turbo head swapped inside one engine:

| | Stock 15-pass | Turbo head |
|---|---|---|
| code predictor per frame, batch 8 | 21.2 ms | 6.25 ms (3.4x) |
| 1 stream | 28.8 frames/s | 50.6 frames/s |
| 8 streams, aggregate | 219 frames/s | 374 frames/s |

cuda backend end to end, with streaming decode (same container):

| | Lure | Turbo |
|---|---|---|
| TTFA median (p90) | 0.543 s (0.642) | 0.358 s (0.411) |
| 1 stream | 18.0 frames/s | 24.6 frames/s |
| 8 streams, aggregate | 39.0 frames/s | 42.2 frames/s |

End to end, the streaming decode (72-frame context) takes most of the time, so the gain is smaller than engine only.

These numbers were measured with an earlier reference clip; the clip now bundled in `conformance/assets/` is a synthetic voice.

Quality, conformance suite (26 clips, streaming), against the reference WER 0.0150 and ECAPA 0.575:

| Backend | Lure WER / ECAPA | Turbo WER / ECAPA |
|---|---|---|
| cuda, L4 | 0.0223 / 0.582 | 0.0130 / 0.570 |
| cpp CUDA, L4 | 0.0150 / 0.562 | 0.0001 / 0.566 |
| cpp CPU, 16 vCPU | 0.0098 / 0.544 (older build) | 0.0141 / 0.579 |

Turbo passes every gate on all three, with 0 tags read and every segment ending on EOS.

On the cpp CPU build, Turbo cuts the non-streaming RTF from 6.72 to 3.70 (same build, 4 cases). Streaming on CPU stays
bound by the codec decode (RTF 12.66 vs 10.26).

Not measured yet: Turbo on a Mac (mlx and cpp Metal). The head is unit-tested on both against the float64 reference.

## Correctness tests

`conformance/turbo/` holds a float64 reference implementation of the head and a tiny random-weight fixture. Each backend
tests against it: cuda checks logits at all 15 positions to 1e-10 in float64, CUDA-graph rows, and greedy frames against
qwen-tts `generate()` with the reference head patched in; mlx checks logits and greedy codes on the MLX CPU device; cpp
checks CPU and Metal logits and codes. On real weights (L4, bf16, 309 frames), the cuda engine head matched the reference
head with a max logit difference of 0.0 and identical codes.
