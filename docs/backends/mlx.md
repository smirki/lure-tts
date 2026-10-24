# mlx backend (Apple silicon)

The mlx backend runs Lure on Apple silicon Macs through MLX. It lives in `backends/mlx` as the `lure-mlx` package
and registers itself as backend `mlx`. For general setup and the Python API, see
[getting-started.md](../getting-started.md) and [api.md](../api.md).

Supported checkpoints and modes:

| Checkpoint | Modes | Layouts | Status |
|---|---|---|---|
| Lure, Lure Turbo (0.6B) | `icl` | non-streaming | verified with the conformance suite on an Apple M5 (speaker-similarity gap under known limitations); Turbo head checked against the reference fixture |
| Lure family | `xvec`, `icl` streaming | both | supported, tested on fixtures |
| Qwen3-TTS-12Hz-0.6B-Base, 1.7B-Base | `icl`, `xvec` | both | supported (tested on fixtures; real-weights reports welcome via GitHub issues) |
| Qwen3-TTS-12Hz-0.6B-CustomVoice, 1.7B-CustomVoice | `custom_voice` | both | supported (tested on fixtures; real-weights reports welcome via GitHub issues) |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | `voice_design` | both | supported (tested on fixtures; real-weights reports welcome via GitHub issues) |

"Tested on fixtures" means tiny random-weight checkpoints of every variant and both size shapes, loaded through the
normal load path, with prompt rows, prefill embeddings and greedy codes matching qwen-tts 0.1.1 (see Tests). The
backend contract is in [architecture.md](../architecture.md#backend-contract).

## How it works

The backend uses the Qwen3-TTS modules of [mlx-audio](https://github.com/Blaizzy/mlx-audio) (MIT), pinned to commit
`402e3b9e4bd9bbecedc917b290896e5588c76f58`, on `mlx==0.32.4`: talker, code predictor, speech tokenizer encoder and
decoder, speaker encoder and weight loading. The backend runs its own generation loop on top of those modules:

- `load` takes the core's `ModelSpec` (`spec=`) and checks the loaded modules against it: talker and code predictor
  hidden sizes, `small_to_mtp_projection` on 1.7B, and a speaker encoder on Base only.
- The prefill is embedded from the core's prompt rows (`req.build_prompt`), so it is the reference layout for every
  mode and layout. The x-vector fills the `spk` row in `icl` and `xvec`, `ref:t` rows come from the reference codes in
  `icl` only, and a preset speaker is the codec row `C(spk_id)`. Generated frame k gets the text row `trailing[k]`,
  then `tts_pad`, which carries the rest of the text in the streaming layouts.
- On 1.7B the code predictor inputs live in the 2048-d talker space and go through `small_to_mtp_projection`
  (2048 to 1024) before the code predictor layers; the reference and feedback sums use the 2048-d embedding tables.
  mlx-audio's modules already handle these shapes.
- Sampling follows the contract: repetition penalty on codebook 0 over the whole segment history, token suppression,
  EOS masked for `min_new_tokens`, temperature, top-k, top-p, with float32 logits.
- Decoding streams through mlx-audio's stateful streaming decoder. In `icl` the decoder is primed with the reference
  frames, and a non-streamed decode runs on `[reference; generated]` and cuts the reference share, as qwen-tts does.
  The other modes decode the generated codes alone. The decoded length is all frames times 1920 samples.

Sampling uses the Gumbel-max trick instead of `torch.multinomial`. The distribution is the same but the random stream
is different, so audio never matches qwen-tts bit for bit, even with the same seed. Seeds are deterministic within
this backend, and streamed and non-streamed runs produce the same frames.

## Install

Requirements: a Mac with Apple silicon, macOS, Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install ./backends/mlx ".[server]"
lure backends          # should list mlx
```

The mlx-audio dependency installs from GitHub at the pinned commit. Other mlx-audio versions are not supported: the
adapter reaches into its internal modules.

## Usage

```bash
lure say "Hello there. (laugh) Really?" --backend mlx --model /path/to/lure \
    --ref-wav ref.wav --ref-text "Transcript of ref.wav." -o out.wav \
    --opt cache_limit_mb=128 --opt memory_limit_gb=3.5
```

```python
from lure_tts import Lure

tts = Lure.load("/path/to/lure", backend="mlx", cache_limit_mb=128, memory_limit_gb=3.5)
```

On a Mac, `backend="auto"` picks mlx when it is installed.

## Options

Pass options as `--opt key=value` on the command line or as keyword arguments to `Lure.load`.

| option | default | effect |
|---|---|---|
| `cache_limit_mb` | 512 | MLX buffer cache limit (`mx.set_cache_limit`). Lower values keep the footprint down. |
| `memory_limit_gb` | none | MLX memory limit (`mx.set_memory_limit`). |
| `compile_cp` | off | Compiles the per-frame code predictor (15 steps, or the 3 Turbo passes) into one `mx.compile` graph. Same output as the eager path. |
| `pipeline` | off | Queues frame t+1 before reading frame t back for the EOS check, so there is no host sync inside a frame. Same output; costs one wasted frame at EOS. |
| `turbo` | auto | 0.6B Lure family only. `auto` uses the Turbo head if the model folder has `turbo.json`, `1` requires it, `0` runs the original 15-step code predictor (for A/B checks only). |
| `turbo_head` | none | Path to another Turbo head file; groups still come from `turbo.json`. |
| `prime_frames` | all | `icl` only: reference frames used to prime the streaming decoder (`0` = none). |
| `dtype` | `bfloat16` | Compute dtype; weights are cast if needed. |
| `--quant` | none | `q8` (talker and code predictor 8-bit) or `mix` (talker 6-bit, code predictor 8-bit, text embedding 4-bit). Experimental and untested on the real checkpoint. |

MLX streams are thread-local, so the core runs every call for this backend on one worker thread.

## Turbo

If the model folder contains `turbo.json` and `turbo_head.safetensors`, the backend loads the Turbo head automatically.
For each 80 ms frame the head predicts codebooks 1 to 15 in 3 grouped causal passes, over (1,3), (4,7) and (8,15),
instead of 15 sequential code-predictor steps. Within a group the later codebooks go through a small add-on network.
Every position samples with the subtalker settings (default temperature 0.9, top-k 50). `compile_cp=1` compiles the
whole Turbo frame and `pipeline=1` works as usual. `--quant` leaves the head in the compute dtype.

The MLX head matches a float64 reference implementation on tiny random weights (max logit error about 5e-5 on the CPU
device, greedy codes identical). It has not run with the real weights on a Mac yet.

## Measured performance

All numbers come from the end-to-end run on an Apple M5 with 16 GB of unified memory, Lure checkpoint,
streaming unless noted, with `cache_limit_mb=128 memory_limit_gb=3.5`. RTF is wall time over audio time (below 1 is
faster than real time). TTFA is time to first audio through the conformance runner, including segmenting and the
voice prompt.

| mode | RTF short | RTF long | TTFA median | peak footprint |
|---|---|---|---|---|
| default (streaming) | 0.79 | 0.67 | 0.83 s | 4.99 GB |
| `compile_cp=1 pipeline=1` | 0.74 | 0.63 | 0.78 s | 4.97 GB |
| non-streaming | 0.93 | 0.71 | 3.56 s | 5.71 GB |

Turbo on the Mac: pending (not measured yet).

These numbers were measured with an earlier reference clip; the clip now bundled in `conformance/assets/` is a synthetic voice.

Quality on the same run (26 clips, 13 cases x 2 takes; WER from Qwen3-ASR-1.7B, speaker similarity from ECAPA against
the voice prompt):

| mode | WER | ECAPA | tags read aloud | segments ended by EOS |
|---|---|---|---|---|
| reference (qwen-tts) | 0.0150 | 0.575 | 0 | all |
| mlx, all three modes | 0.0073 | 0.533 | 0 | all |

The scores are identical across the three modes, and streamed and non-streamed runs produced the same frames. Streamed
chunk joins against the full decode: SNR 52.1 dB minimum, 57.6 dB median.

## Memory guidance

The checkpoint, MLX buffers and the decoder all share unified memory with macOS and your other apps.

- 16 GB Macs: use `cache_limit_mb=128 memory_limit_gb=3.5`. With those settings the footprint was about 5 GB on
  the M5. Without them the test run was stopped by a memory guard because swap grew by more than 1 GB during load.
- 8 GB Macs: untested. Start with the same limits, keep streaming on, quit other large apps and run one model
  process at a time. `--quant q8` should reduce memory but is untested.
- Prefer streaming. Non-streaming decodes the whole utterance at once (5.71 GB peak against 4.99 GB streamed); a
  one-shot decode of an 18 s clip has pushed the footprint past 6.6 GB.
- Run one model process at a time. Two processes each hold a full copy of the weights.

## Known limitations

- Speaker similarity is below the reference on Metal: ECAPA is 0.533 against 0.575 for the reference, which
  fails the conformance gate (reference minus 0.03 = 0.545). The cpp backend on Metal shows the same gap (0.527).
  WER is fine and the speaker embedding of the shared voice matches (cosine 1.0). Suspects are Metal precision in
  the codec decode or sampling numerics. Not yet root-caused.
- `encode_voice` (speaker encoder and speech tokenizer encoder) agrees with the reference encode at 0.959 on
  codebook 0 and 0.46 across all codebooks; the speaker embedding cosine is 1.0.
- Quantization (`q8`, `mix`) is experimental and untested on the real checkpoint.
- Turbo has not been measured on a Mac.

## Tests

```bash
pip install -e ".[dev]" -e "./backends/mlx[test]"
python -m pytest backends/mlx/tests
```

183 tests on tiny random weights, no real checkpoint (about 25 s and under 1 GB peak on an M5). `tests/tiny.py`
writes tiny checkpoints in the Hugging Face layout for 0.6B and 1.7B Base, 0.6B and 1.7B CustomVoice, 1.7B
VoiceDesign and the Lure family. The 1.7B shape keeps the talker wider than the code predictor, so
`small_to_mtp_projection` and the talker-space embedding tables are exercised. The tests cover:

- loading every variant through `MLXBackend.load` and mlx-audio's loader, and rejecting a spec that does not match
- all 40 `prompt_modes.json` entries and the 35 `prompt_ids.json` entries through `dump_layout` (prefill and trailing rows)
- prefill and trailing embeddings against an independent numpy computation of the core rows
- greedy codes, prefill and trailing rows against qwen-tts 0.1.1 on the same weights for 44 cases (every mode and
  layout on both size shapes); `fixtures/qwen_greedy.json` holds the expected values and `make_qwen_fixture.py`
  regenerates it from a qwen-tts environment
- decoding: only `icl` primes the decoder and uses the reference cut; streamed and whole runs give the same codes
- the public API on every variant, with the run list of `real_modes.py`
- sampling filters, compiled against eager code predictor, mlx-audio's own code predictor and voice-clone prefill,
  `pipeline` and `compile_cp` paths, EOS masking, seeds, and the Turbo head against the float64 reference fixture

`tests/real_modes.py` is an optional manual check: it runs every mode of one real checkpoint once and saves the wavs.
