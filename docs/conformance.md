# Conformance

The conformance suite answers two questions about a backend: does it feed the model the reference prompt, and does its
audio match the reference implementation in distribution. It lives in `conformance/`.

Audio is never compared bit for bit. The reference runs fp32 weights under bf16 autocast, the backends run bf16 with
their own kernels and random streams, so the same seed gives different samples. Instead every backend renders the same
cases with the same voice prompt and seeds, and the scores are compared with the reference run.

## 1. Prompt check (no weights)

`conformance/reference/prompt_ids.json` holds 35 Lure prompts (`icl`, non-streaming) produced by the qwen-tts
prompt code. A backend dumps its own
prefill rows for each entry and `check_prompt.py` compares them:

```bash
python conformance/check_prompt.py my_backend_dump.json
```

`conformance/reference/prompt_modes.json` adds 40 prompts that cover every mode and layout (`icl`, `xvec`,
`custom_voice`, `voice_design`, each non-streaming and streaming) on the stock 0.6B and 1.7B configs in
`conformance/reference/configs/`, with preset and dialect speakers, language `auto` and instructs. These entries are
compared exactly, prefill and trailing rows both.

Backends that implement `dump_layout()` get this from the runner automatically (`prompt_check.json`). The runner checks
the `prompt_ids.json` entries and every `prompt_modes.json` entry whose mode and config match the loaded model, and
lists the rest as unsupported. Current results on `prompt_ids.json`: mlx 35/35, cpp 35/35; the cuda backend runs the
same check in its unit tests. No backend dumps the other modes yet. Format details:
[conformance/reference/README.md](../conformance/reference/README.md).

## 2. Cases

`conformance/cases.py` has 13 cases. They clone a voice with a Lure-family model; audio cases for the stock
checkpoints and the other modes are not written yet.

| Case | What |
|---|---|
| a-d | short lines; c and d carry a direction |
| e | a ~97 s passage, cut into 14 segments by the core |
| t1-t6 | tagged lines: `(chuckle)`, `(gasp)`, `(nervous laugh)`, `(sharp inhale)`, `(sigh)`, `(laugh)`, two with a direction |
| cn, dn | c and d without the direction, for A/B |

Each case runs with 2 takes by default, seeded with `cases.seed(case, take)`: 26 clips per backend.

## 3. Render

The runner drives any backend through the public API, so segmentation, prompt ids, seeds and sampling are identical by
construction.

```bash
# Reference run: qwen-tts generate() on a CUDA GPU (pip install -r conformance/requirements-ref.txt)
python conformance/runner.py --backend ref_backend:QwenTTSRef --model /path/to/lure --out runs/ref

# Any backend, cloning the voice prompt the reference encoded
python conformance/runner.py --backend cuda --model /path/to/lure --out runs/cuda \
    --voice-npz runs/ref/voice_qwen-tts-ref.npz
python conformance/runner.py --backend cuda --model /path/to/lure --out runs/cuda_full \
    --voice-npz runs/ref/voice_qwen-tts-ref.npz --no-stream
python conformance/runner.py --backend cpp --opt ggml_backend=cpu --model /path/to/lure --out runs/cpp_cpu \
    --voice-npz runs/ref/voice_qwen-tts-ref.npz --cases short --takes 1
```

| Flag | Meaning |
|---|---|
| `--backend` | backend name or `module:Class` |
| `--voice-npz` | shared voice prompt (`ref_codes` [T, 16] int32, `spk_emb` [1024] float32) so every backend clones the same prompt; the runner also encodes with the backend and records the agreement in `env.json` |
| `--cases` | `short`, `tags`, `long`, `all` or a comma list such as `a,b,t1` |
| `--takes` | takes per case (default 2) |
| `--opt KEY=VALUE` | backend option, repeatable (for example `turbo=1`, `compile_cp=1`) |
| `--no-stream` | decode each segment whole |

Output per run: one wav per case and take, `ref.wav` (the voice clip, used as the ECAPA anchor), `manifest.jsonl`
(segments, frames, EOS per segment, TTFA, RTF), `env.json`, `voice_<backend>.npz`, `summary_local.json` (EOS rate, RTF,
TTFA without ASR) and `prompt_check.json` when available.

## 4. Score

```bash
pip install -r conformance/requirements-score.txt
python conformance/score.py runs/cuda runs/cuda_full runs/cpp_cpu --ref runs/ref
```

For each clip, Qwen3-ASR-1.7B transcribes the audio. WER is computed against the text with tags removed, ECAPA
(SpeechBrain, VoxCeleb) gives the cosine similarity to the voice clip, and the tag-read check looks for tag words
(`laugh`, `sigh`, `chuckle`, `gasp`, `inhale`, ...) in the transcript that are not in the plain text. Results go
to `scores.jsonl`, `summary.json` and, with `--ref`, `gates.json`. The exit code is 1 if a run
fails a gate.

## 5. Gates

From `scoring.GATES`, against the reference run with the same cases, seeds and voice:

| Gate | Value |
|---|---|
| mean WER | at most 0.10, and at most reference + 0.03 |
| mean ECAPA | at least reference - 0.03 |
| tags read aloud | 0 |
| EOS rate | 1.0 (no segment hits its frame cap) |

The reference scores quoted in the docs (WER 0.0150, ECAPA 0.575) were measured with an earlier reference clip; the clip
now bundled in `conformance/assets/` is a synthetic voice, so rerun the reference before applying the gates.

Latest results are in the README and [performance.md](performance.md). Listen as well: tags performed rather than read,
the direction audible, no clicks at chunk or segment joins.

## Unit tests

```bash
pytest conformance
```

Covers WER normalisation, tag-read detection, gates, the runner with the fake backend and `score.py` gating from
existing summaries. No models are loaded.
