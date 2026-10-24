# Conformance suite

Checks that a backend produces the reference prompt and sounds like the reference implementation. Full guide:
[docs/conformance.md](../docs/conformance.md).

| File | What |
|---|---|
| `reference/` | prompt-id and tokenizer dumps from the qwen-tts prompt builder, plus tokenizer files ([reference/README.md](reference/README.md)) |
| `check_prompt.py` | compares a backend's prompt dump with `reference/prompt_ids.json` |
| `cases.py` | the test lines, the voice prompt and per-case seeds |
| `assets/` | the voice prompt clip (24 kHz) |
| `runner.py` | renders the cases through the core API on any backend: wavs, `manifest.jsonl`, `env.json` |
| `ref_backend.py` | the qwen-tts `generate()` wrapped as a backend, used for the reference run (CUDA) |
| `score.py` | Qwen3-ASR WER, ECAPA similarity, tag-read check, pass gates |
| `scoring.py` | the pure parts of scoring (WER, tag-read detection, summaries, gates) |
| `turbo/` | float64 reference of the Turbo head and the tiny fixture the backend tests compare against |
| `requirements-ref.txt`, `requirements-score.txt` | extra deps for the reference run and for scoring |

Quick start:

```bash
python conformance/runner.py --backend ref_backend:QwenTTSRef --model /path/to/lure --out runs/ref      # CUDA GPU
python conformance/runner.py --backend mlx --model /path/to/lure --out runs/mlx --voice-npz runs/ref/voice_qwen-tts-ref.npz
python conformance/score.py runs/mlx --ref runs/ref
```

Unit tests (no models): `pytest conformance`.
