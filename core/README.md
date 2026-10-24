# core: the `lure_tts` package

The Python package every backend sits behind: public API, CLI, OpenAI-compatible server, model inspection, text
segmenter, prompt builder, sampling defaults, on-disk voice cache, streaming scheduler, backend selection and Turbo resolution. It needs
only numpy, regex, soundfile and soxr; it never imports torch or mlx.

It is packaged by the `pyproject.toml` at the repository root (distribution `lure-tts`). Install from the root:

```bash
pip install -e ".[dev]"                      # core + test deps, runs with the fake backend
pip install ./backends/mlx ".[server]"       # plus a real backend, see docs/getting-started.md
```

## Modules

| Module | What |
|---|---|
| `api.py` | `Lure` (load, make_voice, stream, speak, live) and `LiveSession` |
| `cli.py` | the `lure` command |
| `server.py` | FastAPI app for `lure serve` and request validation |
| `text.py` | tag-safe segmenter (batch and incremental modes) |
| `model.py` | `inspect(model_dir)` -> `ModelSpec`: variant, size, speakers, languages, modes, layout, inline tags |
| `prompt.py` | prompt ids, prefill and trailing rows for every mode and layout, checked against `conformance/reference/` |
| `tokenizer.py` | pure-Python Qwen2 BPE tokenizer (added tokens as single ids, NFC) |
| `defaults.py` | `Sampling`, frame rate, chunk sizes, per-segment frame cap |
| `voice.py` | on-disk voice cache |
| `scheduler.py` | segments to backend requests, worker thread, back-pressure and cancellation |
| `turbo.py` | `turbo.json` parsing and the `turbo=auto/1/0` rule shared by all backends |
| `backends/` | backend protocol (`base.py`), registry and auto-selection, the fake test backend |

## Tests

```bash
pytest core/tests
```

No weights needed. The tests cover model inspection on the stock configs, prompts against the qwen-tts reference
dumps for every mode and layout, tokenizer ids, the segmenter
(tags, punctuation, abbreviations, numbers, streaming), CLI and server request parsing, API and server end to end with
the fake backend (including the mode errors), concurrent versus serialized streams, and the Turbo resolve rules.

Usage docs: [docs/api.md](../docs/api.md). How backends plug in: [docs/architecture.md](../docs/architecture.md).
