# Contributing

Thanks for helping. Bug reports with the command, backend, hardware and `lure backends` output are the most useful
thing you can send.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest core/tests conformance           # no model needed
```

Backend tests, CPU only, no real weights:

```bash
pytest backends/mlx/tests                # Apple silicon, after pip install ./backends/mlx
pytest backends/cuda/tests               # CPU torch, see docs/backends/cuda.md for the install steps
```

The cpp backend builds with CMake presets; see [docs/backends/cpp.md](docs/backends/cpp.md).

## Rules for changes

- The prompt contract is fixed by `conformance/reference/`. A change to the segmenter, tokenizer, prompt builder or
  sampling defaults must keep `pytest core conformance` green. If it changes what a backend receives, regenerate the
  reference with `conformance/reference/gen_reference.py` (both `prompt_ids.json` and `prompt_modes.json`) and explain why in the pull request.
- Backend changes that touch generation or decoding need a conformance run (`conformance/runner.py` + `score.py`) on
  real hardware, with the summary in the pull request. Name the hardware.
- Performance claims in docs need a measurement: the script, the hardware and the numbers.
- Upstream code under `backends/` keeps its license and attribution. Add new third-party code to `NOTICE`.
- Style: Python 3.10+, type hints on public functions, comments only where the reason is not obvious from the code.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0.
