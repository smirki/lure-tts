# Benchmarks

Plain scripts that measure speed through the public API on any backend. Each prints a JSON result and writes it to
`--out` if given. They load the real model; run them on the machine you want numbers for.

| Script | Measures |
|---|---|
| `ttfa.py` | time to first audio after warmup (median, p90, min), optionally for several first-chunk sizes |
| `rtf.py` | real-time factor on short lines and on the ~90 s passage, streamed or `--no-stream` |
| `batch.py` | aggregate frames per second with 1, 2, 4, 8 concurrent streams |

```bash
export LURE_MODEL=/path/to/lure
python benchmarks/ttfa.py --backend cuda --first-chunk-frames 1,2,4 --out results/ttfa.json
python benchmarks/rtf.py --backend mlx --opt cache_limit_mb=128 --out results/rtf.json
python benchmarks/batch.py --backend cuda --concurrency 1,2,4,8 --out results/batch.json
python benchmarks/ttfa.py --model /path/to/lure_turbo --opt turbo=0     # Turbo folder, original code predictor
```

Common flags: `--model`, `--backend`, `--opt KEY=VALUE`, `--ref-wav` / `--ref-text` (default: the conformance voice),
`--out`.

These numbers include everything a caller pays: segmentation, prompt building, the scheduler thread and chunked decode.
Backend-level numbers (below the core API) are lower; the cuda backend has its own GPU test scripts in `backends/cuda/tests/gpu/`.
Published results and their hardware: [docs/performance.md](../docs/performance.md).
