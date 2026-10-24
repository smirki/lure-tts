# Lure cpp backend

A C++ inference library for Lure on [ggml](https://github.com/ggml-org/ggml), with Metal, CUDA, Vulkan and CPU
builds. It provides:

Status: implements the [backend contract](../../docs/architecture.md#backend-contract) for every Qwen3-TTS 12Hz
checkpoint, 0.6B and 1.7B: Base (`icl`, `xvec`), CustomVoice (`custom_voice`) and VoiceDesign (`voice_design`), each in
the non_streaming and streaming prompt layouts. Lure and Lure Turbo are verified with the conformance suite on the
CUDA and CPU builds, and Lure on Metal. The stock Qwen3-TTS checkpoints are supported (tested with prompt dumps and
fixtures; real-weights reports welcome via GitHub issues). Implemented but untested: the Vulkan and Windows builds.

- `liblure_tts`, a shared library with a small C API (`include/lure_tts.h`)
- `lure_cli`, for making voices, generating speech and serving a raw-PCM HTTP API
- `lure_gguf`, which converts a model folder (including the Turbo head) to a single GGUF file
- `lure_cpp`, a Python binding that registers backend `cpp` with the `lure_tts` package

## Quick start

```bash
cd backends/cpp
cmake --preset macos-metal          # or linux-cuda, linux-cpu, linux-vulkan, macos-cpu
cmake --build --preset macos-metal -j4
build/macos-metal/bin/lure_cli generate --model /path/to/lure \
    --ref-wav ref.wav --ref-text "Transcript of ref.wav." --text "Hello there." --out out.wav
build/macos-metal/bin/lure_cli generate --model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --speaker vivian --text "Hello there." --out vivian.wav
```

To use it from Python, build the library first, then from the repository root:

```bash
pip install ".[server]"
pip install -e ./backends/cpp/python
```

Windows presets are included but untested. Full build instructions, the C API, the server protocol, measured speed and
known limitations are in [docs/backends/cpp.md](../../docs/backends/cpp.md). For the Python API, see
[docs/getting-started.md](../../docs/getting-started.md) and [docs/api.md](../../docs/api.md).

## License and credits

This code is a trimmed and modified copy of [audio.cpp](https://github.com/0xShug0/audio.cpp) by 0xShug0, licensed
under Apache-2.0 (see `LICENSE`). `NOTICE` lists the changes and the licenses of the vendored libraries in
`external/`.
