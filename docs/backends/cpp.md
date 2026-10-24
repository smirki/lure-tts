# cpp backend (ggml: Metal, CUDA, Vulkan, CPU)

The cpp backend is a C++ library, `liblure_tts`, with a command-line tool and HTTP server (`lure_cli`), a model
converter (`lure_gguf`) and a Python binding that registers backend `cpp`. It lives in `backends/cpp`. For general
setup and the Python API, see [getting-started.md](../getting-started.md) and [api.md](../api.md).

Status: the backend implements the [backend contract](../architecture.md#backend-contract) for every Qwen3-TTS
12Hz checkpoint. Implemented but untested: the Vulkan and Windows builds.

| Checkpoint | Modes | Status |
|---|---|---|
| Lure, Lure Turbo (0.6B Base) | `icl`, `xvec` | verified with the conformance suite (audio) on CUDA and CPU; Lure on Metal |
| Qwen3-TTS-12Hz-0.6B-Base, -1.7B-Base | `icl`, `xvec` | supported (tested with prompt dumps and fixtures; real-weights reports welcome via GitHub issues) |
| Qwen3-TTS-12Hz-0.6B-CustomVoice, -1.7B-CustomVoice | `custom_voice` | supported (tested with prompt dumps and fixtures; real-weights reports welcome via GitHub issues) |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | `voice_design` | supported (tested with prompt dumps and fixtures; real-weights reports welcome via GitHub issues) |

Every mode runs in both prompt layouts, `non_streaming` and `streaming`. The prompt dumps match all 35 entries of
`conformance/reference/prompt_ids.json` and all 40 of `prompt_modes.json`. The Turbo head exists for 0.6B only (it needs
equal talker and code-predictor hidden sizes).

The code is a trimmed copy of [audio.cpp](https://github.com/0xShug0/audio.cpp) by 0xShug0 (Apache-2.0) on
[ggml](https://github.com/ggml-org/ggml) (MIT). Only the Qwen3-TTS model path and the framework code it links are kept.
`backends/cpp/NOTICE` lists what was changed and the licenses of the vendored libraries in `backends/cpp/external/`.

## How it works

- Prompt: the qwen-tts `generate()` prefill for each mode and layout, as specified in
  [architecture.md](../architecture.md) and `core/lure_tts/prompt.py`:
  - `icl` (Base): reference codes, reference transcript and speaker embedding;
  - `xvec` (Base): the speaker embedding only;
  - `custom_voice` (CustomVoice): the preset speaker's codec id; a dialect speaker (for example `eric`, `dylan`)
    switches to its dialect language when the language is `chinese` or `auto`;
  - `voice_design` (VoiceDesign): no speaker row; the instruct describes the voice.

  The optional instruct goes in as `<|im_start|>user\n{instruct}<|im_end|>\n` before the role rows. It is rejected on
  the 0.6B CustomVoice, as in qwen-tts. The text tokenizer applies NFC normalization and keeps special tokens as single
  ids. The prompt and tokenizer match the conformance reference: 35/35 `prompt_ids.json` entries, 40/40
  `prompt_modes.json` entries and 41/41 tokenizer strings.
- Defaults per checkpoint: the Lure family uses the non_streaming layout and English; stock Base uses the streaming
  layout and stock CustomVoice and VoiceDesign the non_streaming layout, all with language `auto`. A request can
  override the layout and language.
- Decode: `icl` decodes the reference codes in front of the generated codes and cuts them; the other modes decode the
  generated codes alone.
- 1.7B: talker hidden size 2048 with a 1024-wide code predictor behind `small_to_mtp_projection`, and a 2048-d speaker
  embedding on Base. The loader checks the tensor shapes against `config.json` and names any mismatch.
- One segment per call. The core splits text into segments; the library never re-chunks text given to
  `lure_generate`.
- Sampling defaults: temperature 0.9, top-k 50, top-p 1.0, repetition penalty 1.05 on codebook 0, and the same 0.9 / 50 /
  1.0 for the code predictor. Every value can be changed per call.
- Streaming decode: each chunk decodes a window of 72 frames of left context plus the new frames, so chunk joins match
  the full decode. The first chunk is primed with the tail of the reference codes. With `chunk_grow` the chunk sizes
  are 1, 2, 4, 4, ... frames for the earliest first audio.
- Voices: a voice is the reference codes plus the speaker embedding. It can be saved to a `.alv` file, cached on disk
  by content, or built straight from the core's cached voice (`lure_voice_from_codes`) without running the encoders.
- Turbo: when the model folder has `turbo.json` (or a GGUF carries it), the 15 sequential code-predictor passes per
  frame become 3 group passes (codebooks 1-3, 4-7, 8-15) plus a small add-on MLP for the later codebooks of each group.
  The talker, prompt and codec are unchanged. See [turbo.md](../turbo.md).
- Metal startup: with `LURE_METAL_AOT=ON` (on in the `macos-metal` preset) the build precompiles
  `ggml-metal-bf16.metallib` and `ggml-metal-bf16-tensor.metallib` next to the library. A fresh binary then loads its
  Metal library in 3 ms instead of compiling shaders for 6.9 s (Apple M5). Ship both files next to
  `liblure_tts.dylib`. If they are missing or fail to load, ggml compiles the embedded source as usual;
  `GGML_METAL_NO_AOT=1` skips the lookup and `GGML_METAL_LIBRARY_DIR` points it elsewhere.

## Build

Requirements: CMake 3.21 or newer, Ninja, and a C++17 compiler. Each preset builds into `build/<preset>/bin/`:
`liblure_tts` (`.dylib`, `.so` or `.dll`), `lure_cli`, `lure_gguf`, `lure_turbo_head_test` and
`lure_qwen3_variants_test`.

| Preset | Platform | Status |
|---|---|---|
| `macos-metal` | macOS, Apple silicon, Metal | built and used for the Apple M5 numbers below |
| `macos-cpu` | macOS, CPU only | built |
| `linux-cuda` | Linux, NVIDIA (sm 75, 80, 86, 89, 90) | built and used for the L4 numbers below |
| `linux-cpu` | Linux, portable x86 (all CPU variants, picked at run time) | built and used for the 16 vCPU numbers below |
| `linux-cpu-native` | Linux, CPU, this host's ISA only | not built |
| `linux-vulkan` | Linux, Vulkan (AMD, Intel, NVIDIA) | not built |
| `windows-cpu`, `windows-cuda`, `windows-vulkan` | Windows, MSVC | untested |

### macOS (Metal)

```bash
xcode-select --install          # compiler; the Metal toolchain (xcrun metal) is needed for the precompiled metallibs
brew install cmake ninja
cd backends/cpp
cmake --preset macos-metal
cmake --build --preset macos-metal -j4
```

Without the Metal toolchain the build prints a warning, skips the metallibs and falls back to the embedded shader
source. `macos-cpu` builds the same tools without Metal.

### Linux (CUDA)

CUDA 12.0 or newer and its `nvcc`.

```bash
sudo apt install cmake ninja-build build-essential
cd backends/cpp
cmake --preset linux-cuda
cmake --build --preset linux-cuda -j8
```

The preset compiles for five GPU architectures. For a local build, add `-DCMAKE_CUDA_ARCHITECTURES=native` (or the one
architecture you need) to the configure step; it is much faster. The GPU presets build ggml's CPU kernels without
`-march=native`, so a binary built on one machine also runs on hosts with other CPUs. A build with native CPU flags
crashed with SIGILL on a different cloud host.

### Linux (CPU or Vulkan)

```bash
cmake --preset linux-cpu    && cmake --build --preset linux-cpu -j8       # portable x86
cmake --preset linux-vulkan && cmake --build --preset linux-vulkan -j8    # needs the Vulkan SDK and glslc
```

`linux-cpu` builds ggml as shared libraries with one CPU backend per x86 ISA level (`libggml-cpu-*.so`) and picks one
at run time. Keep those files next to `liblure_tts.so`; the library loads them from its own directory.
`linux-cpu-native` builds a single static CPU backend for the build host.

### Windows (untested)

The `windows-cpu`, `windows-cuda` and `windows-vulkan` presets exist but have never been built. Run them from an
"x64 Native Tools" prompt (they use `cl` with the `/MD` runtime). `windows-cpu` produces `ggml-cpu-*.dll` variants that
must sit next to `lure_tts.dll`. CUDA needs Visual Studio 2022 17.x with CUDA 12.4 or newer; Vulkan needs the LunarG
SDK and `glslc` on `PATH`. Paths are UTF-8 throughout; ctypes callers should pass UTF-8 bytes, as `lure_cpp` does.

### Build options

| Option | Default | Meaning |
|---|---|---|
| `ENGINE_ENABLE_METAL` / `_CUDA` / `_VULKAN` | Metal on Apple, others off | ggml GPU backend |
| `ENGINE_ENABLE_NATIVE_CPU` | ON | ggml CPU kernels with `-march=native` |
| `ENGINE_ENABLE_CPU_ALL_VARIANTS` | OFF | per-ISA CPU backends as shared libraries |
| `ENGINE_ENABLE_OPENMP` | ON | OpenMP in host code (off in the macOS presets) |
| `ENGINE_ENABLE_RKNN` | OFF | Rockchip NPU speech decoder through librknnrt (untested) |
| `LURE_METAL_AOT` | OFF | precompiled Metal libraries (Apple only) |
| `LURE_BUILD_TESTS` | ON | build `lure_turbo_head_test` and `lure_qwen3_variants_test` |

## Model files

`liblure_tts` loads a model folder directly (`config.json`, `model.safetensors`, tokenizer files and
`speech_tokenizer/`, plus `turbo.json` and `turbo_head.safetensors` for Turbo). Any Qwen3-TTS 12Hz checkpoint folder
works: Base, CustomVoice (preset speakers from `talker_config.spk_id`) and VoiceDesign, at 0.6B or 1.7B. CustomVoice and
VoiceDesign checkpoints have no speaker encoder. A GGUF file is optional:

```bash
build/macos-metal/bin/lure_gguf --hf-dir /path/to/lure --output lure.gguf             # bf16
build/macos-metal/bin/lure_gguf --hf-dir /path/to/lure --output lure-q8_0.gguf --type q8_0
build/macos-metal/bin/lure_gguf --inspect lure.gguf
```

The GGUF embeds the configs and tokenizer, so it is a single file. If the folder has `turbo_head.safetensors`, the head
goes into the GGUF as its own tensor namespace and `turbo.json` is embedded, so the GGUF runs Turbo on its own. A bf16
head converts losslessly with the default `--type bf16`; other types quantize the head like every other matrix.
Symlinked folders (for example a symlinked `speech_tokenizer/`) are followed.

The converter copies every tensor of the checkpoint, so the 1.7B tensors (`small_to_mtp_projection`, the 2048-d speaker
encoder) and the CustomVoice and VoiceDesign configs need no special handling; `turbo.json` may omit `hidden_size`,
which then defaults to the code predictor's. The converter has been tested on a synthetic model folder with small random
tensors (the Turbo head from a GGUF gives the same logits as from safetensors). The numbers below all load the model
folder.

## Command line

```bash
B=backends/cpp/build/macos-metal/bin
$B/lure_cli info --model /path/to/model          # variant, speakers, languages, modes (no weights loaded)
$B/lure_cli make-voice --model /path/to/lure --ref-wav ref.wav --ref-text "Transcript of ref.wav." --out me.alv
$B/lure_cli generate --model /path/to/lure --voice me.alv \
    --text "(sigh) I guess we're doing this again." --instruct "Tired and resigned." --seed 1234 --out out.wav
$B/lure_cli generate --model /path/to/lure --ref-wav ref.wav --x-vector-only --text "Hello." --out xvec.wav
$B/lure_cli generate --model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice --speaker vivian \
    --instruct "Cheerful." --text "Hello." --out vivian.wav
$B/lure_cli generate --model /path/to/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --instruct "A deep, calm male voice." --text "Hello." --out designed.wav
```

The mode follows from the flags: `--voice` or `--ref-wav` clones (`icl` with a transcript, `xvec` without one or with
`--x-vector-only`), `--speaker` picks a preset (`custom_voice`), and neither runs `voice_design`, which needs
`--instruct`. `generate` prints frames, `ended_by_eos` and timings to stderr. Flags:

| Flag | Meaning |
|---|---|
| `--voice V.alv`, `--ref-wav W`, `--ref-text R`, `--x-vector-only` | voice to clone (Base) |
| `--speaker NAME`, `--speaker-id N` | preset speaker (CustomVoice), name case-insensitive |
| `--instruct I` | style direction, or the voice description on VoiceDesign |
| `--backend auto\|metal\|cuda\|vulkan\|cpu`, `--device N`, `--threads N` | where to run (`--threads` for CPU work, 0 = all cores) |
| `--options "k=v;k=v"` | session options, for example `qwen3_tts.weight_type=q8_0`, `turbo=0`, `turbo_head=PATH` |
| `--seed`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--greedy` | talker sampling |
| `--subtalker-temperature`, `--subtalker-top-k`, `--subtalker-top-p` | code-predictor sampling |
| `--max-new-tokens N` | frame cap; 0 = the checkpoint's `generation_config.json` |
| `--chunk-frames N`, `--fixed-chunks` | streaming chunk size in 80 ms frames (0 = decode once at the end); fixed sizes instead of 1, 2, 4, ... |
| `--language L` | a codec language of the checkpoint or `auto`; default: the checkpoint's (`english` for Lure, `auto` for stock) |
| `--layout default\|non_streaming\|streaming` | prompt layout; default: the checkpoint's |
| `dump-prefill ... [--full] [--ref-frames N]` | print the prompt recipe as JSON without generating; without `--full` no weights are loaded and `--ref-text` stands for an `icl` voice of N frames |

Turbo selection (`--options` or the Python `turbo=` argument): `turbo=auto` (default) runs Turbo when the model has
`turbo.json`, `turbo=1` requires it, `turbo=0` uses the 15-pass code predictor and prints a warning on a Turbo model,
since the Turbo talker is matched to its own head. `turbo_head=PATH` loads another head file with the same `turbo.json`
settings.

## HTTP server

```bash
$B/lure_cli serve --model /path/to/lure --host 127.0.0.1 --port 8088 --voice-cache ~/.cache/lure-cpp
```

Plain HTTP, no TLS. This server is lower level than `lure serve` (see [api.md](../api.md)): it takes one segment per
request and streams raw PCM.

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | | status |
| `GET /v1/model` | | variant, speakers, languages, modes, defaults |
| `POST /v1/voices` | `{"ref_wav": path}` or `{"ref_pcm_f32_b64", "sample_rate"}`, plus `"ref_text"`? (Base) | `{"voice_id", "frames", "transcript"}` |
| `POST /v1/generate` | `{"text", "voice"?, "speaker"?, "instruct"?, "language"?, "layout"?, "x_vector_only"?, "seed"?, sampling fields?, "max_new_tokens"?, "chunk_frames"?, "chunk_grow"?}` | `application/x-lure-pcm` stream |
| `POST /v1/prefill` | same as generate | prompt recipe JSON |

`voice` (alias `voice_id`) names a voice made with `/v1/voices` (`icl`, or `xvec` for a voice without a transcript or
with `x_vector_only`) or a preset speaker of a CustomVoice checkpoint (`custom_voice`). Without `voice` and `speaker` the
request runs `voice_design`, where `instruct` describes the voice.

The PCM stream is a sequence of `[u32 LE n][n x f32 LE]` chunks at 24 kHz, then `[u32 0]` followed by one JSON object
with frames, `ended_by_eos` and timings, or `{"error"}`. Closing the connection cancels generation.

## C API

Header: `backends/cpp/include/lure_tts.h` (ABI version 2). The library exports only `lure_*` symbols. An
`lure_tts` handle is not thread safe; serialize calls per handle.

```c
#include "lure_tts.h"
#include <stdio.h>

static int on_pcm(const float * pcm, size_t n, int32_t sample_rate, void * user) {
    fwrite(pcm, sizeof(float), n, (FILE *) user);   /* mono float32 at 24 kHz */
    return 1;                                        /* 0 cancels */
}

int main(void) {
    lure_tts * tts = lure_load("/path/to/lure", "auto", 0, 0, NULL);
    if (!tts) { fprintf(stderr, "%s\n", lure_last_error()); return 1; }
    lure_voice * voice = lure_voice_create_from_wav(tts, "ref.wav", "Transcript of ref.wav.");
    lure_request request;
    lure_default_request(&request);
    request.mode = LURE_MODE_ICL;        /* or XVEC, CUSTOM_VOICE (+ speaker), VOICE_DESIGN */
    request.text = "Hello there.";
    request.instruct = "Warm and calm.";
    request.voice = voice;
    lure_params params;
    lure_default_params(&params);
    params.has_seed = 1;
    params.seed = 1234;
    lure_gen_info info;
    FILE * out = fopen("out.f32", "wb");
    if (lure_generate(tts, &request, &params, on_pcm, out, &info) != 0)
        fprintf(stderr, "%s\n", lure_last_error());
    fprintf(stderr, "%lld frames, eos=%d, first chunk %.0f ms\n",
            (long long) info.frames, info.ended_by_eos, info.first_chunk_ms);
    fclose(out);
    lure_voice_free(voice);
    lure_free(tts);
    return 0;
}
```

Link against `liblure_tts` from `build/<preset>/bin`. `lure_request` carries the mode, the layout
(`LURE_LAYOUT_DEFAULT`, `_NON_STREAMING`, `_STREAMING`), the language (NULL = the checkpoint's), the instruct, and the
voice or the preset speaker (`speaker` name or `speaker_id`). `lure_variant`, `lure_speaker_count`,
`lure_speaker_name`, `lure_speaker_id` and `lure_model_info` (JSON; `lure_model_info_offline` reads only the
config and tokenizer) describe the loaded checkpoint. A voice made with a NULL or empty transcript is x-vector only
(`lure_voice_has_transcript` returns 0) and runs in `xvec` mode. Other calls: `lure_voice_cached` (disk cache keyed by audio,
transcript and model), `lure_voice_save` / `lure_voice_load` (`.alv` files), `lure_voice_from_codes` with
`lure_voice_codes` / `lure_voice_speaker` (exchange voices with the core without running the encoders),
`lure_dump_prefill` and `lure_dump_prefill_offline` (prompt recipe JSON; the offline form reads only configs and
the tokenizer), and `lure_tokenize_offline`.

`lure_load` options are `key=value` pairs separated by `;`. Weight storage can be set with
`qwen3_tts.weight_type` (`native`, `f32`, `f16`, `bf16`, `q8_0`); only the default bf16 weights have been measured.

## Python

Build the library first, then from the repository root:

```bash
pip install ".[server]"
pip install -e ./backends/cpp/python        # editable: finds build/<preset>/bin/ in the source tree
export LURE_CPP_LIB=$PWD/backends/cpp/build/macos-metal/bin/liblure_tts.dylib   # needed for a non-editable install
lure backends                              # should list cpp
```

```python
from lure_tts import Lure

tts = Lure.load("/path/to/lure", backend="cpp", ggml_backend="metal")   # or cuda, vulkan, cpu, auto
```

Backend options: `ggml_backend` (or `backend` when constructing `CppBackend` directly), `device`, `threads`, `quant`
(sets `qwen3_tts.weight_type`), `options` (raw `k=v;k=v` string), `turbo` (`auto`, `1`, `0`), `turbo_head`, `lib_path`.
From the CLI: `lure say ... --backend cpp --opt ggml_backend=cuda`. The binding passes the core's mode, layout, language,
speaker and instruct to the library, and implements the optional `dump_layout()` used by the conformance prompt check.
Before each segment it checks that the library's prompt ids equal the core's `PromptIds`; `LURE_CPP_CHECK_IDS=0` turns that off. Up to 16 voices stay
loaded in the library.

## Tests

No model weights are needed:

```bash
cd backends/cpp
build/macos-metal/bin/lure_turbo_head_test ../../conformance/turbo/fixtures/tiny_turbo.safetensors cpu    # or metal, cuda, vulkan
build/macos-metal/bin/lure_qwen3_variants_test ../../conformance/reference
PYTHONPATH=python python -m pytest python/tests
L=build/macos-metal/bin/liblure_tts.dylib
python tools/lure/conformance_dump.py --lib $L --out /tmp/cpp_prompt_ids.json
python ../../conformance/check_prompt.py /tmp/cpp_prompt_ids.json
python tools/lure/conformance_dump.py --lib $L --reference ../../conformance/reference/prompt_modes.json \
    --out /tmp/cpp_prompt_modes.json
python ../../conformance/check_prompt.py /tmp/cpp_prompt_modes.json --reference ../../conformance/reference/prompt_modes.json
```

The Turbo head test compares logits and greedy codes against the reference fixture: relative error about 5e-7 on CPU
and 1.4e-3 on Metal, where ggml stages some matrix tiles in half precision (tolerance 3e-3). `lure_qwen3_variants_test`
loads each stock config from `conformance/reference/configs` and checks the expected tensor shapes (1.7B projection,
speaker encoder on Base) against a shape-only tensor source. `conformance_dump.py` reads only configs and the tokenizer:
`prompt_ids.json` entries use `conformance/reference/tokenizer` (or `--model-dir`), and each `prompt_modes.json` entry
uses its own stock config. The binding tests cover the struct layouts, model info and mode errors for every variant. For the full conformance run with audio, see [conformance.md](../conformance.md).

## Performance

Measured with the conformance runner: 13 cases x 2 takes, Qwen3-ASR-1.7B WER, ECAPA similarity to the voice
prompt (qwen-tts reference: WER 0.0150, ECAPA 0.575; gate ECAPA >= 0.545). RTF below 1 is faster than real time; TTFA is
the median time to first audio through the Python binding. L4 and CPU ran in cloud containers (CPU: 16 vCPUs).
These numbers were measured with an earlier reference clip; the clip now bundled in `conformance/assets/` is a synthetic voice.

| Build | Hardware | Model | WER | ECAPA | RTF streaming | RTF non-streaming | TTFA |
|---|---|---|---|---|---|---|---|
| CUDA | NVIDIA L4 | Lure | 0.0150 | 0.562 | 0.51 | 0.28 | 0.175 s |
| CUDA | NVIDIA L4 | Turbo | 0.0001 | 0.566 | 0.40 | 0.17 | 0.139 s |
| CPU | 16 vCPU | Turbo | 0.0141 | 0.579 | 10.0 | | 3.78 s |
| Metal | Apple M5, 16 GB | Lure | 0.0112 | 0.527 | 1.05 short, 0.92 long | | 0.36 s |

Same-build CPU comparison on 16 vCPUs (4 short cases, 1 take):

| | Lure | Turbo |
|---|---|---|
| RTF streaming | 12.66 | 10.26 |
| RTF non-streaming | 6.72 | 3.70 |

An older CPU build with 25 frames of stream context measured RTF 6.64 (short) / 6.26 (long) and TTFA 2.36 s for Lure.
Metal peak memory was 4.04 GB. Turbo on Metal has not been measured.

## Known limitations

- CPU is slow: RTF 10 streaming on 16 vCPUs even with Turbo. Fine for batch jobs, not for live use. Streaming costs
  more than non-streaming on CPU because each chunk decodes 72 frames of context.
- Metal streams slower than real time on an Apple M5 (RTF 1.05 on short clips). Its ECAPA score, 0.527, is below the
  0.545 gate; WER is fine and the speaker embedding of the shared voice matches (cosine 1.0). The mlx backend shows the
  same gap on the same machine. The cause is not found yet. The Metal stream-join check has not run.
- CPU output differs between machines. With the portable `linux-cpu` build the same seed gave different frame counts
  on 15 of 16 clips across two hosts. The likely cause is that ggml picks a different CPU kernel variant per host;
  this is not confirmed. GPU builds were bit-identical across reruns.
- Each frame runs 16 small graphs (talker step plus 15 code-predictor steps, or 15 with Turbo), each with a host sync
  for sampling. Moving sampling onto the GPU would remove most of those syncs.
- Vulkan, Windows, RKNN and quantized weights are untested.
