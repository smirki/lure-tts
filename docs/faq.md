# FAQ

**Which backend should I use?**
Mac with Apple silicon: mlx. Linux with an NVIDIA GPU: cuda, or the cpp CUDA build if you want a single binary or the
lowest time to first audio. Anything else: the cpp build for your GPU (Vulkan) or CPU. See the table in the
[README](../README.md#choosing-a-backend).

**Does it run on Windows?**
The cuda backend and the cpp presets (`windows-cpu`, `windows-cuda`, `windows-vulkan`) have Windows notes, but nothing has
been run on Windows yet. Reports are welcome.

**Will the same seed give the same audio on every backend?**
No. Seeds are reproducible within one backend on the same hardware. Across backends the kernels and random streams
differ, so the audio differs sample by sample while matching in distribution (that is what the conformance suite
checks). The cpp CPU build can also differ between machines with different CPUs, and cuda batches of 8 differ from
single requests on H100.

**Why does my tag get read aloud?**
With a Lure-family model it should not: no backend read a tag in the conformance runs. Check that the folder has
its `lure.json` (`lure info` should say `inline tags yes` and layout `non_streaming`) and that the parentheses are
balanced. If you drive the model with another runtime, check that it uses the non-streaming layout
([architecture.md](architecture.md#why-lure-uses-the-non-streaming-layout)). Stock Qwen3-TTS checkpoints have no
inline tags and read parentheses as text.

**Can I change the speaking speed?**
There is no speed parameter. Put the pace in the direction, for example `"Slow and calm."`.

**How long can the input be?**
Any length through the Python API; the core cuts it into segments. The server limits one request to 20,000
characters.

**Which languages work?**
The Lure family defaults to `language="english"` and has been tested only on English. Stock Qwen3-TTS checkpoints
default to `auto`. `lure info MODEL` lists the languages a checkpoint accepts, and `language` can be set per request,
but no language other than English has been measured here.

**Is there an fp16 or quantized path?**
Weights load in bf16. The mlx backend has experimental `quant="q8"` and `quant="mix"`, and the cpp backend passes
`quant` to ggml as the weight type; no quantized setting has been scored with the conformance suite. There is no fp16 path for GPUs without bf16
(such as the T4).

**The Mac uses a lot of memory.**
The mlx backend peaked at about 5 GB in the conformance run. On a 16 GB Mac, pass `--opt cache_limit_mb=128 --opt
memory_limit_gb=3.5` to cap the MLX cache.

**How do I add a backend?**
Implement the four-method interface and register an entry point; see
[architecture.md](architecture.md#backend-contract). Then run the prompt check and the conformance suite.

**Which models does it run?**
Qwen3-TTS 12Hz 0.6B and 1.7B (Base, CustomVoice, VoiceDesign) and the Lure family. Every backend runs all of
them; the stock checkpoints are tested on fixtures, not yet with real weights ([README](../README.md#backend-status)).

**Where do I get the model?**
From Hugging Face: Lure is [Tesslate/Lure-TTS-0.6B](https://huggingface.co/Tesslate/Lure-TTS-0.6B) and
Lure Turbo is [Tesslate/Lure-TTS-Turbo](https://huggingface.co/Tesslate/Lure-TTS-Turbo). Both are
0.6B. The stock Qwen3-TTS 12Hz checkpoints load from their own Hugging Face repositories.
