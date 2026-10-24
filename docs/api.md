# API reference

- [Python API](#python-api)
- [CLI](#cli)
- [Server](#server)
- [Backend options](#backend-options)

## Python API

```python
from lure_tts import Lure, Sampling
```

### Loading

```python
tts = Lure.load(model_dir=None, backend="auto", quant=None, cache_dir=None, layout=None, inline_tags=None,
                  **backend_opts)
print(tts.spec.variant, tts.spec.speakers, tts.spec.layout)
```

| Argument | Meaning |
|---|---|
| `model_dir` | Model folder. Defaults to `$LURE_MODEL`. |
| `backend` | `"auto"`, `"mlx"`, `"cuda"`, `"cpp"`, `"fake"` (test backend, no model), or `"package.module:Class"`. `auto` honours `$LURE_BACKEND`, then picks mlx on Apple silicon, cuda if PyTorch sees a GPU, else cpp. |
| `quant` | Backend-specific weight quantization; `None` means bf16. See the backend docs. |
| `cache_dir` | Voice cache root. Defaults to `$LURE_CACHE` or `~/.cache/lure`. |
| `layout` | `"non_streaming"` or `"streaming"`, overriding the model's default prompt layout. |
| `inline_tags` | `True` or `False`, overriding whether parentheses are event tags. |
| `**backend_opts` | Passed to the backend, for example `turbo=0` or `compile_cp=True`. See [Backend options](#backend-options). |

`Lure` is a context manager; `close()` releases the backend. Useful attributes: `spec` (the `ModelSpec`, see
[architecture.md](architecture.md#modelspec)), `backend_name`, `sample_rate`
(24000), `turbo` (True when the Turbo head is active), `concurrent` (True when the backend batches overlapping
requests), `voices` (the `VoiceCache`), `last_infos` (per-segment info of the last call).

`lure_tts.inspect(model_dir, layout=None, inline_tags=None)` returns the same `ModelSpec` without loading a backend.

### Voices

Base checkpoints and the Lure family only; other variants raise `ModeError`.

```python
voice = tts.make_voice(ref_wav, ref_text=None, name=None)   # -> Voice
voice = tts.voice("me")                                 # load a cached voice by id
tts.voices.list()                                       # cached voice ids
```

`ref_wav` is a path, raw file bytes, or an `(array, sample_rate)` tuple. `ref_text` is the transcript of the clip. Without
it the voice holds only the speaker embedding and synthesis runs in x-vector mode (`xvec`). The encoded voice
(reference codec frames `[T, 16]` and a speaker embedding of 1024 values on 0.6B, 2048 on 1.7B) is saved under
`<cache>/voices/<name>/`. Without a name the id is a hash of the audio and text. Calling `make_voice` again with the same
audio, transcript and model loads the cached copy instead of re-encoding. Every API that takes a voice also accepts its
id string.

### Synthesis

```python
for pcm in tts.stream(text, voice=None, instruct=None, sampling=None, seed=None, chunk_s=None, infos=None, *,
                      speaker=None, language=None, **opts):
    ...                                                 # np.float32 mono chunks at 24 kHz

audio = tts.speak(text, voice=None, instruct=None, sampling=None, seed=None, out=None, infos=None, *,
                  speaker=None, language=None, **opts)
```

What to pass depends on the checkpoint (`tts.spec.variant`):

| Variant | `voice` | `speaker` | `instruct` |
|---|---|---|---|
| Base, Lure family | required | not allowed | Lure family: optional; stock Base: not allowed |
| CustomVoice | not allowed | required, one of `tts.spec.speakers` | 1.7B: optional; 0.6B: not allowed |
| VoiceDesign | not allowed | not allowed | required |

A wrong combination raises `ModeError` (a `ValueError`) before anything is generated.

- `text`: any length. The core splits it into sentences (see [prompting.md](prompting.md#long-text)) and generates them
  one after another with the same voice prompt, inserting short pauses between them.
- `voice`: a `Voice` or a cached voice id.
- `speaker`: a preset speaker name, case-insensitive.
- `instruct`: the direction, a short free-text note on delivery (on VoiceDesign, the description of the voice). Empty
  or whitespace means none.
- `language`: `"auto"` or a language from `tts.spec.languages`. Defaults to `tts.spec.default_language` (`"auto"` on
  stock checkpoints, `"english"` on the Lure family).
- `sampling`: a `Sampling` or a dict of its fields (below). Defaults to `tts.spec.sampling`.
- `seed`: segment `i` uses `seed + i`. Seeds are reproducible within a backend on the same hardware, not across backends.
- `chunk_s`: streamed chunk length, default 0.32 s (4 codec frames).
- `infos`: pass a list to collect one `SegmentInfo` per segment for this call (safer than `last_infos` when calls
  overlap).
- `**opts`: `first_chunk_frames` (default 4; lower gives earlier first audio), `layout` (per-call layout override),
  `pauses` (seconds after each segment kind: `{"sentence": 0.25, "clause": 0.1, "paragraph": 0.45, "end": 0.0}`),
  `stream` (False decodes each segment whole).
- `speak` returns the whole clip and writes a 16-bit wav when `out` is set.

`SegmentInfo` fields: `frames`, `ended_by_eos` (False means the segment hit its frame cap), `samples`, `ttfa_s`,
`wall_s`, `extra` (backend details).

### Incremental text

```python
live = tts.live(voice=None, instruct=None, sampling=None, seed=None, *, speaker=None, language=None, **segmenter_kwargs)
for delta in llm_tokens:
    for pcm in live.feed(delta):
        play(pcm)
for pcm in live.flush():
    play(pcm)
```

`feed` returns audio as soon as the buffered text completes a segment. The first segment is cut at the earliest clause
boundary with at least 4 words, so audio starts early; later segments prefer whole sentences.

### Sampling

```python
Sampling(temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05,
         subtalker_temperature=0.9, subtalker_top_k=50, subtalker_top_p=1.0,
         do_sample=True, subtalker_do_sample=True, min_new_tokens=2, max_new_tokens=None)
```

These are the qwen-tts defaults; `tts.spec.sampling` applies the checkpoint's `generation_config.json` on top. The
`subtalker_*` fields apply to codebooks 1-15 (and to every position of the Turbo head). `max_new_tokens=None` uses a
per-segment cap of `12.5 * (1.8 * words / 2.6 + 4)` frames. Why these defaults: [prompting.md](prompting.md#sampling).

### Lower-level helpers

| Function | Use |
|---|---|
| `lure_tts.inspect(model_dir)` | The `ModelSpec` of a model folder, from its JSON files only. |
| `lure_tts.segment_text(text, tags=True)` | The segments `speak` / `stream` would generate (`tags=tts.spec.inline_tags`). |
| `lure_tts.Segmenter(mode="stream")` | The incremental segmenter behind `live`. |
| `lure_tts.build_prompt_ids(tokenizer, text, ref_text, direction)` | Token ids of one segment's prompt. |
| `lure_tts.build_prompt(prompt_ids, model_ids, mode=, layout=, language=, n_ref_frames=, speaker=)` | The `Prompt` (prefill and trailing rows) a backend must embed (see [architecture.md](architecture.md#the-prompt-contract)). |
| `lure_tts.build_layout(prompt_ids, model_ids, n_ref_frames)` | Prefill rows only, `icl` non-streaming by default. |

## CLI

Common flags: `--model DIR` (default `$LURE_MODEL`), `--backend auto|mlx|cuda|cpp|fake`, `--quant`, `--cache DIR`,
`--opt KEY=VALUE` (repeatable backend option), `--layout non_streaming|streaming`, `--inline-tags` /
`--no-inline-tags`.

### `lure say`

```bash
lure say TEXT [-o OUT.wav | -o - | --play]
           [--voice ID | --ref-wav WAV (--ref-text TEXT | --x-vector-only) [--save-voice ID] | --speaker NAME]
           [--instruct TEXT] [--language L] [--seed N] [--chunk-s S] [--stats]
           [--temperature T] [--top-k K] [--top-p P] [--repetition-penalty R]
           [--subtalker-temperature T] [--subtalker-top-k K] [--subtalker-top-p P] [--max-new-tokens N]
```

`TEXT` of `-` reads stdin. `-o -` writes raw 16-bit PCM (24 kHz mono) to stdout as it streams. `--play` needs the
`play` extra. `--direction` is an alias of `--instruct`. Pass one of `--voice`, `--ref-wav` or `--speaker`, or only
`--instruct` on a VoiceDesign checkpoint; the same rules as the Python API apply. The sampling flags override the
checkpoint's defaults.

### `lure voice`

```bash
lure voice add NAME REF.wav "transcript"     # encode and cache
lure voice list
```

### `lure serve`

```bash
lure serve [--host 127.0.0.1] [--port 8880]
```

### `lure info`

```bash
lure info [MODEL] [--json] [--layout L] [--inline-tags | --no-inline-tags]
```

Prints the `ModelSpec` of a model folder: family, variant and size, hidden sizes, modes, default layout, inline tags,
instruct rule, languages, speakers (with dialects), sampling defaults and Turbo. It reads only the JSON files.

### `lure prompt`

```bash
lure prompt TEXT [--mode M] [--layout L] [--speaker NAME] [--direction D] [--ref-text R] [--n-ref-frames N]
              [--language L] [--segment] [--json]
```

Prints the prompt ids, prefill layout and trailing rows for TEXT. Mode, layout and language default to the model's. It reads only the tokenizer files and `config.json`, never the weights.
`--segment` runs the segmenter first and prints one prompt per segment.

### `lure backends`

Lists the known backends, whether each is installed and usable on this machine, and what `auto` resolves to.

## Server

`lure serve` starts an HTTP server with the OpenAI speech API shape. Blocking work runs off the event loop. With a
batching backend (cuda), concurrent requests share one decode loop.

### `POST /v1/audio/speech`

| Field | Type | Default | Notes |
|---|---|---|---|
| `input` | string | required | Up to 20,000 characters. Tags stay in the text. |
| `voice` | string or object | see below | Base and Lure family: a cached voice id, or `{"ref_audio": "<base64 wav>", "ref_text": "...", "name": "optional"}` to clone inline (cached under `name` if given; without `ref_text` the clone is x-vector only). CustomVoice: a preset speaker name. VoiceDesign: optional and ignored. |
| `model` | string | | Accepted and ignored (one model per server). |
| `instructions` | string | | The direction (required on VoiceDesign). `instruct` is an alias. |
| `language` | string | model default | Extra field: `auto` or a language the model lists. |
| `response_format` | `wav`, `pcm`, `flac` | `wav` | `mp3`, `opus` and `aac` are rejected. |
| `stream_format` | `audio`, `sse` | `audio` | `sse` always streams. |
| `stream` | bool | `true` | `flac` is always returned whole. |
| `speed` | number | `1.0` | Only 1.0 is accepted; ask for pace in `instructions`. |
| `seed` | integer | | Extra field. |
| `sampling` | object | | Extra field: any `Sampling` field, range-checked. |

Responses:

- `wav`, streaming: a WAV header with open-ended sizes, then 16-bit PCM as it is generated.
- `pcm`: raw s16le, 24 kHz, mono.
- `sse`: `data: {"type": "speech.audio.delta", "audio": "<base64 pcm16>"}` events, then
  `data: {"type": "speech.audio.done", "samples": N, "sample_rate": 24000, "segments": [...]}`.
- Errors: HTTP 400 with `{"error": {"message": "...", "type": "invalid_request_error"}}`.

Closing the connection cancels generation at the next chunk.

### Other endpoints

| Endpoint | Returns |
|---|---|
| `GET /health` | `{"ok": true, "backend": "<name>"}` |
| `GET /v1/models` | one model entry, id `lure`, with the `ModelSpec` fields (variant, modes, layout, speakers, languages, ...) |
| `GET /v1/voices` | `{"kind": "clone", "voices": [cached ids], "speakers": []}` on Base, `{"kind": "speaker", "voices": [names], "speakers": [names]}` on CustomVoice, `{"kind": "none", ...}` on VoiceDesign |
| `POST /v1/voices` | Base only. Body `{"name", "ref_audio" (base64), "ref_text"}`; returns `{"voice", "n_ref_frames"}` |

The cpp backend also has its own server inside `lure_cli serve` with a different, lower-level protocol; see
[backends/cpp.md](backends/cpp.md).

## Backend options

Pass with `Lure.load(..., key=value)`, `--opt key=value`, or `--opt` on the conformance runner and benchmarks.
`--opt` values `1/0/true/false/on/off` become booleans and numbers become numbers.

| Option | Backends | Meaning |
|---|---|---|
| `turbo` | all | `auto` (default): Turbo if `turbo.json` exists. `1`: require it. `0`: original code predictor. See [turbo.md](turbo.md). |
| `turbo_head` | all | Path to another Turbo head file. |

Backend-specific options (batch size, CUDA graphs, MLX cache limits and compilation, ggml device and threads) are listed
in [backends/cuda.md](backends/cuda.md), [backends/mlx.md](backends/mlx.md) and [backends/cpp.md](backends/cpp.md).
