# Architecture

```
  Lure API / lure CLI / OpenAI-compatible server            core/lure_tts/api.py, cli.py, server.py
                         |
  model inspection: folder -> ModelSpec                         core/lure_tts/model.py
  segmenter: text -> segments                                   core/lure_tts/text.py
  prompt builder: segment -> PromptIds -> Prompt                core/lure_tts/prompt.py, tokenizer.py
  voice cache: clip (+ transcript) -> Voice (codes, x-vector)   core/lure_tts/voice.py
  scheduler: one GenerateRequest per segment, worker thread,    core/lure_tts/scheduler.py
             pauses, back-pressure, cancellation
                         |
            Backend.generate(GenerateRequest) -> Chunk...       core/lure_tts/backends/base.py
       +-----------------+------------------+
  backends/mlx      backends/cuda       backends/cpp (ctypes -> liblure_tts)
```

The core decides everything the model sees: which kind of checkpoint it is, where text is cut, the token ids, the prompt
mode and layout, the voice prompt, the sampling parameters, the per-segment seed and the frame cap. A backend only turns
one prompt into audio, so the backends can differ only in numerics.

## Supported checkpoints

Lure TTS runs Qwen3-TTS 12Hz checkpoints in the Hugging Face layout. `lure_tts.inspect(model_dir)` reads
`config.json`, `generation_config.json` and the optional `lure.json` and `turbo.json`, and returns a `ModelSpec`.
`lure info MODEL` prints it.

| Checkpoint | `variant` | Modes | Default layout | Inline tags | Instruct |
|---|---|---|---|---|---|
| Qwen3-TTS-12Hz-0.6B-Base, 1.7B-Base | `base` | `icl`, `xvec` | `streaming` | no | none |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | `custom_voice` | `custom_voice` | `non_streaming` | no | none |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | `custom_voice` | `custom_voice` | `non_streaming` | no | optional |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | `voice_design` | `voice_design` | `non_streaming` | no | required |
| Lure, Lure Turbo | `base` | `icl`, `xvec` | `non_streaming` | yes | optional |

The stock defaults match qwen-tts 0.1.1: `generate_voice_clone` uses the streaming layout, `generate_custom_voice` and
`generate_voice_design` use the non-streaming one, `generate_voice_clone` takes no instruct, and
`generate_custom_voice` drops the instruct on 0.6B. There is no 0.6B VoiceDesign checkpoint.

The Lure family has the same `config.json` as Qwen3-TTS-12Hz-0.6B-Base, so the folder marks itself with `lure.json`:

```json
{"format": "lure-model-v1", "family": "lure", "prompt_layout": "non_streaming", "inline_tags": true, "name": "Lure"}
```

`name` is optional. A folder with `turbo.json` and no `lure.json` also counts as the Lure family. Anything else is
stock. `layout=` and `inline_tags=` on `Lure.load` (`--layout`, `--inline-tags` / `--no-inline-tags` on the CLI)
override the defaults.

### ModelSpec

```python
@dataclass(frozen=True)
class ModelSpec:
    model_dir: str
    name: str                        # "Qwen3-TTS-12Hz-1.7B-CustomVoice", "Lure Turbo", ...
    family: str                      # "qwen3_tts" | "lure"
    variant: str                     # "base" | "custom_voice" | "voice_design" (config tts_model_type)
    size: str                        # "0b6" | "1b7" (config tts_model_size); size_label gives "0.6B" | "1.7B"
    talker_hidden_size: int          # 1024 | 2048
    code_predictor_hidden_size: int  # 1024 on both sizes
    speaker_encoder_dim: int | None  # Base only: x-vector size (1024 | 2048)
    num_code_groups: int             # 16
    speakers: tuple                  # preset speakers, lowercase, sorted (CustomVoice)
    speaker_dialects: dict           # dialect speakers only: name -> dialect language key
    languages: tuple                 # what language= accepts: "auto" plus the non-dialect codec language keys
    default_language: str            # "auto" for stock, "english" for the Lure family
    modes: tuple                     # prompt modes this variant runs
    layout: str                      # "non_streaming" | "streaming"
    inline_tags: bool                # parentheses are event tags (True) or plain text (False)
    instruct: str                    # "none" | "optional" | "required"
    sampling: Sampling               # generation_config.json over the qwen-tts defaults
    max_new_tokens: int              # generation_config.json cap
    turbo: TurboConfig | None        # lure_tts.turbo.resolve
    ids: ModelIds                    # special ids, language ids, speaker ids and dialects from config.json
    lure_json: dict
```

`inspect` rejects anything other than the 12Hz speech tokenizer with 16 code groups, and a `lure.json` on a
checkpoint that is not Base.

## Choosing the mode

The API picks the mode from what the caller passes (`Lure.resolve_mode`):

| Variant | `voice=` | `speaker=` | `instruct=` | Mode |
|---|---|---|---|---|
| Base | a voice with a transcript | | per `spec.instruct` | `icl` |
| Base | a voice made without a transcript (`ref_text` empty) | | per `spec.instruct` | `xvec` |
| CustomVoice | | a preset name, required | optional on 1.7B, rejected on 0.6B | `custom_voice` |
| VoiceDesign | | | required | `voice_design` |

Any other combination raises `ModeError` (a `ValueError`) before a backend is called, with a message that names the fix:
a voice and a speaker together, a speaker on Base, a voice on CustomVoice or VoiceDesign, an unknown speaker, a missing
voice, a missing or unsupported instruct, or a language the model does not list.

## The prompt contract

`lure_tts.prompt.build_prompt` is the normative prompt. It reproduces the qwen-tts 0.1.1 prefill for every mode and
layout, and `conformance/reference/` holds dumps of the qwen-tts code run on stub models with symbolic embeddings
(no weights), which the tests compare row for row.

Text is passed exactly as written, tags included. The core's tokenizer produces the ids:

```
input_ids    = tok("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
ref_ids      = tok("<|im_start|>assistant\n{ref_text}<|im_end|>\n")    # icl only, else None
instruct_ids = tok("<|im_start|>user\n{instruct}<|im_end|>\n")         # None when there is no instruct
```

### Rows

A prefill row is `[text, codec]` and its embedding is `T(text) + C(codec)`. Either side may be `None` (adds 0).

- `T(id)` = `talker.text_projection(talker.model.text_embedding(id))`
- `C(id)` = `talker.model.codec_embedding(id)`, codec vocab 3072
- `"spk"` = the speaker-encoder x-vector of the reference clip (`enc_dim` = talker hidden size), cast to the talker dtype
- `"ref:t"` = `C(ref_codes[t,0]) + sum_{g=1..15} code_predictor.model.codec_embedding[g-1](ref_codes[t,g])`
- a preset speaker is the plain codec row `C(spk_id)`

### Prefill

Every prompt starts the same way:

1. instruct: `[id, None]` for each instruct id (absent without an instruct)
2. role: `[id, None]` for `input_ids[:3]`
3. prefix, text side `tts_pad` except the last row:
   `[pad, think] [pad, think_bos] [pad, lang] [pad, think_eos] [pad, speaker] [tts_bos, codec_pad]`

The language row: `language="auto"` gives `[pad, nothink] [pad, think_bos] [pad, think_eos]` (3 rows, no language id).
A dialect speaker (`spk_is_dialect`) with language `chinese` or `auto` uses its dialect id, so `auto` turns on the 4
think rows for that speaker. The speaker row is `"spk"` in `icl` and `xvec`, `spk_id` in `custom_voice`, and absent in
`voice_design`.

Then, with `X = ref_ids[3:-2] + input_ids[3:-5] + [tts_eos]` and `Y = input_ids[3:-5] + [tts_eos]`:

| Mode | Layout | Rows after the prefix | Trailing |
|---|---|---|---|
| `icl` | `non_streaming` | `[x, codec_pad]` for x in X, then `[tts_pad, codec_bos]`, then `[tts_pad, "ref:t"]` per reference frame | `[tts_pad]` |
| `icl` | `streaming` | `K = [codec_bos, ref:0 .. ref:T-1]`; row i is `[X[i], K[i]]` for i < len(K), X padded with `tts_pad` | `X[len(K):]`, or `[tts_pad]` when X fits |
| other | `non_streaming` | `[y, codec_pad]` for y in Y, then `[tts_pad, codec_bos]` | `[tts_pad]` |
| other | `streaming` | one row `[input_ids[3], codec_bos]` | `input_ids[4:-5] + [tts_eos]` |

Position ids are 0..L-1 on all 3 mrope sections, with no padding and a causal mask.

### Generation

- Generated frame k (k = 0 is the frame sampled from the prefill) is fed back as
  `T(trailing[k]) + C(cb0) + sum_{g=1..15} CP_EMB[g-1](cb_g)`, with `T(tts_pad)` once `trailing` runs out.
- Codebook 0: ids `vocab-1024 .. vocab-1` except EOS are masked at every step, EOS is masked for the first
  `min_new_tokens = 2` frames, the repetition penalty covers every earlier cb0 id of the segment (unique set,
  `logit < 0 ? logit * p : logit / p`), then temperature, top-k, top-p. Generation stops when cb0 is EOS, and the EOS
  frame is dropped.
- Codebooks 1-15: the code predictor, 15 sequential steps per frame, sampled with the `subtalker_*` settings, with no
  penalty and no mask. The Turbo head replaces these steps ([turbo.md](turbo.md)).
- Sampling comes from `req.sampling`, which defaults to `spec.sampling`.

### Decode

- `icl`: decode `[ref_codes ; generated]` and drop the first `int(T_ref / (T_ref + T_gen) * n_samples)` samples. A
  streaming decode primes the decoder's left context with the tail of `ref_codes`.
- `xvec`, `custom_voice`, `voice_design`: decode the generated codes alone, with no reference context.
- Output: float32 mono PCM at 24 kHz.

### Why Lure uses the non-streaming layout

In the streaming layouts the text is overlaid on the audio, and the part that does not fit is fed one token per
generated frame. With that layout Lure reads tags such as `(chuckle)` aloud: it consumes the text in step with the
audio, so each tag lands where speech is expected. In the non-streaming layout the whole target text sits in the
prefill before the first audio frame, the per-frame text input is only `tts_pad`, and the tags are performed. The Lure
checkpoints expect this layout, and the conformance tag-read check verifies it.

The layout does not decide whether audio streams. Every layout streams by decoding the generated codec frames in small
chunks while generation continues.

## Backend contract

A backend is a class with this shape (`core/lure_tts/backends/base.py`):

```python
class MyBackend:
    name = "mine"
    sample_rate = 24000
    concurrent = False                      # True if generate() may run on several threads at once

    @classmethod
    def load(cls, model_dir, quant=None, *, spec, **opts): ...
    def encode_voice(self, ref_wav_24k, ref_text):    # Base only -> (codes int32 [T, 16], x-vector float32 [enc_dim])
        ...
    def generate(self, req):                # GenerateRequest -> iterator of Chunk(pcm); the last Chunk carries SegmentInfo
        ...
    def close(self): ...
    def dump_layout(self, text, ref_text, direction, n_ref_frames, language="english",
                    mode="icl", layout="non_streaming", speaker=None):   # optional, for the prompt check
        ...                                 # -> {"layout": prefill rows, "trailing": trailing rows}
```

`load` receives `spec=ModelSpec`, already resolved by the core, plus the backend options. Read `spec.variant`,
`spec.talker_hidden_size`, `spec.speaker_encoder_dim` and `spec.ids` instead of guessing from the weights. `turbo` and
`turbo_head` arrive in `**opts`, and `spec.turbo` holds the resolved `TurboConfig`.

`GenerateRequest` fields:

| Field | Meaning |
|---|---|
| `text` | the segment |
| `mode` | `icl`, `xvec`, `custom_voice` or `voice_design` |
| `layout` | `non_streaming` or `streaming` |
| `voice` | `Voice(ref_codes [T, 16], spk_emb [enc_dim], ref_text)` for `icl` and `xvec`, else `None`. An `xvec` voice has `ref_text == ""`. |
| `speaker`, `speaker_id` | preset name and its codec id, `custom_voice` only |
| `prompt` | the core's `PromptIds`; `prompt.ref_ids` is `None` outside `icl`, `prompt.instruct_ids` is `None` without an instruct |
| `instruct` | the instruct text (already in `prompt.instruct_ids`) |
| `language` | language key or `auto` |
| `sampling`, `seed`, `max_new_tokens` | sampling settings, per-segment seed and frame cap |
| `stream`, `chunk_frames`, `first_chunk_frames` | chunking; `stream=False` may yield one chunk |

`req.build_prompt(spec.ids)` returns the `Prompt` (`prefill`, `trailing`, `spans`) for the request. Embed those rows
rather than rebuilding the layout; tokenizing again is allowed only if it gives the same ids. `spans` names the blocks
(`instruct`, `role`, `prefix`, then `icl_text` / `icl_codec`, `icl_overlay`, `text` / `codec_bos` or `first_text`) for
backends that build the embedding in pieces.

Per variant, a backend must:

- Base, `icl`: put the x-vector in the `"spk"` row, embed `ref:t` from `voice.ref_codes`, and decode with the reference
  cut.
- Base, `xvec`: put the x-vector in the `"spk"` row, embed no reference frames, and decode the generated codes alone.
- CustomVoice: embed `C(req.speaker_id)` in the speaker row. There is no speaker encoder in these checkpoints, and
  `encode_voice` is never called.
- VoiceDesign: no speaker row, no speaker encoder.
- Every mode: feed `trailing[k]` as the text side of generated frame k. In both `non_streaming` layouts that is
  `tts_pad` throughout; in the streaming layouts it carries the rest of the text and then `tts_eos`.

### 1.7B shapes

Both sizes have 28 talker layers with 16 heads, 8 KV heads and head dim 128, and a 5-layer code predictor with hidden
size 1024. The 1.7B checkpoints differ here:

| Tensor | 0.6B | 1.7B |
|---|---|---|
| talker hidden / MLP intermediate | 1024 / 3072 | 2048 / 6144 |
| `text_embedding` | [151936, 2048] | [151936, 2048] |
| `text_projection` (2048 -> 2048 -> hidden) | output 1024 | output 2048 |
| `codec_embedding`, `codec_head` | hidden 1024 | hidden 2048 |
| code predictor `codec_embedding[g]` | [2048, 1024] | [2048, 2048] (talker space) |
| code predictor `small_to_mtp_projection` | identity (absent) | Linear 2048 -> 1024 with bias, applied to the code predictor inputs |
| speaker encoder `enc_dim` (Base) | 1024 | 2048 |

On 1.7B the code predictor's input sequence (`[talker_hidden_last, C(cb0)]`, then `CP_EMB[g-1](cb_g)`) lives in the
2048-d talker space and goes through `small_to_mtp_projection` before the 1024-d code predictor layers. The ref and
feedback sums use the 2048-d `CP_EMB` tables directly. The Turbo head exists only for the 0.6B Lure family.

### Registering

```toml
[project.entry-points."lure.backends"]
mine = "my_package:MyBackend"
```

Then `Lure.load(model, backend="mine")` works, and so do the CLI, server, benchmarks and conformance runner. A
backend is done when the prompt check passes for every mode its target models use and the conformance run passes the
gates ([conformance.md](conformance.md)).

## Segments and the voice prompt

Long text is cut by the core (rules in [prompting.md](prompting.md#long-text)) and every segment gets a fresh prefill
with the same cached voice: the reference codes and x-vector are computed once per voice and reused. Prompt length is
bounded by one segment plus the reference, so memory and per-frame cost stay flat on long inputs.

## Streaming

Backends emit float32 PCM at 24 kHz. The first chunk is `first_chunk_frames` frames (default 4, 0.32 s of audio) and
later chunks are `chunk_frames` (default 4). The codec decoder runs on windows with left context taken from earlier
frames (and, in `icl`, from the reference codes for the first chunk), so chunk joins match a full decode closely: 22 to
30 dB SNR on cuda with 72 frames of context, and an exact stateful decode on mlx. With `stream=False` each segment is
decoded whole, as qwen-tts does.

## Threading

By default every backend call runs on one worker thread, which mlx requires. A backend class that sets
`concurrent = True` (cuda) gets one producer thread per stream, so overlapping requests reach `generate()` at the same
time and the backend can batch them. Consumers read from a bounded queue: a slow client applies back-pressure, and
closing the iterator cancels generation at the next chunk.

## Backends

All three implement every mode, layout and size in this contract. Lure and Lure Turbo voice cloning (`icl`,
`non_streaming`, 0.6B) is verified on each with real weights by the conformance suite. The stock Qwen3-TTS paths are
tested against qwen-tts 0.1.1 on random-weight fixtures with the real configs (prompt rows and greedy codes).

One known difference: when a segment reaches `max_new_tokens` without EOS, qwen-tts 0.1.1 returns `max_new_tokens - 1`
frames and the backends return `max_new_tokens`. Parity tests compare the common prefix.

| Backend | Built on | Notes |
|---|---|---|
| [mlx](backends/mlx.md) | MLX, using model modules from mlx-audio | Apple silicon. Prefill built from the core's layout rows. |
| [cuda](backends/cuda.md) | PyTorch, CUDA graph recipe from faster-qwen3-tts | Continuous batching of up to 8 requests, KV-length graph buckets. |
| [cpp](backends/cpp.md) | ggml, from audio.cpp | Single library and CLI with Metal, CUDA, Vulkan and CPU builds; C API, own HTTP server, Python binding. |
