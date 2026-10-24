# Prompt reference

Every backend must feed the model this exact prefill. These dumps come from running the qwen-tts 0.1.1 prompt code on a
stub model with symbolic embeddings, so they need no weights. The core's builder (`core/lure_tts/prompt.py` and
`core/lure_tts/tokenizer.py`) matches every entry. The full contract for every mode is in
[docs/architecture.md](../../docs/architecture.md#the-prompt-contract); this page details the Lure `icl`
non-streaming case.

Regenerate (CPU, about 0.5 GB, no weights loaded, deps in `../requirements-ref.txt`):

```bash
python conformance/reference/gen_reference.py /path/to/lure                       # both files
python conformance/reference/gen_reference.py conformance/reference/tokenizer --only modes   # prompt_modes.json only
```

`--only lure` writes `prompt_ids.json` (and `tokenizer_ids.json`) and needs the model folder's tokenizer;
`--only modes` needs only the tokenizer files and `configs/`. `--out DIR` changes the output folder.

## Files

- `prompt_ids.json`: 35 entries. Each has `text`, `direction`, `ref_text`, `language`, the three id lists (`input_ids`,
  `ref_ids`, `instruct_ids`), the full prefill `layout` (one row per position), `trailing_text_hidden`, `tts_pad_embed`,
  the `talker_kwargs` qwen-tts passes to the talker, and the codebook-0 suppress range. Entries: conformance cases a-d
  (plus `.nodir` variants), the passage `e` already cut by the core segmenter (`e.seg00` to `e.seg13`, what each backend
  receives), tag lines t1-t6, and edge cases `x.*` (unicode, numbers, newlines, added-token strings typed in the text,
  empty direction, language `auto`, a different reference transcript).
- `prompt_modes.json`: 40 entries covering every mode and layout (`icl`, `xvec`, `custom_voice`, `voice_design`,
  each `non_streaming` and `streaming`), made through the public qwen-tts wrappers (`generate_voice_clone`,
  `generate_custom_voice`, `generate_voice_design`, plus `model.generate` directly for a Base call with an instruct).
  Each entry records the `config` it used, the `call`, `mode`, `prompt_layout`, `text`, `language`, `ref_text`,
  `speaker`, the requested and effective `direction`, the three id lists, `n_ref_frames`, the prefill `layout` and the
  `trailing` rows. They cover preset speakers, dialect speakers with `auto` and `chinese`, short texts in the streaming
  layout and the 0.6B CustomVoice instruct drop.
- `configs/`: the stock `config.json` of Qwen3-TTS-12Hz 0.6B and 1.7B Base, 0.6B and 1.7B CustomVoice, and 1.7B
  VoiceDesign.
- `tokenizer_ids.json`: raw strings and their ids from the qwen-tts processor (`AutoProcessor`, `fix_mistral_regex=True`).
- `tokenizer/`: copies of the model's `vocab.json`, `merges.txt`, `tokenizer_config.json`, `config.json`,
  `generation_config.json` and `lure.json`, so prompt checks need no checkpoint. The stock checkpoints share this
  tokenizer.
- `gen_reference.py`: the generator.

## The call being reproduced (`prompt_ids.json`)

```
input_ids    = tok("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n")
ref_ids      = tok("<|im_start|>assistant\n{ref_text}<|im_end|>\n")
instruct_ids = tok("<|im_start|>user\n{direction}<|im_end|>\n")   # None when the direction is None or empty
model.generate(input_ids, instruct_ids, ref_ids, voice_clone_prompt(ICL: ref codes + speaker embedding),
               languages=["English"], non_streaming_mode=True, **generate_defaults)
```

Every `<|...|>` and `<tts_...>` string is an added token: always one id, never BPE-split, even when a user types it in
the text (`x.specialish`). No BOS or EOS is added. Text is NFC-normalised before BPE. Tags like `(laugh)` are ordinary
text.

## Row format

`[text, codec]`: the prefill embedding at that position is `T(text) + C(codec)`, and either side may be `null` (adds 0).

- `T(id)` = `talker.text_projection(talker.model.text_embedding(id))`
- `C(id)` = `talker.model.codec_embedding(id)` (codec vocab 3072)
- `"spk"` = the speaker-encoder embedding of the reference clip (1024-d), cast to the talker dtype, not passed through any layer
- `"ref:t"` = `C(ref_codes[t,0]) + sum_{g=1..15} code_predictor.model.codec_embedding[g-1](ref_codes[t,g])`

The dump uses 6 symbolic reference frames (`ref:0` to `ref:5`). A real voice has one row per 12.5 Hz codec frame of the
reference clip; everything before the `ref:` rows is independent of the clip length.

## Layout (ICL clone, English, with a direction)

| Block | Rows | Content |
|---|---|---|
| instruct | len(instruct_ids) | `[id, null]` (absent without a direction) |
| role | 3 | `[151644,null] [77091,null] [198,null]` (= input_ids[:3]) |
| prefix | 6 | `[pad,2154 think] [pad,2156 think_bos] [pad,2050 english] [pad,2157 think_eos] [pad,"spk"] [151672 tts_bos, 2148 codec_pad]` |
| icl text | len(ref_ids[3:-2]) + len(input_ids[3:-5]) + 1 | `[id, 2148]` for the reference transcript ids, then the target text ids, then `[151673 tts_eos, 2148]` |
| icl codec | 1 + T_ref | `[151671 tts_pad, 2149 codec_bos]`, then `[151671, "ref:t"]` |

`pad` = 151671 (`<tts_pad>`). Language `auto` uses `[2155 nothink, 2156, 2157]` (3 rows) instead of the 4 think rows.
Position ids are 0..L-1 on all 3 mrope sections, with no padding and a causal mask.

The streaming (overlay) ICL layout, where text and reference codes are summed position by position and the text
continues in `trailing_text_hidden`, is wrong for Lure: with it the model reads tags such as `(chuckle)` aloud.

## Generation contract

- `trailing_text_hidden = [T(151671)]`, so every generated frame feeds back `T(tts_pad) + C(cb0) + sum_{g=1..15} CP_EMB[g-1](cb_g)`.
- Codebook 0: ids 2048..3071 except EOS 2150 are set to -inf at every step; EOS is also masked for the first
  `min_new_tokens = 2` frames; repetition penalty 1.05 over all previously generated cb0 ids of the segment (unique set,
  `logit < 0 ? logit * p : logit / p`, no window); then temperature 0.9, top_k 50, top_p 1.0, multinomial. Generation
  stops when cb0 is 2150, and the EOS frame is not kept.
- Codebooks 1-15 (code predictor, 15 sequential steps per frame, input `[talker_hidden_last, C(cb0)]` then
  `CP_EMB[g-1](cb_g)`): temperature 0.9, top_k 50, top_p 1.0, no repetition penalty, no suppression. The Turbo head
  replaces these steps; see [docs/turbo.md](../../docs/turbo.md).
- Per-segment cap: `max_new_tokens = int(12.5 * (1.8 * words / 2.6 + 4))`, passed by the core.
- Decode: run the codec decoder on `[ref_codes ; generated]` and drop the first `int(T_ref / (T_ref + T_gen) * n_samples)`
  samples. Streaming backends decode chunks of about 4 frames with the decoder's left context primed from the tail of
  `ref_codes` and emit only the new samples (cuda and cpp use a 72-frame left context; mlx primes a stateful decoder with
  the reference frames).
- Output: float32 mono PCM at 24 kHz.

## Checking a backend

1. Token ids: tokenize every `tokenizer_ids.json` string and every `prompt_ids.json` template and compare. Backends that
   take the core's `PromptIds` (recommended) get this for free.
2. Layout: for each `prompt_ids.json` entry, emit your prefill recipe with a 6-frame fake reference as JSON rows, then run
   `python conformance/check_prompt.py your_dump.json`. It compares everything before the `ref:` rows exactly and checks
   that the `ref:` rows run 0..T-1 in order. `prompt_modes.json` entries (any mode other than `icl` non-streaming) are
   compared exactly, including `trailing` when the dump has it. Pass `--reference conformance/reference/prompt_modes.json` for those. Dump format:
   `{"backend": "name", "entries": [{"id": "c", "input_ids": [...], "ref_ids": [...], "instruct_ids": [...] or null, "layout": [[151644, null], ...], "n_ref_frames": 6}]}`.
   A backend that implements `dump_layout()` gets this check from `runner.py` automatically (`prompt_check.json`).
3. `lure prompt "text" --direction "..." --json` prints any prompt from the tokenizer files alone.

Audio is compared in distribution (WER, ECAPA and tag-read against the reference run), never bit for bit: the reference
ran with fp32 weights under bf16 autocast, and backends run bf16 with their own kernels and random streams.
