# Prompting

A request has three parts: the text, an optional direction (instruct), and a voice. What counts as a voice depends on
the checkpoint: a cloned voice on Base models and the Lure family, a preset speaker on CustomVoice, and the direction
itself on VoiceDesign. `lure info MODEL` shows which applies.

## Inline tags

Inline tags are a Lure-family feature (`inline_tags` is on in its `lure.json`). On stock Qwen3-TTS checkpoints
parentheses are ordinary text and the segmenter treats them like any other characters. Force either behaviour with
`inline_tags=` on `Lure.load` or `--inline-tags` / `--no-inline-tags`.

With the Lure family, write non-speech events and delivery cues in parentheses, inside the text:

```
Wait, wait, wait. You're telling me the whole thing was a joke? (laugh) Unbelievable.
I opened the box and (gasp) it was empty. Completely empty.
(nervous laugh) So, um, about the car. There's a tiny scratch.
```

Tags are plain text to the core. Nothing strips or rewrites them: the model sees the parentheses exactly as written
and performs the tag, or stays silent, instead of saying the word. The conformance suite checks this on every backend
with `(laugh)`, `(chuckle)`, `(sigh)`, `(gasp)`, `(nervous laugh)` and `(sharp inhale)`, and found 0 tags read aloud.
Other tags may work, but those six are the ones measured.

Tips:

- Put the tag where the sound happens, between words or sentences.
- A tag on its own does nothing useful. The segmenter never sends a lone tag as a segment; it waits for the words that
  follow.
- Keep parentheses balanced. An unclosed `(` stops the segmenter from cutting until the text ends.

## Directions

A direction is a short note about delivery for the whole request. The Lure family and 1.7B CustomVoice take one
optionally, 1.7B VoiceDesign requires one (it describes the voice), and stock Base and 0.6B CustomVoice reject one, as
qwen-tts does.

```python
tts.speak("Grab the wrench, grab it, grab it!", voice, instruct="Frantic panic, breathless.")
```

CLI: `--instruct` (or `--direction`). Server: `instructions`. The direction goes into the prompt as a user turn before
the assistant turn, `<|im_start|>user\n{direction}<|im_end|>\n`, and applies to every segment of the request.

With Lure, a direction changes the audio on every backend. In the conformance runs, "Frantic panic, breathless."
made the line shorter on most backends and higher pitched on most, but with 2 takes per case the effect was not
consistent across backends. Treat directions as a nudge and listen to a few seeds. Short, concrete descriptions of
emotion and pace ("Tired and resigned, then a reluctant smile.") are what was tested.

There is no speed parameter; the server rejects `speed` other than 1.0. Ask for pace in the direction.

## Long text

Pass any amount of text to `speak`, `stream` or the server. The core cuts it into segments and generates them one after
another with the same cached voice prompt, so long text never builds a huge prompt and memory stays flat.

How text is cut (full text known, `speak` / `stream`):

- Whole sentences are packed together up to 25 spoken words per segment (tags do not count as words). Fewer cuts give
  better prosody.
- A sentence longer than 30 words is cut at its last clause boundary (`,` `;` `:` em dash, ellipsis) within the limit,
  or at a word boundary if there is none.
- Never inside `( ... )`, never right after an abbreviation such as `Dr.`, never inside a number such as `3.5` or `3,000`.
- Never across a blank line: a paragraph break always ends a segment.

Between segments the scheduler inserts a short silence: 0.25 s after a sentence, 0.1 s after a clause, 0.45 s after a
paragraph. Change it with `pauses={...}`. Segment `i` uses `seed + i`, so a seeded long render is reproducible.

Incremental text (`tts.live`, for LLM output) uses the same rules, except that the first segment is cut at the earliest
clause boundary with at least 4 words, so audio starts sooner.

`lure prompt "your text" --segment` shows the segments without loading weights. In Python:
`[s.text for s in lure_tts.segment_text(text)]`.

## Voices

On Base checkpoints and the Lure family the voice prompt is a reference clip plus its exact transcript (in-context
cloning, mode `icl`). Use clean speech with no music or
background noise; the clip in the conformance suite is about 8 seconds long. The transcript must match
what is said: it is part of the prompt.

Cache a voice once (`make_voice(..., name=...)`, `lure voice add`, or `POST /v1/voices`) and refer to it by name. A
cached voice holds the reference codec frames and the speaker embedding, so later requests skip the encoders. It is
re-encoded automatically if the clip, transcript or model changes.

Without a transcript (`make_voice(clip)` with no `ref_text`, `--x-vector-only` on the CLI, or an inline server voice
without `ref_text`) the voice holds only the speaker embedding and the model runs in mode `xvec`. It needs no
transcript and usually resembles the speaker less closely.

## Speakers

CustomVoice checkpoints have preset speakers instead of cloning. Pass one with `speaker=` in Python, `--speaker` on
the CLI, or as `voice` in a server request; `lure info MODEL` and `GET /v1/voices` list them. Some speakers are
dialect speakers: with language `chinese` or `auto` they use their dialect's language id, as qwen-tts does.

## Languages

`language=` takes `auto` or any language the checkpoint lists (`lure info MODEL`). Stock checkpoints default to
`auto`, the qwen-tts default, and the Lure family defaults to `english`.

## Sampling

Defaults, used by every backend:

| Parameter | Default | Applies to |
|---|---|---|
| `temperature` | 0.9 | codebook 0 (talker) |
| `top_k` | 50 | codebook 0 |
| `top_p` | 1.0 (off) | codebook 0 |
| `repetition_penalty` | 1.05 | codebook 0, over every earlier codebook-0 token of the segment |
| `subtalker_temperature` / `top_k` / `top_p` | 0.9 / 50 / 1.0 | codebooks 1-15, and every Turbo head position |
| `min_new_tokens` | 2 | EOS is blocked for the first 2 frames |

Why these values:

- They come from the checkpoint's `generation_config.json` (read at load time) and match the defaults of the qwen-tts reference implementation, and
  the conformance reference was rendered with them. Changing them moves you away from the measured quality.
- `repetition_penalty` 1.05 is mild and has no window. Some Qwen3-TTS runtimes force 1.5 over the last 64 tokens; Lure
  TTS always uses the value you pass.
- `max_new_tokens` defaults to a per-segment cap of `12.5 * (1.8 * words / 2.6 + 4)` frames, about 1.8x the expected
  length plus 4 s. A segment that hits it reports `ended_by_eos=False`.

Override per request: `sampling={"temperature": 0.7}` in Python or the server body, `--temperature 0.7` on the CLI.
