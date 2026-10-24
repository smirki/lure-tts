# Examples

Each script is short and runs as is once `LURE_MODEL` points at a model folder. They clone the conformance voice by
default; set `REF_WAV` and `REF_TEXT` to use your own clip.

| File | Shows |
|---|---|
| `stream_to_wav.py` | streaming chunks into a wav file as they arrive |
| `voice_cache.py` | encoding a voice once and reusing it by name |
| `direction_and_tags.py` | inline tags and a direction, with and without |
| `live_text.py` | feeding text word by word (LLM output) |
| `turbo_on_off.py` | Turbo auto-selection and `turbo=0` |
| `openai_client.py` | the server from the `openai` package and from plain HTTP (needs `pip install openai httpx`) |
| `cli.sh` | a tour of the `lure` command |

```bash
export LURE_MODEL=/path/to/lure
python examples/stream_to_wav.py out.wav
```
