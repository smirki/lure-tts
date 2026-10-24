"""Feed text as it arrives (for example LLM tokens). Audio starts after the first few words.

    LURE_MODEL=/path/to/lure python examples/live_text.py
"""
import numpy as np

from lure_tts import Lure
from lure_tts.audio import write_wav
from _voice import REF_TEXT, REF_WAV

tokens = "Sure, I can help with that. (laugh) First, open the settings page, then pick the audio tab.".split(" ")
with Lure.load() as tts:
    live = tts.live(tts.make_voice(REF_WAV, REF_TEXT, name="demo"), seed=5)
    parts = []
    for tok in tokens:
        parts += list(live.feed(tok + " "))  # yields audio for each segment the text so far completes
    parts += list(live.flush())
    write_wav("live.wav", np.concatenate(parts))
print("wrote live.wav")
