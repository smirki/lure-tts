"""Encode a voice once, then reuse it by name (also from the CLI and the server).

    LURE_MODEL=/path/to/lure python examples/voice_cache.py
"""
import time

from lure_tts import Lure
from _voice import REF_TEXT, REF_WAV

with Lure.load() as tts:
    t0 = time.perf_counter()
    tts.make_voice(REF_WAV, REF_TEXT, name="willow")  # encodes and saves to ~/.cache/lure/voices/willow
    print(f"first make_voice: {time.perf_counter() - t0:.2f} s")

    t0 = time.perf_counter()
    tts.make_voice(REF_WAV, REF_TEXT, name="willow")  # same wav, text and model: loaded from the cache
    print(f"second make_voice: {time.perf_counter() - t0:.3f} s")

    print("cached voices:", tts.voices.list())
    tts.speak("I was encoded once and reused after that.", "willow", out="cached_voice.wav")
