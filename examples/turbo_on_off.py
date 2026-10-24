"""Turbo is picked automatically when the model folder has turbo.json. turbo=0 forces the original code predictor.

    python examples/turbo_on_off.py /path/to/lure_turbo
"""
import sys
import time

from lure_tts import Lure
from _voice import REF_TEXT, REF_WAV

model = sys.argv[1]
text = "Okay, the engine just made a noise I have never heard before."
for turbo in ("auto", "0"):
    with Lure.load(model, turbo=turbo) as tts:
        voice = tts.make_voice(REF_WAV, REF_TEXT, name="demo")
        tts.speak(text, voice, seed=1)  # warmup
        t0 = time.perf_counter()
        audio = tts.speak(text, voice, seed=1, out=f"turbo_{turbo}.wav")
        rtf = (time.perf_counter() - t0) / (len(audio) / tts.sample_rate)
        print(f"turbo={turbo}: active={tts.turbo} rtf={rtf:.2f}")
