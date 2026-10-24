"""Stream speech chunk by chunk and write it to a wav as it arrives.

    LURE_MODEL=/path/to/lure python examples/stream_to_wav.py out.wav
"""
import sys

import soundfile as sf

from lure_tts import Lure
from _voice import REF_TEXT, REF_WAV

out = sys.argv[1] if len(sys.argv) > 1 else "out.wav"
with Lure.load() as tts:  # model from $LURE_MODEL, backend picked for this machine
    voice = tts.make_voice(REF_WAV, REF_TEXT, name="demo")
    with sf.SoundFile(out, "w", samplerate=tts.sample_rate, channels=1, subtype="PCM_16") as f:
        for pcm in tts.stream("Hello there. This is streaming, one small chunk at a time.", voice, seed=7):
            f.write(pcm)  # float32 mono, 24 kHz, about 0.32 s per chunk
print(f"wrote {out} with backend {tts.backend_name}")
