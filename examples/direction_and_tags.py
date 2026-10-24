"""Inline tags are performed, never read aloud. A direction sets the delivery for the whole request.

    LURE_MODEL=/path/to/lure python examples/direction_and_tags.py
"""
from lure_tts import Lure
from _voice import REF_TEXT, REF_WAV

with Lure.load() as tts:
    voice = tts.make_voice(REF_WAV, REF_TEXT, name="demo")
    text = "(sigh) I guess we're doing this again. (chuckle) Fine. I'm in."
    tts.speak(text, voice, seed=3, out="plain.wav")
    tts.speak(text, voice, instruct="Tired and resigned, then a reluctant smile.", seed=3, out="directed.wav")
    tts.speak("You actually did it? (laugh) I can't believe you actually did it!", voice,
              instruct="Delighted, warm.", seed=3, out="laugh.wav")
print("wrote plain.wav, directed.wav, laugh.wav")
