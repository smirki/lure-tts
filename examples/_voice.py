"""Default voice prompt for the examples: the conformance reference clip. Pass your own with REF_WAV / REF_TEXT."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_WAV = os.environ.get("REF_WAV", os.path.join(ROOT, "conformance", "assets", "reference_voice.wav"))
REF_TEXT = os.environ.get("REF_TEXT", "Yes, the weekend order can stay exactly as it is. "
                                      "Everything's fine on our end, truly it is, thank you for checking.")
