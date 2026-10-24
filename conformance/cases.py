"""Shared conformance test lines. Pure data, importable by every backend runner (no deps).

Cases a-d are short lines (c and d with a direction), e is a ~90 s passage that goes through the core segmenter, and t1-t6
are tag-read lines: each tag must be performed (or silent), never spoken as a word. cn / dn repeat c / d without the
direction for A/B listening. Seeds: seed(case, k).
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REF_WAV = os.path.join(HERE, "assets", "reference_voice.wav")
REF_TEXT = ("Yes, the weekend order can stay exactly as it is. "
            "Everything's fine on our end, truly it is, thank you for checking.")
VOICE_ID = "reference_voice"

PASSAGE = (
    "The storm had been building all afternoon, and by the time Mara reached the lighthouse, the sky was the color of a bruise. "
    "She pushed the heavy door open with her shoulder and stepped inside, shaking the rain from her coat. "
    "The keeper, an old man named Tobias, looked up from his chair by the stove and raised an eyebrow. "
    "\"You're either very brave or very lost,\" he said. "
    "Mara laughed, a little too loudly, and admitted that it was probably a bit of both. "
    "She had driven three hours along the coast road, following a letter that had arrived at her apartment with no return address. "
    "Inside the envelope was a single photograph of this lighthouse, and on the back, in her grandmother's handwriting, were four words: come before the winter. "
    "Tobias studied the photograph for a long time. Then he stood, slowly, and walked to a wooden cabinet against the far wall. "
    "He unlocked it with a brass key and took out a small tin box, dented and rusted at the corners. "
    "\"Your grandmother left this with me forty years ago,\" he said quietly. \"She told me someone would come for it one day. I stopped believing her a long time ago.\" "
    "Mara held the box in both hands. It was lighter than she expected. "
    "Outside, the wind howled against the glass, and somewhere far below, the waves crashed against the rocks. "
    "She opened the lid. Inside were dozens of letters, tied together with a faded blue ribbon, and a map drawn by hand on yellowed paper. "
    "At the center of the map was a small island she had never heard of, circled twice in red ink. "
    "Mara looked up at Tobias, and for the first time that night, he smiled. "
    "\"Well,\" he said, \"I suppose you'll be needing a boat.\""
)

# name -> (text, direction or None, kind)
CASES: dict[str, tuple[str, str | None, str]] = {
    "a": ("Okay, the engine just made a noise I have never heard before.", None, "short"),
    "b": ("Wait, wait, wait. You're telling me the whole thing was a joke? (laugh) Unbelievable.", None, "short"),
    "c": ("(sigh) I guess we're doing this again. (chuckle) Fine. I'm in.", "Tired and resigned, then a reluctant smile.", "short"),
    "d": ("Grab the wrench, grab it, grab it!", "Frantic panic, breathless.", "short"),
    "e": (PASSAGE, None, "long"),
    "t1": ("(chuckle) Okay, okay, you got me there.", None, "tag"),
    "t2": ("I opened the box and (gasp) it was empty. Completely empty.", None, "tag"),
    "t3": ("(nervous laugh) So, um, about the car. There's a tiny scratch.", "Guilty and nervous.", "tag"),
    "t4": ("(sharp inhale) Don't move. There's something on your shoulder.", None, "tag"),
    "t5": ("Another Monday. (sigh) Coffee first, then the world.", None, "tag"),
    "t6": ("You actually did it? (laugh) I can't believe you actually did it!", "Delighted, warm.", "tag"),
    # direction A/B: same text as c and d without the direction (appended last so earlier seeds are unchanged)
    "cn": ("(sigh) I guess we're doing this again. (chuckle) Fine. I'm in.", None, "nodir"),
    "dn": ("Grab the wrench, grab it, grab it!", None, "nodir"),
}

# Words a correct render must not contain unless they are in the plain text (a tag read aloud). "shuckle" is how the ASR
# spells a read-aloud "(chuckle)".
TAG_WORDS = {"laugh", "laughs", "laughing", "sigh", "sighs", "chuckle", "chuckles", "shuckle", "gasp", "gasps", "inhale",
             "inhales", "nervous", "sharp"}

TAG_RE = re.compile(r"\([^)]*\)")


def plain_words(text: str) -> list[str]:
    """Words of the text with (tags) removed, lowercased, punctuation stripped: the WER reference."""
    return re.findall(r"[a-z0-9']+", TAG_RE.sub(" ", text).lower())


def tag_words_read(text: str, transcript: str) -> list[str]:
    allowed = set(plain_words(text))
    return [w for w in re.findall(r"[a-z']+", transcript.lower()) if w in TAG_WORDS and w not in allowed]


def seed(case: str, k: int = 0, variant: int = 0) -> int:
    keys = list(CASES)
    return 1000 + 100 * k + 10 * variant + keys.index(case)


SHORT = [k for k, v in CASES.items() if v[2] == "short"]
TAGS = [k for k, v in CASES.items() if v[2] == "tag"]
LONG = [k for k, v in CASES.items() if v[2] == "long"]
