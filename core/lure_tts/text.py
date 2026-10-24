"""Text front-end. Lives in core only; backends receive one segment at a time.

Two modes:
- batch (segment_text, full text known): whole sentences packed up to pack_words (25) spoken words; a sentence longer than max_words is cut at its last clause
  boundary within max_words, else at a word boundary. Fewer cuts = better prosody; streaming TTFA comes from chunked decode.
- stream (Segmenter default, LLM tokens arriving): commit a segment at sentence punctuation (. ! ? ... with closing quotes/brackets), at a blank line, or at clause
  punctuation (, ; : ... …) once enough words are buffered; the first segment cuts at the earliest boundary (low TTFA);
- never inside an open ( ... ) tag, never right after an abbreviation ("Dr.") or a bare number ("3." / "3,000");
- a segment that is only tags waits for its words; a sentence shorter than 3 spoken words joins the next one;
- runaway text with no punctuation is cut at a word boundary after max_words;
- tags are plain text: nothing is stripped or normalised. Only whitespace at segment edges is trimmed.

tags=False (models without inline tags, e.g. stock Qwen3-TTS): parentheses are ordinary text. Words inside them count,
an open parenthesis does not block a cut, and a parenthetical is never held back as a "lone tag". The text still passes
through verbatim; only the sentence segmentation applies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "etc", "e.g", "i.e", "no", "mt", "prof", "approx", "fig", "inc", "ltd"}
SENT_END = re.compile(r'([.!?]+["\')\]”’]*|\.\.\.|…)\s')
CLAUSE_END = re.compile(r'([,;:\u2014]|\.\.\.|…)\s')  # \u2014: em dash in the input text
PARA = re.compile(r"\n[ \t]*\n")
TAG = re.compile(r"\([^)]*\)")
LONE_TAGS = re.compile(r"(\s*\([^)]*\))+\s*[.!?,;:]*")


def n_words(s: str, tags: bool = True) -> int:
    """Spoken words: tags do not count (when tags is True)."""
    return len((TAG.sub(" ", s) if tags else s).split())


@dataclass
class Segment:
    text: str
    kind: str  # "sentence" | "clause" | "paragraph" | "end"
    index: int = 0


class Segmenter:
    def __init__(self, first_min_words: int = 4, clause_min_words: int = 10, max_words: int = 30, sentence_min_words: int = 3,
                 mode: str = "stream", pack_words: int = 25, tags: bool = True):
        if mode not in ("stream", "batch"):
            raise ValueError("mode must be 'stream' or 'batch'")
        self.mode, self.final, self.pack_words, self.tags = mode, False, pack_words, tags
        self.buf, self.first, self.count = "", True, 0
        self.first_min_words, self.clause_min_words, self.max_words = first_min_words, clause_min_words, max_words
        self.sentence_min_words = sentence_min_words  # "(chuckle) Fine." alone is too short: join it with the next sentence

    def _n(self, s: str) -> int:
        return n_words(s, self.tags)

    def _ok(self, text: str, end: int) -> bool:
        head = text[:end]
        if self.tags and head.count("(") > head.count(")"):
            return False  # inside an event tag
        words = head.split()
        if not words:
            return False
        last = words[-1].rstrip('.!?,;:"\')]”’').lower()
        if last in ABBREV and head.rstrip()[-1:] == ".":
            return False
        if re.fullmatch(r"\d+", last) and head.rstrip()[-1:] in (".", ","):
            return False  # "3." / "3," could continue ("3.5", "3,000")
        return not (self.tags and LONE_TAGS.fullmatch(head))  # a lone tag waits for its words

    def _runaway(self, buf):
        if self._n(buf) > self.max_words:  # runaway: cut at the last word boundary outside a tag
            for i in range(len(buf.rstrip()) - 1, 0, -1):
                if buf[i] == " " and self._ok(buf, i) and self._n(buf[:i]) <= self.max_words:
                    return i + 1, "clause"
        return None

    def _cut(self):
        buf = self.buf
        strong, clauses = [], []
        for m in SENT_END.finditer(buf):
            if self._ok(buf, m.end(1)) and self._n(buf[:m.end()]) >= self.sentence_min_words:
                strong.append((m.end(), "sentence"))
        for m in PARA.finditer(buf):
            if self._ok(buf, m.start()) and buf[:m.start()].strip():
                strong.append((m.end(), "paragraph"))
        for m in CLAUSE_END.finditer(buf):
            if self._ok(buf, m.end(1)):
                clauses.append((m.end(), "clause"))
        if self.mode == "batch" and self.final and buf.strip() and not any(c[0] >= len(buf.rstrip()) for c in strong):
            strong.append((len(buf), "end"))
        strong.sort()
        if self.mode == "batch":
            # whole text known: pack whole sentences up to pack_words; clause cuts only inside a sentence longer than max_words
            paras = [c[0] for c in strong if c[1] == "paragraph"]
            if paras:  # never pack across a blank line
                strong = [c for c in strong if c[0] <= paras[0]]
            packed = [c for c in strong if self._n(buf[:c[0]]) <= self.pack_words]
            if packed:
                return packed[-1]
            end = strong[0][0] if strong else len(buf)
            if self._n(buf[:end]) <= self.max_words:
                return strong[0] if strong else None
            ok = [c for c in clauses if c[0] < end and self.clause_min_words <= self._n(buf[:c[0]]) <= self.max_words]
            if ok:
                return ok[-1]
            cut = self._runaway(buf[:end])
            return cut or (strong[0] if strong else None)
        # incremental (LLM tokens): earliest boundary for the first segment (low TTFA), then prefer sentences
        min_clause = self.first_min_words if self.first else self.clause_min_words
        cands = sorted(strong + [c for c in clauses if self._n(buf[:c[0]]) >= min_clause])
        if not cands:
            return self._runaway(buf)
        if self.first:
            return cands[0]
        return strong[0] if strong else cands[0]

    def _emit(self, seg: str, kind: str) -> Segment:
        s = Segment(seg, kind, self.count)
        self.count += 1
        self.first = False
        return s

    def feed(self, delta: str) -> list[Segment]:
        self.buf += delta
        out = []
        while (c := self._cut()):
            seg, self.buf = self.buf[:c[0]].strip(), self.buf[c[0]:]
            if seg:
                out.append(self._emit(seg, c[1]))
        return out

    def flush(self) -> list[Segment]:
        seg, self.buf = self.buf.strip(), ""
        return [self._emit(seg, "end")] if seg else []


def segment_text(text: str, mode: str = "batch", **kw) -> list[Segment]:
    """Whole text known up front (speak / stream of a full string): sentence segments, clause cuts only for long sentences."""
    s = Segmenter(mode=mode, **kw)
    s.final = True
    return s.feed(text) + s.flush()


def check_tags_balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            return False
    return depth == 0
