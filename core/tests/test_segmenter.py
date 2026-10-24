import re

import pytest

from lure_tts.text import Segmenter, check_tags_balanced, n_words, segment_text


def texts(t, **kw):
    return [s.text for s in segment_text(t, **kw)]


@pytest.mark.parametrize("mode", ["stream", "batch"])
def test_never_splits_inside_tag(mode):
    t = ("Well (a very long nervous laugh, with a pause; then. a gasp! and more) okay. Next sentence here, and more words follow. "
         + " ".join(["filler"] * 26) + " (tag, with. punctuation!) " + " ".join(["tail"] * 10) + ".")
    segs = texts(t, mode=mode, first_min_words=1, clause_min_words=1, sentence_min_words=1, pack_words=5)
    assert len(segs) > 2
    for s in segs:
        assert check_tags_balanced(s), s


def test_tags_kept_verbatim():
    t = "Wait, wait, wait. You're telling me the whole thing was a joke? (laugh) Unbelievable."
    segs = texts(t)
    assert " ".join(segs) == t
    assert any("(laugh)" in s for s in segs)


def test_lone_tag_waits_for_words():
    segs = texts("(sigh) I guess we're doing this again. (chuckle) Fine. I'm in.")
    assert all(re.sub(r"\([^)]*\)", "", s).strip(" .") for s in segs)
    assert segs[0].startswith("(sigh) I guess")


def test_sentence_punctuation_and_quotes():
    t = '"You\'re either very brave or very lost," he said. Mara laughed. Then she left!'
    segs = stream_texts(t)
    assert segs[-1] == "Mara laughed. Then she left!"  # 2-word sentence joins the next
    assert all(s[-1] in '.!?"' for s in segs)


def stream_texts(t, **kw):
    s = Segmenter(**kw)
    return [x.text for x in s.feed(t) + s.flush()]


def test_stream_first_segment_cuts_early_at_clause():
    s = Segmenter()
    segs = s.feed("The storm had been building all afternoon, and by the time Mara reached the lighthouse, the sky was dark. ")
    assert segs[0].text == "The storm had been building all afternoon,"
    assert segs[0].kind == "clause"


def test_stream_later_segments_prefer_sentences():
    t = "First one here. Second sentence, with a short clause, ends here. Third one now."
    assert stream_texts(t) == ["First one here.", "Second sentence, with a short clause, ends here.", "Third one now."]


def test_batch_packs_short_sentences():
    t = "First one here. Second sentence, with a short clause, ends here. Third one now."
    assert texts(t) == [t]
    t = "Wait, wait, wait. You're telling me the whole thing was a joke? (laugh) Unbelievable."
    assert texts(t) == [t]  # a short tagged line stays one call


def test_batch_short_sentence_joins_next():
    assert stream_texts("(chuckle) Fine. I'm in now.") == ["(chuckle) Fine. I'm in now."]


def test_batch_long_sentence_cut_at_clause():
    t = ("Inside the envelope was a single photograph of this lighthouse, and on the back, in her grandmother's handwriting, "
         "were four words that she had never once expected to read again in her whole life: come before the winter.")
    segs = texts(t)
    assert len(segs) == 2 and segs[0].endswith(",") and all(n_words(s) <= 30 for s in segs)
    assert " ".join(segs) == t


def test_batch_packing_respects_pack_words():
    from cases import PASSAGE
    segs = texts(PASSAGE)
    assert all(n_words(s) <= 25 for s in segs)


def test_abbreviations_and_numbers_not_split():
    t = "Dr. Smith paid 3. 5 dollars? No. He paid 3,000 dollars to Mr. Jones yesterday afternoon."
    segs = texts(t)
    assert not any(s.endswith("Dr.") or s.endswith("Mr.") for s in segs)
    assert not any(s.endswith(" 3.") for s in segs)


def test_ellipsis_and_unicode_ellipsis():
    segs = texts("Well... I think so… maybe. Fine.")
    assert " ".join(segs).replace("  ", " ") == "Well... I think so… maybe. Fine."


def test_runaway_text_cut_at_word_boundary_outside_tags():
    t = " ".join(["word"] * 25) + " (long tag with words inside it) " + " ".join(["more"] * 20)
    segs = texts(t, max_words=30)
    assert len(segs) >= 2
    for s in segs:
        assert check_tags_balanced(s)
        assert n_words(s) <= 30


def test_paragraph_break():
    segs = texts("Heading without punctuation\n\nBody text goes here.")
    assert segs == ["Heading without punctuation", "Body text goes here."]


def test_incremental_feed_matches_batch():
    t = ("She opened the lid. Inside were dozens of letters, tied together with a faded blue ribbon, and a map. "
         "(sigh) At the center was an island, circled twice in red ink.")
    s = Segmenter()
    inc = []
    for i in range(0, len(t), 3):
        inc += [x.text for x in s.feed(t[i:i + 3])]
    inc += [x.text for x in s.flush()]
    assert " ".join(inc) == " ".join(t.split())
    for x in inc:
        assert check_tags_balanced(x)


def test_streaming_never_commits_open_tag():
    s = Segmenter(first_min_words=1, clause_min_words=1)
    out = s.feed("Okay, so (nervous, very")
    assert all(check_tags_balanced(x.text) for x in out)
    out = s.feed(" nervous laugh) fine. ")
    out += s.flush()
    assert all(check_tags_balanced(x.text) for x in out)


def test_segment_indices_and_kinds():
    segs = segment_text("One two three four. Five six.")
    assert [s.index for s in segs] == list(range(len(segs)))
    assert segs[-1].kind in ("end", "sentence")


@pytest.mark.parametrize("t", ["", "   ", "\n\n"])
def test_empty(t):
    assert segment_text(t) == []


def test_passage_segments_reasonable():
    from cases import PASSAGE
    segs = segment_text(PASSAGE)
    assert " ".join(s.text for s in segs) == PASSAGE.strip()
    assert all(n_words(s.text) <= 30 for s in segs)


def test_tags_false_treats_parentheses_as_text():
    from lure_tts.text import Segmenter, n_words, segment_text
    text = "He paused for a long moment by the window (he said it twice. Then he left the house) and the door closed behind him slowly, without a single sound at all."
    assert [s.text for s in segment_text(text)] == [text]  # tags: never cut inside ( ... )
    plain = [s.text for s in segment_text(text, tags=False)]
    assert plain == ["He paused for a long moment by the window (he said it twice.",
             "Then he left the house) and the door closed behind him slowly, without a single sound at all."]
    assert n_words("(laugh) hi", tags=False) == 2 and n_words("(laugh) hi") == 1
    seg = Segmenter(tags=False)
    out = seg.feed("(Aside one two three.) Next part here. ") + seg.flush()
    assert out[0].text == "(Aside one two three.)"
