"""Pure scoring helpers with no model dependencies. score.py runs the ASR and ECAPA models and calls these.

WER uses the normalisation in norm() (tags removed, contractions expanded, a few number words mapped to digits).
Tag-read detection is cases.tag_words_read: a tag word in the transcript that is not in the plain text means the tag was
spoken aloud.
"""
from __future__ import annotations

import re
import statistics

import cases as C

NUM = {"forty": "40", "three": "3", "four": "4"}
_CONTR = (("won't", "will not"), ("can't", "can not"), ("n't", " not"), ("'ll", " will"), ("'re", " are"), ("'ve", " have"),
          ("'m", " am"), ("there's", "there is"), ("that's", "that is"), ("it's", "it is"), ("he's", "he is"),
          ("she's", "she is"), ("you'd", "you would"))


def norm(t: str) -> list[str]:
    t = re.sub(r"\([^)]*\)", " ", t.lower()).replace("’", "'")
    for c, e in _CONTR:
        t = t.replace(c, e)
    t = t.replace("cannot", "can not").replace("'", "")
    return [NUM.get(x, x) for x in re.sub(r"[^a-z0-9 ]", " ", t).split()]


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(1, len(ref))


def score_row(row: dict, transcript: str, ecapa: float | None) -> dict:
    return {"file": row["file"], "case": row["case"], "take": row.get("take"), "kind": row.get("kind"), "asr": transcript,
            "wer": round(wer(norm(row["text"]), norm(transcript)), 4), "ecapa": None if ecapa is None else round(float(ecapa), 4),
            "tags_read": C.tag_words_read(row["text"], transcript),
            "eos_all": all(row.get("ended_by_eos") or [True]), "rtf": row.get("rtf"), "ttfa_s": row.get("ttfa_s")}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def summarize(scores: list[dict]) -> dict:
    """Per kind and overall: mean WER, mean ECAPA, tag-read count, EOS rate, mean RTF. Compare backends in distribution
    (against the qwen-tts ref run), never bit-exact audio: bf16 vs fp32+autocast and kernel order change every sample."""
    out = {}
    groups = {"all": scores, **{k: [s for s in scores if s.get("kind") == k] for k in sorted({s.get("kind") for s in scores})}}
    for k, ss in groups.items():
        if not ss:
            continue
        out[k] = {"n": len(ss), "wer": _mean([s["wer"] for s in ss]), "wer_max": max(s["wer"] for s in ss),
                  "ecapa": _mean([s["ecapa"] for s in ss]), "tags_read": sum(len(s["tags_read"]) for s in ss),
                  "eos_rate": _mean([1.0 if s["eos_all"] else 0.0 for s in ss]), "rtf": _mean([s["rtf"] for s in ss])}
    return out


# Pass bars vs the qwen-tts reference run (same cases, seeds, voice). Starting points; tune after the first real run.
GATES = {"wer_abs_max": 0.10, "wer_delta_max": 0.03, "ecapa_delta_min": -0.03, "tags_read_max": 0, "eos_rate_min": 1.0}


def gate(summary: dict, ref_summary: dict | None) -> list[str]:
    a = summary.get("all", {})
    fails = []
    if a.get("wer") is not None and a["wer"] > GATES["wer_abs_max"]:
        fails.append(f"mean WER {a['wer']} > {GATES['wer_abs_max']}")
    if a.get("tags_read", 0) > GATES["tags_read_max"]:
        fails.append(f"{a['tags_read']} tag words read aloud")
    if a.get("eos_rate") is not None and a["eos_rate"] < GATES["eos_rate_min"]:
        fails.append(f"EOS rate {a['eos_rate']} (a segment hit max_new_tokens)")
    r = (ref_summary or {}).get("all")
    if r:
        if a.get("wer") is not None and r.get("wer") is not None and a["wer"] - r["wer"] > GATES["wer_delta_max"]:
            fails.append(f"WER {a['wer']} vs ref {r['wer']}")
        if a.get("ecapa") is not None and r.get("ecapa") is not None and a["ecapa"] - r["ecapa"] < GATES["ecapa_delta_min"]:
            fails.append(f"ECAPA {a['ecapa']} vs ref {r['ecapa']}")
    return fails
