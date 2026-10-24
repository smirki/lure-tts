"""Compare a backend's prompt dump against conformance/reference/prompt_ids.json (or prompt_modes.json).

usage: python conformance/check_prompt.py BACKEND_DUMP.json [--reference reference/prompt_modes.json]
Dump: {"backend": name, "entries": [{"id", "input_ids", "ref_ids", "instruct_ids", "layout", "trailing", "n_ref_frames"}]}
Any field a backend does not produce may be omitted (it is then skipped and reported). Exit code 1 on any mismatch.

icl non_streaming entries (all of prompt_ids.json) compare everything before the ref rows exactly and check that the ref
rows run 0..T-1 with tts_pad text, so a dump made with any reference length passes. Every other entry (streaming icl,
where text overlays the ref rows, and the modes without ref rows) must match the full layout made with the entry's
n_ref_frames, and `trailing` too when the dump has it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def norm_codec(c):
    return c if c is None or isinstance(c, (int, str)) else int(c)


def split_ref(layout):
    head, refs = [], []
    for t, c in layout:
        c = norm_codec(c)
        if isinstance(c, str) and c.startswith("ref:"):
            refs.append((t, int(c[4:])))
        else:
            if refs:
                raise ValueError("non-ref row after ref rows")
            head.append([t, c])
    return head, refs


def _first_diff(name: str, want: list, got: list) -> list[str]:
    for i, (a, b) in enumerate(zip(want, got)):
        if a != b:
            return [f"{name} row {i}: expected {a} got {b}"]
    if len(want) != len(got):
        return [f"{name} length: expected {len(want)} got {len(got)}"]
    return []


def compare_entry(ref: dict, got: dict) -> list[str]:
    errs = []
    for k in ("input_ids", "ref_ids", "instruct_ids"):
        if k in got and got[k] != ref[k]:
            errs.append(f"{k}: expected {ref[k]} got {got[k]}")
    exact = ref.get("mode", "icl") != "icl" or ref.get("prompt_layout", "non_streaming") != "non_streaming"
    if exact:
        if "layout" in got:
            errs += _first_diff("layout", ref["layout"], [[t, norm_codec(c)] for t, c in got["layout"]])
        if "trailing" in got:
            errs += _first_diff("trailing", ref["trailing"], [[t, norm_codec(c)] for t, c in got["trailing"]])
        return errs
    if "layout" in got:
        rh, _ = split_ref(ref["layout"])
        gh, gr = split_ref(got["layout"])
        if rh != gh:
            for i, (a, b) in enumerate(zip(rh, gh)):
                if a != b:
                    errs.append(f"layout row {i}: expected {a} got {b}")
                    break
            if len(rh) != len(gh):
                errs.append(f"layout prefix length: expected {len(rh)} got {len(gh)}")
        n = got.get("n_ref_frames", len(gr))
        if [t for _, t in gr] != list(range(n)):
            errs.append(f"ref rows not 0..{n - 1} in order")
        if any(t != 151671 for t, _ in gr):
            errs.append("ref rows must carry text id 151671 (tts_pad)")
    return errs


def check(dump: dict, reference: dict) -> tuple[int, int, list[str]]:
    refs = {e["id"]: e for e in reference["entries"]}
    n_ok, report = 0, []
    for e in dump["entries"]:
        if e["id"] not in refs:
            report.append(f"{e['id']}: not in reference")
            continue
        errs = compare_entry(refs[e["id"]], e)
        if errs:
            report += [f"{e['id']}: {x}" for x in errs]
        else:
            n_ok += 1
    missing = sorted(set(refs) - {e["id"] for e in dump["entries"]})
    if missing:
        report.append(f"missing entries (not checked): {missing}")
    return n_ok, len(dump["entries"]), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--reference", default=os.path.join(HERE, "reference", "prompt_ids.json"))
    a = ap.parse_args()
    dump, reference = json.load(open(a.dump)), json.load(open(a.reference))
    n_ok, n, report = check(dump, reference)
    for line in report:
        print(line)
    print(f"{dump.get('backend', '?')}: {n_ok}/{n} entries match")
    sys.exit(0 if n_ok == n and not any("missing" not in r for r in report) else 1)


if __name__ == "__main__":
    main()
