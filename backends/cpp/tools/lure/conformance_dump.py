"""Dump the C++ backend's prompt recipe for every conformance entry, then check its tokenizer.

No weights are loaded and nothing is generated: lure_dump_prefill_offline reads config.json and the tokenizer
files only. Each conformance/reference/prompt_modes.json entry runs against a temporary folder holding the
reference tokenizer and the entry's configs/<config>; prompt_ids.json entries run against --model-dir (default
conformance/reference/tokenizer, the Lure config).

usage: python tools/lure/conformance_dump.py --lib build/macos-cpu/bin/liblure_tts.dylib \
           --reference ../../conformance/reference/prompt_modes.json --out /tmp/cpp_prompt_modes.json
       python ../../conformance/check_prompt.py /tmp/cpp_prompt_modes.json \
           --reference ../../conformance/reference/prompt_modes.json
"""
import argparse
import ctypes
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REFERENCE = os.path.join(REPO, "conformance", "reference")
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

from lure_cpp import _bind, make_request, recipe_rows  # noqa: E402

TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")


def prompt_only_dir(root: str, config: str) -> str:
    """A folder with the reference tokenizer and one stock config.json: enough for the offline dump."""
    path = os.path.join(root, os.path.splitext(config)[0])
    if not os.path.isdir(path):
        os.makedirs(path)
        for name in TOKENIZER_FILES:
            os.symlink(os.path.join(REFERENCE, "tokenizer", name), os.path.join(path, name))
        os.symlink(os.path.join(REFERENCE, "configs", config), os.path.join(path, "config.json"))
    return path


def dump(lib, model_dir: str, e: dict, n_ref: int) -> dict:
    mode = e.get("mode", "icl")
    ref_text = e.get("ref_text") if mode == "icl" else None
    r = make_request(lib, e["text"], mode, e.get("prompt_layout", "non_streaming"), e.get("language"),
                     e.get("direction"), e.get("speaker"))
    ptr = lib.lure_dump_prefill_offline(model_dir.encode(), ctypes.byref(r),
                                          None if ref_text is None else ref_text.encode("utf-8"), n_ref)
    if not ptr:
        raise SystemExit(f"{e['id']}: {lib.lure_last_error().decode()}")
    recipe = json.loads(ctypes.string_at(ptr).decode("utf-8"))
    lib.lure_string_free(ptr)
    return {
        "id": e["id"],
        "input_ids": recipe["input_ids"],
        "ref_ids": recipe["ref_ids"] or None,
        "instruct_ids": recipe["instruct_ids"] or None,
        "layout": recipe_rows(recipe),
        "trailing": recipe_rows(recipe, "trailing_"),
        "n_ref_frames": n_ref,
        "mode": recipe["mode"],
        "layout_name": recipe["layout"],
    }


def check_tokenizer(lib, model_dir: str) -> int:
    path = os.path.join(REFERENCE, "tokenizer_ids.json")
    if not os.path.exists(path):
        return 0
    bad = 0
    entries = json.load(open(path))["entries"]
    for t in entries:
        ptr = lib.lure_tokenize_offline(model_dir.encode(), t["text"].encode("utf-8"))
        if not ptr:
            raise SystemExit(f"tokenize: {lib.lure_last_error().decode()}")
        ids = json.loads(ctypes.string_at(ptr).decode("utf-8"))
        lib.lure_string_free(ptr)
        if ids != t["ids"]:
            bad += 1
            print(f"tokenizer mismatch: {t['text']!r}\n  expected {t['ids']}\n  got      {ids}")
    print(f"tokenizer: {len(entries) - bad}/{len(entries)} strings match")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--reference", default=os.path.join(REFERENCE, "prompt_ids.json"),
                    help="prompt_ids.json (icl, one model) or prompt_modes.json (every mode, per-entry config)")
    ap.add_argument("--model-dir", default=os.path.join(REFERENCE, "tokenizer"),
                    help="model for prompt_ids.json entries: a checkpoint, or any folder with config.json + tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref-frames", type=int, default=6, help="symbolic reference frames for prompt_ids.json entries")
    a = ap.parse_args()
    lib = _bind(ctypes.CDLL(os.path.abspath(a.lib)))
    reference = json.load(open(a.reference))
    entries = []
    with tempfile.TemporaryDirectory(prefix="lure_prompt_") as tmp:
        for e in reference["entries"]:
            if "config" in e:
                entries.append(dump(lib, prompt_only_dir(tmp, e["config"]), e, int(e["n_ref_frames"])))
            else:
                entries.append(dump(lib, a.model_dir, e, a.ref_frames))
    json.dump({"backend": "cpp", "entries": entries}, open(a.out, "w"), indent=1)
    print(f"wrote {len(entries)} entries to {a.out}")
    sys.exit(1 if check_tokenizer(lib, a.model_dir) else 0)


if __name__ == "__main__":
    main()
