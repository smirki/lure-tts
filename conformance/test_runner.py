"""Runner and score.py plumbing with the fake backend and precomputed summaries (no models)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import runner  # noqa: E402
import score  # noqa: E402

TOKDIR = os.path.join(HERE, "reference", "tokenizer")


def test_runner_fake_backend(tmp_path):
    out = tmp_path / "fake"
    rows = runner.run("fake", TOKDIR, str(out), ["a", "t1"], takes=1, cache_dir=str(tmp_path / "cache"))
    assert [r["file"] for r in rows] == ["a_k0.wav", "t1_k0.wav"]
    assert all(all(r["ended_by_eos"]) for r in rows)
    for name in ("ref.wav", "a_k0.wav", "manifest.jsonl", "env.json", "voice_fake.npz", "prompt_check.json"):
        assert (out / name).exists()
    pc = json.loads((out / "prompt_check.json").read_text())
    assert pc["prompt_ids"]["ok"] == pc["prompt_ids"]["total"] == 35
    assert pc["prompt_modes"]["ok"] == pc["prompt_modes"]["total"] > 0 and not pc["prompt_modes"]["report"]
    s = runner.summarize(rows)
    assert s["short"]["eos_rate"] == 1.0


def _summary(path, wer, ecapa, tags=0):
    path.mkdir()
    summ = {"all": {"n": 2, "wer": wer, "ecapa": ecapa, "tags_read": tags, "eos_rate": 1.0}}
    (path / "summary.json").write_text(json.dumps(summ))


def test_score_gates_from_existing_summaries(tmp_path):
    _summary(tmp_path / "ref", 0.015, 0.575)
    _summary(tmp_path / "good", 0.02, 0.57)
    _summary(tmp_path / "bad", 0.02, 0.50, tags=1)
    assert score.main([str(tmp_path / "good"), "--ref", str(tmp_path / "ref")]) == 0
    assert score.main([str(tmp_path / "bad"), "--ref", str(tmp_path / "ref")]) == 1
    fails = json.loads((tmp_path / "bad" / "gates.json").read_text())["fails"]
    assert any("ECAPA" in f for f in fails) and any("tag words" in f for f in fails)


def test_check_prompt_modes_exact():
    from check_prompt import check
    ref = json.load(open(os.path.join(HERE, "reference", "prompt_modes.json")))
    good = [{"id": e["id"], "layout": e["layout"], "trailing": e["trailing"]} for e in ref["entries"]]
    n_ok, n, report = check({"entries": good}, ref)
    assert n_ok == n == len(ref["entries"]) and not report
    e = next(e for e in ref["entries"] if e["id"] == "base.icl.s")
    bad = [dict(id=e["id"], layout=e["layout"], trailing=e["trailing"][1:])]
    assert check({"entries": bad}, ref)[0] == 0
