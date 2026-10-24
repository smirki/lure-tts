"""Regenerate fixtures/qwen_greedy.json: greedy codes and prefill / trailing row fingerprints from qwen-tts 0.1.1 on the
tiny random checkpoints of tiny.py, for every case in tiny.greedy_cases().

Run with this backend's Python (mlx + mlx-audio) and point --qwen-python at a qwen-tts 0.1.1 environment (PyTorch, CPU,
a few hundred MB):

    python backends/mlx/tests/make_qwen_fixture.py --qwen-python /path/to/qwen-tts-env/bin/python
"""
import argparse
import json
import os
import subprocess
import tempfile

import tiny

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "qwen_greedy.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qwen-python", required=True, help="python of an environment with qwen-tts 0.1.1 and torch")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as root:
        cases, projection = [], {}
        for c in tiny.greedy_cases():
            d = tiny.make_model_dir(root, c["variant"])
            projection[d] = tiny.projection(tiny.SHAPES[tiny.VARIANTS[c["variant"]][1]]["hidden"]).tolist()
            cases.append(dict(c, model_dir=d))
        job, res = os.path.join(root, "job.json"), os.path.join(root, "out.json")
        with open(job, "w") as f:
            json.dump({"cases": cases, "projection": projection}, f)
        subprocess.run([args.qwen_python, os.path.join(HERE, "qwen_worker.py"), job, res], check=True)
        with open(res) as f:
            got = json.load(f)
    rnd = lambda xs: [float(f"{x:.7g}") for x in xs]  # noqa: E731
    results = {r["id"]: {"codes": r["codes"], "prefill_proj": rnd(r["prefill_proj"]), "trailing_proj": rnd(r["trailing_proj"])}
               for r in got["results"]}
    meta = dict(got["meta"], generator="backends/mlx/tests/make_qwen_fixture.py", shapes=tiny.SHAPES, seeds=tiny.SEEDS,
                frames=tiny.GREEDY_FRAMES, sampling="greedy talker and code predictor, repetition_penalty 1.05",
                dtype="float32 CPU")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"meta": meta, "results": results}, f, separators=(",", ":"))
    print(f"wrote {OUT}: {len(results)} cases")


if __name__ == "__main__":
    main()
