"""Score conformance runs: Qwen3-ASR-1.7B transcript -> WER and tag-read check, ECAPA cosine vs the voice prompt.

    python conformance/score.py runs/cuda runs/mlx --ref runs/ref [--device cuda] [--force]

Each run dir is runner.py output (manifest.jsonl, ref.wav, wavs). Writes per dir:
    scores.jsonl    one row per wav: ASR transcript, WER, ECAPA, tags read, EOS, RTF, TTFA
    summary.json    means per case kind and overall (scoring.summarize)
With --ref, the reference run is scored too (unless it already has summary.json) and every other run is checked against
scoring.GATES; the result goes to gates.json next to each run and the exit code is 1 if any run fails.

Requirements: requirements-score.txt (qwen-asr, speechbrain, librosa, torch). A GPU is recommended; the 1.7B ASR model
in bf16 needs about 5 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scoring  # noqa: E402

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


class Scorer:
    def __init__(self, device: str = "cuda", batch: int = 8):
        import torch
        from qwen_asr import Qwen3ASRModel
        from speechbrain.inference.speaker import EncoderClassifier

        self.torch, self.device, self.batch = torch, device, batch
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.asr = Qwen3ASRModel.from_pretrained(ASR_MODEL, dtype=dtype, device_map=device,
                                                 max_inference_batch_size=batch, max_new_tokens=1024)
        savedir = os.path.join(os.path.expanduser("~"), ".cache", "lure", "ecapa")
        self.ecapa = EncoderClassifier.from_hparams(source=ECAPA_MODEL, savedir=savedir, run_opts={"device": device})

    def embed(self, path: str):
        import librosa
        x, _ = librosa.load(path, sr=16000, mono=True)
        with self.torch.no_grad():
            e = self.ecapa.encode_batch(self.torch.from_numpy(x)[None].to(self.device))[0, 0]
        return e / e.norm()

    def score_dir(self, d: str) -> dict:
        rows = [json.loads(line) for line in open(os.path.join(d, "manifest.jsonl"))]
        ref = self.embed(os.path.join(d, "ref.wav"))
        scores = []
        with open(os.path.join(d, "scores.jsonl"), "w") as f:
            for i in range(0, len(rows), self.batch):
                chunk = rows[i:i + self.batch]
                res = self.asr.transcribe(audio=[os.path.join(d, r["file"]) for r in chunk], language=["English"] * len(chunk))
                for r, t in zip(chunk, res):
                    cos = float(self.embed(os.path.join(d, r["file"])) @ ref)
                    s = scoring.score_row(r, t.text.strip(), cos)
                    scores.append(s)
                    f.write(json.dumps(s) + "\n")
                    print(f"{os.path.basename(d)}/{s['file']}: wer={s['wer']} ecapa={s['ecapa']} tags_read={s['tags_read']}",
                          flush=True)
        summary = scoring.summarize(scores)
        json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
        return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="runner.py output dirs")
    ap.add_argument("--ref", default=None, help="reference run dir (ref_backend); enables the pass gates")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="rescore dirs that already have summary.json")
    a = ap.parse_args(argv)

    scorer = None

    def summary_of(d: str) -> dict:
        nonlocal scorer
        p = os.path.join(d, "summary.json")
        if os.path.exists(p) and not a.force:
            return json.load(open(p))
        scorer = scorer or Scorer(a.device, a.batch)
        return scorer.score_dir(d)

    ref = summary_of(a.ref) if a.ref else None
    failed = False
    for d in a.runs:
        s = summary_of(d)
        line = f"{d}: {json.dumps(s.get('all'))}"
        if ref is not None:
            fails = scoring.gate(s, ref)
            json.dump({"gates": scoring.GATES, "ref": a.ref, "fails": fails}, open(os.path.join(d, "gates.json"), "w"), indent=1)
            line += "  PASS" if not fails else "  FAIL: " + "; ".join(fails)
            failed |= bool(fails)
        print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
