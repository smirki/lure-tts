"""Greedy reference run in qwen-tts 0.1.1 (PyTorch, CPU, float32). Started by make_qwen_fixture.py with the Python of a
qwen-tts environment; it reads job.json and writes the result JSON. Inputs are explicit (token ids, reference codes,
x-vector), so this script needs nothing from this repository.

    python qwen_worker.py job.json out.json
"""
import json
import sys
from importlib.metadata import version

import numpy as np
import torch
from safetensors.torch import load_file

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration


def load(model_dir):
    cfg = json.load(open(f"{model_dir}/config.json"))
    for k in ("architectures", "model_type"):
        cfg.pop(k, None)
    config = Qwen3TTSConfig(**cfg)
    config._attn_implementation = "sdpa"
    config.talker_config._attn_implementation = "sdpa"
    config.talker_config.code_predictor_config._attn_implementation = "sdpa"
    m = Qwen3TTSForConditionalGeneration(config).float().eval()
    sd = load_file(f"{model_dir}/model.safetensors")
    talker = {k[len("talker."):]: v for k, v in sd.items() if k.startswith("talker.")}
    missing, unexpected = m.talker.load_state_dict(talker, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{model_dir}: talker weights do not match qwen-tts: missing {missing}, unexpected {unexpected}")
    return m


def run_case(m, c, proj):
    captured = {}
    orig = m.talker.generate

    def capture(**kw):
        captured["prefill"] = kw["inputs_embeds"][0].detach().numpy().astype(np.float64)
        captured["trailing"] = kw["trailing_text_hidden"][0].detach().numpy().astype(np.float64)
        return orig(**kw)

    m.talker.generate = capture
    vcp = None
    if c["mode"] in ("icl", "xvec"):
        icl = c["mode"] == "icl"
        vcp = {"ref_code": [torch.tensor(c["ref_codes"], dtype=torch.long)] if icl else None,
               "ref_spk_embedding": [torch.tensor(c["x_vector"], dtype=torch.float32)],
               "x_vector_only_mode": [not icl], "icl_mode": [icl]}
    ins = [torch.tensor([c["instruct_ids"]]) if c["instruct_ids"] else None]
    try:
        with torch.no_grad():
            codes, _ = m.generate(
                input_ids=[torch.tensor([c["input_ids"]])], instruct_ids=ins,
                ref_ids=[torch.tensor([c["ref_ids"]])] if c["mode"] == "icl" else None,
                voice_clone_prompt=vcp, languages=[c["language"]],
                speakers=[c["speaker"]] if c["speaker"] else None,
                non_streaming_mode=c["layout"] == "non_streaming", max_new_tokens=c["max_new_tokens"],
                do_sample=False, subtalker_dosample=False, repetition_penalty=1.05)
    finally:
        m.talker.generate = orig
    p = np.asarray(proj, np.float64)
    return {"id": c["id"], "codes": codes[0].tolist(),
            "prefill_proj": (captured["prefill"] @ p).tolist(), "trailing_proj": (captured["trailing"] @ p).tolist()}


def main():
    job = json.load(open(sys.argv[1]))
    torch.manual_seed(0)
    models, out = {}, []
    for c in job["cases"]:
        if c["model_dir"] not in models:
            models[c["model_dir"]] = load(c["model_dir"])
        out.append(run_case(models[c["model_dir"]], c, job["projection"][c["model_dir"]]))
    meta = {"qwen_tts": version("qwen-tts"), "torch": torch.__version__}
    json.dump({"meta": meta, "results": out}, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
