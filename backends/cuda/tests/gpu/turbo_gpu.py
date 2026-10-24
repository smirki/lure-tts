"""Real-weight Turbo checks on a CUDA GPU (not collected by pytest). Needs a Turbo model dir (with turbo.json).

    python tests/gpu/turbo_gpu.py --model /path/to/lure_turbo --out OUT
    python tests/gpu/turbo_gpu.py --model /path/to/lure_turbo --out OUT --bench

1. head: the engine's TurboHead vs the reference head (conformance/turbo/ref_fastcp.py) built from the same code predictor
   and turbo_head.safetensors, bf16, on real (talker hidden, cb0) inputs recorded from a qwen-tts generate() whose
   code_predictor.generate is patched to the reference head. Checked (a) free-running greedy and (b) teacher-forced on
   the reference's codes, so (b) isolates numerics. Reports max |logit diff| and greedy code agreement per position.
2. graph: TurboPredictorGraph CUDA-graph replay (B = max_batch) vs the eager head, greedy, same inputs.
3. engine: greedy engine frames (graphs) vs that patched qwen-tts generate (eager), exact match prefix per case.
   bf16 graph and eager kernels eventually flip near-ties, so a short prefix is expected.
Writes OUT/turbo_parity.json (or OUT/turbo_bench.json with --bench).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ENGINE = os.path.abspath(os.path.join(ROOT, "..", ".."))
for p in (ROOT, os.path.join(ENGINE, "core"), os.path.join(ENGINE, "conformance"), os.path.join(ENGINE, "conformance", "turbo")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

import cases as C  # noqa: E402


def _shim():
    import transformers.masking_utils as mu
    orig = mu.create_causal_mask
    if getattr(orig, "_lure_shim", False):
        return

    def shim(*a, input_embeds=None, **kw):
        if input_embeds is not None:
            kw["inputs_embeds"] = input_embeds
        kw.pop("cache_position", None)
        return orig(*a, **kw)
    shim._lure_shim = True
    mu.create_causal_mask = shim


def run(a):
    _shim()
    import lure_cuda
    import ref_fastcp as R
    from safetensors.torch import load_file
    from lure_cuda.engine import Request, drain
    from lure_cuda.sampling import SamplingParams, make_generator

    be = lure_cuda.load(a.model, turbo=1)
    assert be.turbo
    m, eng = be.m, be.engine
    head = eng.pgraph.head
    cp, e0 = m.talker.code_predictor, m.talker.get_input_embeddings()
    ref = R.make_fastcp(cp, e0, R.HEADS["turbo"], addon=True)
    ref.load_state_dict(load_file(os.path.join(a.model, "turbo_head.safetensors")))
    ref = ref.to(device=be.device, dtype=be.dtype).eval()
    voice = be.make_voice(C.REF_WAV, C.REF_TEXT, voice_id=C.VOICE_ID)
    rc, spk = be._voice_tensors(voice)
    sp = SamplingParams(do_sample=False, subtalker_do_sample=False)
    res = {"gpu": torch.cuda.get_device_name(), "cases": {}}
    recorded = []
    for k in [x for x in a.cases.split(",") if x]:
        text, direction, _ = C.CASES[k]
        p = be.prompt_ids(text, voice.ref_text, direction)
        with torch.inference_mode():
            emb, pad = be.prefill(p, voice, "english")
        r = Request(prefill=emb, tts_pad=pad, sampling=sp, max_new_tokens=a.max_new, generator=make_generator(0, be.device))
        eng.submit(r)
        got = list(drain(r, timeout=600))

        def gen(inputs_embeds=None, **kw):
            recorded.append(inputs_embeds.detach().clone())
            return types.SimpleNamespace(sequences=ref.sample_codes(inputs_embeds, R.temp_fn("0")))
        orig = cp.generate
        cp.generate = gen
        try:
            dev = rc.device
            vcp = dict(ref_code=[rc], ref_spk_embedding=[spk], x_vector_only_mode=[False], icl_mode=[True])
            with torch.inference_mode():
                codes, _ = m.generate(input_ids=[torch.tensor([p.input_ids], device=dev)],
                                      instruct_ids=[None if p.instruct_ids is None else torch.tensor([p.instruct_ids], device=dev)],
                                      ref_ids=[torch.tensor([p.ref_ids], device=dev)], voice_clone_prompt=vcp, languages=["english"],
                                      non_streaming_mode=True, max_new_tokens=a.max_new, do_sample=False, subtalker_dosample=False,
                                      repetition_penalty=1.05)
        finally:
            cp.generate = orig
        want = codes[0].tolist()
        n = 0
        while n < min(len(got), len(want)) and got[n] == want[n]:
            n += 1
        cb0 = 0
        while cb0 < min(len(got), len(want)) and got[cb0][0] == want[cb0][0]:
            cb0 += 1
        res["cases"][k] = dict(engine_frames=len(got), ref_frames=len(want), identical=got == want, match_prefix=n, cb0_match_prefix=cb0)
        print(k, res["cases"][k], flush=True)

    # head-level parity on the recorded real inputs
    X = torch.cat(recorded, 0)  # [N, 2, H]
    N = X.shape[0]
    logs_r, logs_m = [], []
    orig_s = R.sample

    def rec_r(logits, temp, top_k=50):
        logs_r.append(logits.float())
        return logits.argmax(-1)
    R.sample = rec_r
    try:
        with torch.inference_mode():
            codes_r = torch.cat([ref.sample_codes(X[i:i + 64], R.temp_fn("0")) for i in range(0, N, 64)], 0)
    finally:
        R.sample = orig_s
    st = head.make_static(X.device, X.dtype)

    def rec_m(logits, i):
        logs_m.append(logits.float())
        return logits.argmax(-1)
    with torch.inference_mode():
        codes_m = torch.cat([head.frame(X[i:i + 64], rec_m, st) for i in range(0, N, 64)], 0)
    LR = torch.cat([torch.stack(logs_r[j * 15:(j + 1) * 15], 1) for j in range(len(logs_r) // 15)], 0)
    LM = torch.cat([torch.stack(logs_m[j * 15:(j + 1) * 15], 1) for j in range(len(logs_m) // 15)], 0)
    d = (LR - LM).abs()
    res["head"] = dict(frames=int(N), max_abs_logit_diff=float(d.max()), mean_abs_logit_diff=float(d.mean()),
                       logit_scale=float(LR.abs().mean()), codes_identical_frac=float((codes_r == codes_m).float().mean()),
                       per_position_agree=[round(float(x), 4) for x in (codes_r == codes_m).float().mean(0)])
    # teacher-forced: the engine head fed the reference's own codes, so differences are numerics only (no divergence)
    logs_f = []

    def rec_f(logits, i):
        logs_f.append(logits.float())
        return forced_now[:, i]
    with torch.inference_mode():
        for i in range(0, N, 64):
            forced_now = codes_r[i:i + 64]
            head.frame(X[i:i + 64], rec_f, st)
    LF = torch.cat([torch.stack(logs_f[j * 15:(j + 1) * 15], 1) for j in range(len(logs_f) // 15)], 0)
    df = (LR - LF).abs()
    res["head_forced"] = dict(max_abs_logit_diff=float(df.max()), mean_abs_logit_diff=float(df.mean()),
                              argmax_agree=float((LR.argmax(-1) == LF.argmax(-1)).float().mean()),
                              bitwise_identical_frac=float((LR == LF).float().mean()))
    # graph replay vs eager on B rows
    pg = eng.pgraph
    B = pg.B
    pg_rows = X[:B]
    saved = (pg.do_sample.clone(),)
    with torch.inference_mode():
        pg.do_sample.zero_()
        g_codes = pg.run(pg_rows) if pg.graph is not None else None
        e_codes = head.frame(pg_rows, lambda lg, i: lg.float().argmax(-1), st)
        pg.do_sample.copy_(saved[0])
    res["graph"] = dict(captured=pg.graph is not None, rows=B,
                        identical=None if g_codes is None else bool(torch.equal(g_codes, e_codes)),
                        agree_frac=None if g_codes is None else float((g_codes == e_codes).float().mean()))
    be.close()
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "turbo_parity.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def bench(a):
    """Engine-only speed, no codec decode: CP step alone and whole engine frames/s at B = 1 and 8, Turbo head vs the stock
    15-pass predictor graph, swapped inside one loaded engine (same talker, same GPU). Writes OUT/turbo_bench.json."""
    import threading
    import time
    import lure_cuda
    from lure_cuda.engine import Request, drain
    from lure_cuda.graphs import BatchedPredictorGraph
    from lure_cuda.sampling import resolve_sampling, make_generator

    be = lure_cuda.load(a.model, turbo=1)
    eng = be.engine
    tg = eng.pgraph
    cp = be.m.talker.code_predictor
    sg = BatchedPredictorGraph(cp, cp.model.config, be.m.config.talker_config.hidden_size, eng.B, be.device, be.dtype, True)
    sg.capture()
    voice = be.make_voice(C.REF_WAV, C.REF_TEXT, voice_id=C.VOICE_ID)
    text, direction, _ = C.CASES["a"]
    res = {"gpu": torch.cuda.get_device_name(), "max_batch": eng.B}
    x = torch.randn(eng.B, 2, be.m.config.talker_config.hidden_size, device=be.device, dtype=be.dtype)
    for name, g in (("turbo", tg), ("stock", sg)):
        for _ in range(5):
            g.run(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            g.run(x)
        torch.cuda.synchronize()
        res[f"cp_ms_{name}"] = (time.perf_counter() - t0) / 50 * 1000
    sp = resolve_sampling(None, min_new_tokens=200)
    p = be.prompt_ids(text, voice.ref_text, direction)
    with torch.inference_mode():
        emb, pad = be.prefill(p, voice, "english")
    for name, g in (("turbo", tg), ("stock", sg)):
        eng.pgraph = g
        for n in (1, 8):
            reqs = [Request(prefill=emb, tts_pad=pad, sampling=sp, max_new_tokens=201, generator=make_generator(i, be.device)) for i in range(n)]
            outs = {}

            def w(i):
                outs[i] = len(list(drain(reqs[i], timeout=600)))
            ts = [threading.Thread(target=w, args=(i,)) for i in range(n)]
            t0 = time.perf_counter()
            for r in reqs:
                eng.submit(r)
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            wall = time.perf_counter() - t0
            fr = sum(outs.values())
            res[f"engine_{name}_B{n}"] = dict(frames=fr, wall_s=round(wall, 3), frames_per_s=round(fr / wall, 2),
                                              ms_per_step=round(wall / (fr / n) * 1000, 2))
            print(name, n, res[f"engine_{name}_B{n}"], flush=True)
    eng.pgraph = tg
    be.close()
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "turbo_bench.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="a,b,c,t1")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--bench", action="store_true", help="engine-only speed instead of parity")
    args = ap.parse_args()
    if args.bench:
        bench(args)
    else:
        run(args)
