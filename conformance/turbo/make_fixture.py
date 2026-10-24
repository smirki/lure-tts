"""Build the tiny-weight Turbo parity fixture from the reference implementation (ref_fastcp.py).

Needs torch + qwen-tts 0.1.1 + transformers 4.57.3 (make_fastcp calls create_causal_mask(input_embeds=...), which
transformers 5 renamed). CPU only, about 500 MB RSS, a few seconds:
    python conformance/turbo/make_fixture.py

Writes fixtures/tiny_turbo.safetensors and tiny_turbo.json:
    head.*           tiny random Turbo head state_dict (key names as turbo_head.safetensors), float32
    cemb.{i}.weight  CP codec embeddings 0..14 (the talker's code_predictor.model.codec_embedding), float32
    x2               [B, 2, H] = [talker hidden, emb0(cb0)] inputs
    greedy_codes     [B, 15] reference sample_codes with every temperature 0 (argmax)
    greedy_logits    [B, 15, V] the logits the reference sampled from on that path
    forced_codes     [B, 15] random codes forced at every position (exercises the add-on off the argmax path)
    forced_logits    [B, 15, V] the reference logits along the forced path
Computed in float64, stored float32.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "backends", "cuda"))  # for lure_cuda._compat

TINY = dict(hidden=64, layers=2, heads=4, kv_heads=2, head_dim=16, inter=96, vocab=512, talker_vocab=3072, B=3, seed=7)


def tiny_cp(dtype):
    """A tiny qwen-tts code predictor + talker codec embedding (talker hidden == CP hidden, as in the 0.6B)."""
    import torch
    try:
        from lure_cuda._compat import import_qwen_tts
        import_qwen_tts()
    except Exception:
        import qwen_tts  # noqa: F401
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerCodePredictorModelForConditionalGeneration

    t = TINY
    cpc = Qwen3TTSTalkerCodePredictorConfig(hidden_size=t["hidden"], intermediate_size=t["inter"], num_hidden_layers=t["layers"],
                                            num_attention_heads=t["heads"], num_key_value_heads=t["kv_heads"], head_dim=t["head_dim"],
                                            vocab_size=t["vocab"], num_code_groups=16, rope_theta=1000000, rms_norm_eps=1e-6,
                                            layer_types=["full_attention"] * t["layers"], max_position_embeddings=65536)
    cpc._attn_implementation = "sdpa"
    tc = Qwen3TTSTalkerConfig(hidden_size=t["hidden"], vocab_size=t["talker_vocab"], num_code_groups=16)
    torch.manual_seed(t["seed"])
    cp = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(cpc, tc).to(dtype).eval()
    emb0 = torch.nn.Embedding(t["talker_vocab"], t["hidden"]).to(dtype)
    return cp, emb0


def randomize_head(head, seed):
    """Every head parameter random (the reference zero-inits the add-on's w2 and addon_pos, which would hide it)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in head.named_parameters():
            if n.endswith("norm.weight") or ".n1.weight" in n or ".n2.weight" in n or ".n3.weight" in n:
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g, dtype=torch.float64))
            elif p.dim() == 1:
                p.copy_(0.1 * torch.randn(p.shape, generator=g, dtype=torch.float64))
            elif n.startswith("lm_head"):
                p.copy_(0.5 * torch.randn(p.shape, generator=g, dtype=torch.float64))
            else:
                p.copy_(0.15 * torch.randn(p.shape, generator=g, dtype=torch.float64))


def build_reference(dtype=None):
    """-> (head, cp, emb0): the reference Turbo head with random weights, on CPU."""
    import torch
    import ref_fastcp as R
    dtype = dtype or torch.float64
    cp, emb0 = tiny_cp(dtype)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for e in cp.model.codec_embedding:
            e.weight.copy_(torch.randn(e.weight.shape, generator=g, dtype=torch.float64))
        emb0.weight.copy_(torch.randn(emb0.weight.shape, generator=g, dtype=torch.float64))
    head = R.make_fastcp(cp, emb0, R.HEADS["turbo"], addon=True).to(dtype).eval()
    randomize_head(head, 5)
    return head, cp, emb0


def run_reference(head, x2, forced=None):
    """Reference sample_codes with every temperature 0, recording the logits at each position.
    forced (B, 15): instead of argmax, the recorded position takes the forced code (same code path, so the add-on sees it)."""
    import torch
    import ref_fastcp as R
    logs = []
    orig = R.sample

    def rec(logits, temp, top_k=50):
        i = len(logs)
        logs.append(logits.detach().clone())
        if forced is not None:
            return forced[:, i].clone()
        return orig(logits, temp, top_k)
    R.sample = rec
    try:
        codes = head.sample_codes(x2, lambda k, first: 0.0)
    finally:
        R.sample = orig
    assert len(logs) == 15
    return codes, torch.stack(logs, 1)


def main():
    import torch
    from safetensors.torch import save_file
    t = TINY
    head, cp, emb0 = build_reference(torch.float64)
    g = torch.Generator().manual_seed(3)
    B, H = t["B"], t["hidden"]
    h = torch.randn(B, H, generator=g, dtype=torch.float64)
    cb0 = torch.randint(0, 2048, (B,), generator=g)
    x2 = torch.stack([h, emb0(cb0)], 1).detach()
    gc, gl = run_reference(head, x2)
    forced = torch.randint(0, t["vocab"], (B, 15), generator=g)
    fc, fl = run_reference(head, x2, forced)
    assert torch.equal(fc, forced)
    out = {f"head.{k}": v.detach().float().contiguous() for k, v in head.state_dict().items()}
    for i, e in enumerate(cp.model.codec_embedding):
        out[f"cemb.{i}.weight"] = e.weight.detach().float().contiguous()
    out.update(x2=x2.float().contiguous(), greedy_codes=gc.to(torch.int32).contiguous(), greedy_logits=gl.float().contiguous(),
               forced_codes=fc.to(torch.int32).contiguous(), forced_logits=fl.float().contiguous())
    os.makedirs(os.path.join(HERE, "fixtures"), exist_ok=True)
    save_file(out, os.path.join(HERE, "fixtures", "tiny_turbo.safetensors"))
    meta = dict(TINY, groups=[[1, 3], [4, 7], [8, 15]], addon_hidden=512, layernorm_eps=1e-5, rms_norm_eps=1e-6, rope_theta=1000000,
                dtype="computed float64, stored float32", reference="conformance/turbo/ref_fastcp.py",
                keys=sorted(k for k in out if k.startswith("head.")))
    json.dump(meta, open(os.path.join(HERE, "fixtures", "tiny_turbo.json"), "w"), indent=1)
    print("greedy codes", gc.tolist())
    print("wrote", len(out), "tensors")


if __name__ == "__main__":
    main()
