"""Generate the reference prompt dumps from the prompt builder in qwen-tts 0.1.1 (no weights, no generation).

Two dumps:
  prompt_ids.json    the Lure call: `Qwen3TTSForConditionalGeneration.generate` with non_streaming_mode=True, an ICL
                     voice_clone_prompt and instruct_ids, over the conformance cases.
  prompt_modes.json  every mode and layout of stock Qwen3-TTS, through the public qwen-tts wrappers
                     (generate_voice_clone, generate_custom_voice, generate_voice_design) on the configs in configs/.
                     Base + instruct is not reachable through generate_voice_clone, so those entries call
                     model.generate directly, like prompt_ids.json.

How: the real qwen-tts code runs on a stub talker whose embeddings are symbolic one-hot-ish vectors (float64):
  dim 0       = text id + 1          (text_projection(text_embedding(id)), projection = identity)
  dim 1       = codec id + 1         (talker codec embedding; also ref codebook 0)
  dim 2..16   = ref code g + 1       (code_predictor.codec_embedding[g-1], g = 1..15)
  dim 17      = 1                    (speaker x-vector)
talker.generate is intercepted, so the captured inputs_embeds decode back to exact [text, codec] rows. Token ids come from the
qwen-tts processor (AutoProcessor, fix_mistral_regex=True) loaded from the LURE checkpoint dir (tokenizer files only).

Run (CPU, < 1 GB, qwen-tts 0.1.1 from requirements-ref.txt):
    python gen_reference.py /path/to/lure                 # both dumps + tokenizer_ids.json
    python gen_reference.py /path/to/lure --only modes    # prompt_modes.json only (tokenizer files are enough)
"""
import argparse
import hashlib
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoConfig, AutoProcessor  # noqa: E402

import qwen_tts  # noqa: E402
from qwen_tts import Qwen3TTSModel  # noqa: E402
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor  # noqa: E402

from lure_tts.text import segment_text  # noqa: E402
from cases import CASES, REF_TEXT  # noqa: E402

D = 18
N_REF = 6  # symbolic reference frames; real voices have T_ref = codec frames of the ref wav (see README)
DT = torch.float64


def fake_ref_codes(n=N_REF):
    t = torch.arange(n).view(-1, 1)
    g = torch.arange(16).view(1, -1)
    return ((37 * t + 101 * g + 7) % 2048).long()


class Stop(Exception):
    pass


def make_stub(config, calls=None):
    def text_emb(ids):
        out = torch.zeros(*ids.shape, D, dtype=DT)
        out[..., 0] = ids.to(DT) + 1
        return out

    def codec_emb(ids):
        out = torch.zeros(*ids.shape, D, dtype=DT)
        out[..., 1] = ids.to(DT) + 1
        return out

    def cp_emb(g):
        def f(ids):
            out = torch.zeros(*ids.shape, D, dtype=DT)
            out[..., 1 + g] = ids.to(DT) + 1
            return out
        return f

    cap = {}
    talker = types.SimpleNamespace(
        device=torch.device("cpu"), dtype=DT,
        config=types.SimpleNamespace(num_code_groups=config.talker_config.num_code_groups),
        get_text_embeddings=lambda: text_emb,
        text_projection=lambda x: x,
        get_input_embeddings=lambda: codec_emb,
        code_predictor=types.SimpleNamespace(get_input_embeddings=lambda: [cp_emb(g) for g in range(1, 16)]),
    )

    def gen(**kw):
        cap.update(kw)
        raise Stop()
    talker.generate = gen
    m = types.SimpleNamespace(config=config, talker=talker, device=torch.device("cpu"),
                              tts_model_type=config.tts_model_type, tts_model_size=config.tts_model_size,
                              tokenizer_type=config.tokenizer_type)
    # the same lists Qwen3TTSForConditionalGeneration.__init__ builds (used by the wrappers' validation)
    langs = ["auto"] + [k for k in config.talker_config.codec_language_id if "dialect" not in k]
    m.get_supported_languages = lambda: langs
    m.get_supported_speakers = lambda: config.talker_config.spk_id.keys()
    m.generate_speaker_prompt = types.MethodType(Qwen3TTSForConditionalGeneration.generate_speaker_prompt, m)
    m.generate_icl_prompt = types.MethodType(Qwen3TTSForConditionalGeneration.generate_icl_prompt, m)
    real = types.MethodType(Qwen3TTSForConditionalGeneration.generate, m)

    def generate(**kw):  # record what the wrapper passed, then run the real prompt builder
        if calls is not None:
            calls.clear()
            calls.update(kw)
        return real(**kw)
    m.generate = generate
    return m, cap


def decode_row(v, ref_codes):
    v = [int(round(float(x))) for x in v]
    text = v[0] - 1 if v[0] else None
    cps = v[2:17]
    if any(cps):
        frame = [v[1] - 1] + [c - 1 for c in cps]
        hits = [t for t in range(ref_codes.shape[0]) if ref_codes[t].tolist() == frame]
        assert len(hits) == 1, (frame, hits)
        assert not v[17]
        return [text, f"ref:{hits[0]}"]
    if v[17]:
        assert v[17] == 1 and not v[1]
        return [text, "spk"]
    return [text, v[1] - 1 if v[1] else None]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def gen_lure(model_dir, out_dir, processor):
    config = Qwen3TTSConfig.from_pretrained(model_dir)
    gd = json.load(open(os.path.join(model_dir, "generation_config.json")))
    stub, cap = make_stub(config)
    tts = Qwen3TTSModel(model=stub, processor=processor, generate_defaults=gd)
    ref_codes = fake_ref_codes()
    spk = torch.zeros(D, dtype=DT)
    spk[17] = 1

    entries = []

    def add(eid, text, direction, ref_text=REF_TEXT, language="English", source=""):
        item = qwen_tts.VoiceClonePromptItem(ref_code=ref_codes, ref_spk_embedding=spk, x_vector_only_mode=False, icl_mode=True,
                                             ref_text=ref_text)
        vcp = tts._prompt_items_to_voice_clone_prompt([item])
        ids = tts._tokenize_texts([tts._build_assistant_text(text)])
        ref_ids = [tts._tokenize_texts([tts._build_ref_text(item.ref_text)])[0]]
        ins = [tts._tokenize_texts([tts._build_instruct_text(direction)])[0] if direction else None]
        gk = tts._merge_generate_kwargs(max_new_tokens=4096)
        cap.clear()
        try:
            stub.generate(input_ids=ids, instruct_ids=ins, ref_ids=ref_ids, voice_clone_prompt=vcp, languages=[language],
                          non_streaming_mode=True, **gk)
        except Stop:
            pass
        emb = cap["inputs_embeds"][0]
        assert bool(cap["attention_mask"].all())
        rows = [decode_row(r, ref_codes) for r in emb]
        trailing = [decode_row(r, ref_codes) for r in cap["trailing_text_hidden"][0]]
        pad = decode_row(cap["tts_pad_embed"].reshape(-1), ref_codes)
        kw = {k: cap[k] for k in ("max_new_tokens", "min_new_tokens", "do_sample", "top_k", "top_p", "temperature", "subtalker_dosample",
                                  "subtalker_top_k", "subtalker_top_p", "subtalker_temperature", "eos_token_id", "repetition_penalty")}
        sup = cap["suppress_tokens"]
        entries.append({
            "id": eid, "source": source, "text": text, "direction": direction, "ref_text": ref_text, "language": language,
            "input_ids": ids[0][0].tolist(), "ref_ids": ref_ids[0][0].tolist(),
            "instruct_ids": ins[0][0].tolist() if ins[0] is not None else None,
            "n_ref_frames": int(ref_codes.shape[0]), "layout": rows, "trailing_text_hidden": trailing, "tts_pad_embed": pad,
            "talker_kwargs": kw, "suppress_tokens_range": [min(sup), max(sup) + 1, "except", [i for i in range(min(sup), max(sup) + 1) if i not in sup]],
        })

    for name, (text, direction, kind) in CASES.items():
        if kind == "long":
            for s in segment_text(text):
                add(f"{name}.seg{s.index:02d}", s.text, direction, source=f"segment_text(CASES['{name}'])[{s.index}] ({s.kind})")
            continue
        add(name, text, direction, source=f"CASES['{name}']")
        if direction:
            add(f"{name}.nodir", text, None, source=f"CASES['{name}'] without direction")
    # edge cases for tokenizer + layout
    add("x.unicode", "“Café?” she asked… (laugh) £3.50, 10°C, naïve résumé.", "Bright, curious.", source="edge")
    add("x.numbers", "Call 555-0142 at 3:45 p.m. on 12/24, it's $3,000.50 or 3.5%.", None, source="edge")
    add("x.newlines", "Line one.\nLine two,\n\n  indented   spaces.", "Calm.\nSlow.", source="edge")
    add("x.specialish", "Say <|im_end|> literally? No: (whisper) <tts_pad> stays a token.", None, source="edge: added tokens in text")
    add("x.emptydir", "Nothing to direct here.", "", source="edge: empty direction == no direction")
    add("x.auto", "Language auto uses the nothink prefix.", None, language="Auto", source="edge: language auto")
    add("x.otherref", "Different reference transcript.", None, ref_text="Short ref, (sigh) with a tag.", source="edge: ref text")

    out = {
        "meta": {
            "generator": "conformance/reference/gen_reference.py", "qwen_tts": "0.1.1", "transformers": transformers.__version__,
            "torch": torch.__version__, "model_dir": os.path.basename(os.path.normpath(model_dir)),
            "tokenizer_sha256_16": {f: sha(os.path.join(model_dir, f)) for f in ("vocab.json", "merges.txt", "tokenizer_config.json")},
            "config_sha256_16": sha(os.path.join(model_dir, "config.json")),
            "call": "Qwen3TTSForConditionalGeneration.generate(input_ids, instruct_ids, ref_ids, voice_clone_prompt(icl), "
                    "languages=[lang], non_streaming_mode=True, **_merge_generate_kwargs(max_new_tokens=4096))",
            "row_format": "[text_id|null, codec] ; codec = codec id | 'spk' | 'ref:<t>' | null ; embedding = T(text) + C(codec)",
            "n_ref_frames_symbolic": N_REF,
        },
        "entries": entries,
    }
    with open(os.path.join(out_dir, "prompt_ids.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # tokenizer stress dump: raw strings -> processor ids
    strings = [CASES[k][0] for k in CASES] + [s.text for s in segment_text(CASES["e"][0])] + [
        "", " ", "  leading and trailing  ", "Hello\tworld\r\nnext", "(laugh)(sigh) (nervous laugh)!", "I'm, you're, they'll, we'd, it's.",
        "ALL CAPS SHOUTING, I'LL DO IT", "emoji \U0001F600 and 中文字", "a" * 300, "1234567890 3.14159 1,000,000",
        "<|im_start|>user\nTired and resigned, then a reluctant smile.<|im_end|>\n", "<tts_text_bos_single><tts_text_bos>x",
        "é (decomposed) vs é", "tabs\t\tand   spaces   ", "Dr. Smith vs. Mr. Jones, etc.", "—dash–dash-dash",
    ]
    toks = []
    for s in strings:
        toks.append({"text": s, "ids": processor(text=s, return_tensors="pt", padding=True)["input_ids"][0].tolist()})
    with open(os.path.join(out_dir, "tokenizer_ids.json"), "w") as f:
        json.dump({"meta": out["meta"], "entries": toks}, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(entries)} prompt entries, {len(toks)} tokenizer entries")


def _ids(t):
    return t[0].tolist() if t is not None else None


def gen_modes(model_dir, out_dir, processor):
    """prompt_modes.json: every mode x layout through the public qwen-tts wrappers on the stock configs."""
    gd = json.load(open(os.path.join(model_dir, "generation_config.json")))
    ref_codes = fake_ref_codes()
    spk = torch.zeros(D, dtype=DT)
    spk[17] = 1
    entries = []
    models = {}

    def model(cfg_name):
        if cfg_name not in models:
            config = Qwen3TTSConfig.from_json_file(os.path.join(HERE, "configs", cfg_name))
            calls = {}
            stub, cap = make_stub(config, calls)
            models[cfg_name] = (Qwen3TTSModel(model=stub, processor=processor, generate_defaults=gd), stub, cap, calls)
        return models[cfg_name]

    def add(eid, cfg_name, mode, layout, text, language="English", ref_text=None, instruct=None, speaker=None,
            direct=False, note=None):
        tts, stub, cap, calls = model(cfg_name)
        ns = layout == "non_streaming"
        fn = {"custom_voice": "generate_custom_voice", "voice_design": "generate_voice_design"}.get(mode, "generate_voice_clone")
        call = (f"model.generate(voice_clone_prompt, instruct_ids, non_streaming_mode={ns})" if direct
                else f"{fn}(non_streaming_mode={ns})")
        cap.clear()
        try:
            if mode in ("icl", "xvec"):
                icl = mode == "icl"
                item = qwen_tts.VoiceClonePromptItem(ref_code=ref_codes if icl else None, ref_spk_embedding=spk,
                                                     x_vector_only_mode=not icl, icl_mode=icl, ref_text=ref_text)
                if direct:  # Base + instruct: the wrapper has no instruct argument
                    ids = tts._tokenize_texts([tts._build_assistant_text(text)])
                    ref_ids = [tts._tokenize_texts([tts._build_ref_text(ref_text)])[0] if icl else None]
                    ins = [tts._tokenize_texts([tts._build_instruct_text(instruct)])[0] if instruct else None]
                    stub.generate(input_ids=ids, instruct_ids=ins, ref_ids=ref_ids,
                                  voice_clone_prompt=tts._prompt_items_to_voice_clone_prompt([item]),
                                  languages=[language], non_streaming_mode=ns, **tts._merge_generate_kwargs())
                else:
                    tts.generate_voice_clone(text=text, language=language, voice_clone_prompt=[item],
                                             non_streaming_mode=ns)
            elif mode == "custom_voice":
                tts.generate_custom_voice(text=text, speaker=speaker, language=language, instruct=instruct,
                                          non_streaming_mode=ns)
            else:
                tts.generate_voice_design(text=text, instruct=instruct, language=language, non_streaming_mode=ns)
        except Stop:
            pass
        emb = cap["inputs_embeds"][0]
        assert bool(cap["attention_mask"].all())
        ins = (calls.get("instruct_ids") or [None])[0]
        ref = (calls.get("ref_ids") or [None])[0]
        entries.append({
            "id": eid, "config": cfg_name, "call": call, "mode": mode, "prompt_layout": layout, "text": text,
            "language": language.lower(), "ref_text": ref_text if mode == "icl" else None,
            "direction": instruct if ins is not None else None, "requested_direction": instruct, "speaker": speaker,
            "input_ids": _ids(calls["input_ids"][0]), "ref_ids": _ids(ref) if mode == "icl" else None,
            "instruct_ids": _ids(ins), "n_ref_frames": int(ref_codes.shape[0]) if mode == "icl" else 0,
            "layout": [decode_row(r, ref_codes) for r in emb],
            "trailing": [decode_row(r, ref_codes) for r in cap["trailing_text_hidden"][0]],
            "note": note,
        })

    B06, B17 = "Qwen3-TTS-12Hz-0.6B-Base.json", "Qwen3-TTS-12Hz-1.7B-Base.json"
    CV06, CV17 = "Qwen3-TTS-12Hz-0.6B-CustomVoice.json", "Qwen3-TTS-12Hz-1.7B-CustomVoice.json"
    VD17 = "Qwen3-TTS-12Hz-1.7B-VoiceDesign.json"
    a, t1 = CASES["a"][0], CASES["t1"][0]
    for lay in ("non_streaming", "streaming"):
        s = "ns" if lay == "non_streaming" else "s"
        # Base: ICL clone (text longer than the 7-row codec column, and shorter: the padded branch)
        add(f"base.icl.{s}", B06, "icl", lay, a, ref_text=REF_TEXT)
        add(f"base.icl.{s}.short", B06, "icl", lay, "Ok.", ref_text="Hi.")
        add(f"base.icl.{s}.tags", B06, "icl", lay, t1, ref_text=REF_TEXT)
        add(f"base.icl.{s}.auto", B06, "icl", lay, a, language="Auto", ref_text=REF_TEXT)
        add(f"base.icl.{s}.instruct", B06, "icl", lay, a, ref_text=REF_TEXT, instruct="Warm and slow.", direct=True)
        add(f"base17.icl.{s}", B17, "icl", lay, a, ref_text=REF_TEXT)
        # Base: x-vector only
        add(f"base.xvec.{s}", B06, "xvec", lay, a)
        add(f"base.xvec.{s}.one", B06, "xvec", lay, "Hi", language="Auto")
        add(f"base.xvec.{s}.instruct", B06, "xvec", lay, t1, instruct="Warm and slow.", direct=True)
        # CustomVoice
        add(f"cv17.{s}", CV17, "custom_voice", lay, a, speaker="vivian")
        add(f"cv17.{s}.instruct", CV17, "custom_voice", lay, a, speaker="Ryan", instruct="Cheerful, fast.")
        add(f"cv17.{s}.eric.chinese", CV17, "custom_voice", lay, "你好，今天天气不错。", language="Chinese", speaker="eric",
            note="dialect speaker + chinese: sichuan_dialect language id")
        add(f"cv17.{s}.dylan.auto", CV17, "custom_voice", lay, "你好，今天天气不错。", language="Auto", speaker="dylan",
            note="dialect speaker + auto: beijing_dialect id with the 4-row think prefix")
        add(f"cv17.{s}.eric.english", CV17, "custom_voice", lay, a, speaker="eric", note="dialect only for chinese/auto")
        add(f"cv17.{s}.ryan.auto", CV17, "custom_voice", lay, a, language="Auto", speaker="ryan")
        add(f"cv06.{s}", CV06, "custom_voice", lay, a, speaker="serena")
        add(f"cv06.{s}.instruct_dropped", CV06, "custom_voice", lay, a, speaker="serena", instruct="Cheerful.",
            note="qwen-tts drops instruct on the 0.6B CustomVoice (tts_model_size in '0b6')")
        # VoiceDesign
        add(f"vd17.{s}", VD17, "voice_design", lay, a, instruct="A deep, calm male voice, slow pace.")
        add(f"vd17.{s}.auto", VD17, "voice_design", lay, t1, language="Auto", instruct="Bright young female voice.")
        add(f"vd17.{s}.empty", VD17, "voice_design", lay, a, instruct="",
            note="qwen-tts accepts an empty instruct (no instruct rows); the Lure TTS API rejects it")
    out = {
        "meta": {
            "generator": "conformance/reference/gen_reference.py (gen_modes)", "qwen_tts": "0.1.1",
            "transformers": transformers.__version__, "torch": torch.__version__,
            "tokenizer_sha256_16": {f: sha(os.path.join(model_dir, f)) for f in ("vocab.json", "merges.txt", "tokenizer_config.json")},
            "configs": {c: sha(os.path.join(HERE, "configs", c)) for c in sorted(models)},
            "row_format": "[text_id|null, codec] ; codec = codec id | 'spk' | 'ref:<t>' | null ; embedding = T(text) + C(codec)",
            "trailing": "trailing_text_hidden rows; generated frame k adds trailing[k], then tts_pad once it runs out",
            "n_ref_frames_symbolic": N_REF,
        },
        "entries": entries,
    }
    with open(os.path.join(out_dir, "prompt_modes.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(entries)} mode entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="model folder (tokenizer files, config.json, generation_config.json)")
    ap.add_argument("--only", choices=("lure", "modes"), default=None)
    ap.add_argument("--out", default=HERE, help="output folder (default: this folder)")
    a = ap.parse_args()
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
    processor = AutoProcessor.from_pretrained(a.model_dir, fix_mistral_regex=True)
    if a.only in (None, "lure"):
        gen_lure(a.model_dir, a.out, processor)
    if a.only in (None, "modes"):
        gen_modes(a.model_dir, a.out, processor)


if __name__ == "__main__":
    main()
