"""Lure TTS on Apple silicon: adapter over mlx-audio (pinned 402e3b9e4bd9bbecedc917b290896e5588c76f58).

Uses mlx-audio's Qwen3-TTS modules (talker, code predictor, speech tokenizer encoder/decoder, speaker encoder) and runs
its own generation loop. The prefill and the per-frame text rows come from the core's prompt (req.build_prompt, embedded
by embed.py) and sampling follows the backend contract (sampling.py), with the codebook-0 repetition penalty over the
full segment history, kept on device.

Every Qwen3-TTS 12Hz checkpoint loads: 0.6B and 1.7B, Base, CustomVoice and VoiceDesign, plus the Lure family. The
1.7B code predictor input goes through small_to_mtp_projection (mlx-audio creates it whenever the talker and code
predictor hidden sizes differ).

Options (Lure.load(..., backend="mlx", **opts) or `lure say --opt key=value`):
  compile_cp=False      mx.compile the per-frame 15-step code predictor (cp.CompiledCodePredictor)
  pipeline=False        async_eval pipelining: frame t+1's graph is built and queued before frame t is read back
                        (costs one wasted frame at EOS)
  cache_limit_mb=512    mx.set_cache_limit
  memory_limit_gb=None  mx.set_memory_limit
  prime_frames=None     icl only: reference frames used to prime the streaming decoder (None = all, 0 = none)
  dtype="bfloat16"      compute dtype (weights are cast if they are not already)
  turbo=auto            Lure Turbo head (turbo.TurboCodePredictor, 0.6B Lure family only): auto = iff
                        model_dir/turbo.json exists, 1 = required, 0 = original 15-step code predictor.
                        turbo_head=PATH loads another head file. compile_cp=1 compiles the Turbo per-frame graph instead.
quant: None | "q8" (talker + CP linears 8-bit) | "mix" (talker 6-bit, CP 8-bit, text embedding 4-bit). Experimental, untested.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Iterator, Optional

import numpy as np

from lure_tts.backends.base import Chunk, GenerateRequest, SegmentInfo
from lure_tts.defaults import SAMPLE_RATE
from lure_tts.prompt import Prompt, build_prompt, build_prompt_ids

MLX_AUDIO_COMMIT = "402e3b9e4bd9bbecedc917b290896e5588c76f58"
CLONE_MODES = ("icl", "xvec")


class MLXBackend:
    name = "mlx"
    sample_rate = SAMPLE_RATE

    def __init__(self, model, model_dir: str, opts: dict, spec=None):
        import mlx.core as mx
        if spec is None:
            from lure_tts.model import inspect
            spec = inspect(model_dir)
        self.mx = mx
        self.model, self.model_dir, self.opts, self.spec = model, model_dir, opts, spec
        self.talker = model.talker
        self.ids = spec.ids
        self.dtype = getattr(mx, opts.get("dtype", "bfloat16"))
        self._voices: OrderedDict = OrderedDict()
        self._cps: dict = {}
        self.turbo_cfg = None
        self.turbo_head = None
        self.tts_pad_embed = self._text_embed([self.ids.tts_pad])  # [1, 1, D]
        V = self.ids.codec_vocab_size
        sup = np.zeros((1, V), np.float32)
        sup[0, self.ids.suppress_tokens()] = -np.inf
        self.suppress = mx.array(sup)
        eos = np.zeros((1, V), np.float32)
        eos[0, self.ids.codec_eos] = -np.inf
        self.eos_mask = mx.array(eos)
        self.vocab_iota = mx.arange(V)[None]
        mx.eval(self.tts_pad_embed, self.suppress, self.eos_mask)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, model_dir: str, quant: Optional[str] = None, *, spec=None, **opts) -> "MLXBackend":
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_audio.tts.utils import load_model
        if spec is None:
            from lure_tts.model import inspect
            spec = inspect(model_dir, turbo=opts.get("turbo"), turbo_head=opts.get("turbo_head"))
        mx.set_cache_limit(int(opts.get("cache_limit_mb", 512)) * 2**20)
        if opts.get("memory_limit_gb"):
            mx.set_memory_limit(int(float(opts["memory_limit_gb"]) * 2**30))
        model = load_model(model_dir)
        model.tokenizer = None  # the core tokenizes; mlx-audio's text tokenizer is never used here
        check_model(model, spec)
        dtype = getattr(mx, opts.get("dtype", "bfloat16"))
        if any(v.dtype != dtype for _, v in tree_flatten(model.talker.parameters())):
            model.talker.set_dtype(dtype)
        if quant:
            _quantize(model, quant)
        mx.eval(model.parameters())
        be = cls(model, model_dir, dict(opts, quant=quant), spec=spec)
        be._use_turbo(spec.turbo)
        return be

    def _setup_turbo(self, turbo=None, turbo_head=None):
        """Resolve the Turbo head from model_dir (lure_tts.turbo.resolve: turbo.json present, or turbo=1) and use it."""
        from lure_tts.turbo import resolve
        return self._use_turbo(resolve(self.model_dir, turbo=turbo, turbo_head=turbo_head))

    def _use_turbo(self, cfg):
        head = None
        if cfg is not None:
            if self.talker.config.hidden_size != self.talker.code_predictor.config.hidden_size:
                raise ValueError("the Turbo head exists only for the 0.6B Lure family; this checkpoint has talker "
                                 f"hidden {self.talker.config.hidden_size}")
            from .turbo import load_head_for
            head = load_head_for(self.talker, cfg, self.talker.codec_head.weight.dtype)
        self.turbo_cfg, self.turbo_head = cfg, head
        self._cps.clear()
        return cfg

    @property
    def turbo(self) -> bool:
        return self.turbo_head is not None

    def close(self):
        self._voices.clear()
        self._cps.clear()
        self.turbo_head = None
        self.mx.clear_cache()

    # ------------------------------------------------------------------ voices
    def encode_voice(self, ref_wav: np.ndarray, ref_text: str):
        if self.spec.variant != "base" or getattr(self.model, "speaker_encoder", None) is None:
            raise ValueError(f"{self.spec.name} has no speaker encoder: only Base checkpoints clone a voice")
        mx = self.mx
        a = mx.array(np.asarray(ref_wav, dtype=np.float32))
        codes = self.model.speech_tokenizer.encode(a[None, None, :])  # [1, 16, T]
        spk = self.model.extract_speaker_embedding(a)  # [1, enc_dim]
        mx.eval(codes, spk)
        out = np.array(codes[0]).T.astype(np.int32), np.array(spk.astype(mx.float32)).reshape(-1)
        mx.clear_cache()
        return out

    def _voice_tensors(self, req: GenerateRequest):
        """(spk [D], refs [T, D] | None) for icl and xvec; (None, None) for the preset and designed voices."""
        if req.mode not in CLONE_MODES:
            return None, None
        voice = req.voice
        if voice is None:
            raise ValueError(f"mode {req.mode} needs a voice")
        hidden = self.talker.config.hidden_size
        if np.asarray(voice.spk_emb).size != hidden:
            raise ValueError(f"voice x-vector has {np.asarray(voice.spk_emb).size} dims, this model needs {hidden}: "
                             "encode the voice with this model")
        icl = req.mode == "icl"
        key = (voice.voice_id, voice.meta.get("wav_sha256"), voice.n_ref_frames, icl)
        if key not in self._voices:
            from .embed import ref_block
            spk = self.mx.array(np.asarray(voice.spk_emb, np.float32)).astype(self.dtype)
            refs = ref_block(self.talker, voice.ref_codes, self.dtype) if icl else None
            self.mx.eval(spk, *([refs] if icl else []))
            self._voices[key] = (spk, refs)
            while len(self._voices) > 8:
                self._voices.popitem(last=False)
        self._voices.move_to_end(key)
        return self._voices[key]

    def _cp(self, req):
        from .cp import CompiledCodePredictor, EagerCodePredictor
        s = req.sampling
        key = (bool(self.opts.get("compile_cp")), s.subtalker_temperature, s.subtalker_top_k, s.subtalker_top_p, s.subtalker_do_sample)
        if key not in self._cps and self.turbo_head is not None:
            from .turbo import TurboCodePredictor
            self._cps[key] = TurboCodePredictor(self.talker, self.turbo_head, s.subtalker_temperature, s.subtalker_top_k,
                                                s.subtalker_top_p, s.subtalker_do_sample, compile=key[0])
        if key not in self._cps:
            cls = CompiledCodePredictor if key[0] else EagerCodePredictor
            self._cps[key] = cls(self.talker, s.subtalker_temperature, s.subtalker_top_k, s.subtalker_top_p, s.subtalker_do_sample)
        return self._cps[key]

    # ------------------------------------------------------------------ prompt
    def prompt(self, req: GenerateRequest) -> Prompt:
        """The core's prompt for this request, after checking that this checkpoint runs the mode."""
        if req.mode not in self.spec.modes:
            raise ValueError(f"{self.spec.name} runs modes {self.spec.modes}, not {req.mode!r}")
        if req.mode == "custom_voice" and req.speaker_id is not None and req.speaker_id != self.ids.speaker_id(req.speaker):
            raise ValueError(f"speaker_id {req.speaker_id} does not match speaker {req.speaker!r}")
        return req.build_prompt(self.ids)

    def dump_layout(self, text, ref_text, direction, n_ref_frames, language="english", mode="icl",
                    layout="non_streaming", speaker=None):
        """Prompt-check hook (conformance/runner.py): the rows generate() embeds, from the same core builder."""
        if not hasattr(self, "_tok"):
            from lure_tts.tokenizer import QwenTokenizer
            self._tok = QwenTokenizer.from_dir(self.model_dir)
        p = build_prompt_ids(self._tok, text, ref_text if mode == "icl" else None, direction)
        pr = build_prompt(p, self.ids, mode=mode, layout=layout, language=language,
                          n_ref_frames=n_ref_frames if mode == "icl" else 0, speaker=speaker)
        return {"layout": pr.prefill, "trailing": pr.trailing}

    def _text_embed(self, ids):
        mx = self.mx
        t = mx.array([list(ids)], dtype=mx.int32)
        return self.talker.text_projection(self.talker.get_text_embeddings()(t)).astype(self.dtype)  # [1, n, D]

    def _trailing(self, prompt: Prompt):
        """Per-frame text inputs T(trailing[k]) as [1, n, D], or None when every row is tts_pad (both non_streaming
        layouts), so those frames add the precomputed T(tts_pad) exactly as before."""
        ids = [r[0] for r in prompt.trailing]
        if all(i == self.ids.tts_pad for i in ids):
            return None
        out = self._text_embed(ids)
        self.mx.eval(out)
        return out

    # ------------------------------------------------------------------ generation
    def generate(self, req: GenerateRequest) -> Iterator[Chunk]:
        from .embed import embed_rows
        from .sampling import gumbel, repetition_penalty, sample
        mx = self.mx
        t0 = time.time()
        prompt = self.prompt(req)
        if req.seed is not None:
            mx.random.seed(int(req.seed))
        spk, refs = self._voice_tensors(req)
        icl = req.mode == "icl"
        embeds = embed_rows(self.talker, prompt.prefill, spk, refs, self.dtype)
        trailing = self._trailing(prompt)
        n_trailing = 0 if trailing is None else int(trailing.shape[1])
        s = req.sampling
        cp = self._cp(req)
        cache = self.talker.make_cache()
        V = self.ids.codec_vocab_size
        eos = self.ids.codec_eos
        n_cp = self.ids.num_code_groups - 1
        cp_vocab = self.talker.code_predictor.config.vocab_size
        state = {"seen": mx.zeros((1, V), dtype=mx.bool_)}

        def text_in(k):
            """Text side of generated frame k: trailing[k], then tts_pad."""
            return trailing[:, k:k + 1] if k < n_trailing else self.tts_pad_embed

        def step(inp, i):
            logits, hidden = self.talker(inp, cache=cache)
            lg = logits[:, -1, :].astype(mx.float32)
            lg = repetition_penalty(lg, state["seen"], s.repetition_penalty)
            lg = lg + self.suppress
            if i < s.min_new_tokens:
                lg = lg + self.eos_mask
            cb0 = sample(lg, gumbel((1, V)), s.temperature, s.top_k, s.top_p, s.do_sample)  # [1]
            codes15, emb = cp(hidden[:, -1:, :], cb0, gumbel((n_cp, cp_vocab)))
            state["seen"] = state["seen"] | (self.vocab_iota == cb0[:, None])
            return cb0, codes15, emb + text_in(i)

        frames: list = []
        ended = False
        pipeline = bool(self.opts.get("pipeline"))
        stream = req.stream
        dec = self.model.speech_tokenizer.decoder
        decoded, ttfa, n_out = 0, None, 0
        if stream:
            dec.reset_streaming_state()
            if icl:
                self._prime_decoder(dec, req.voice)
        step_frames = req.first_chunk_frames

        def emit(upto):
            nonlocal decoded, ttfa, n_out, step_frames
            codes = mx.stack([mx.concatenate([c0, c15]) for c0, c15 in frames[decoded:upto]], axis=0)  # [n, 16]
            wav = dec.streaming_step(codes.T[None].astype(mx.int32))  # [1, 1, n*1920]
            pcm = np.array(wav[0, 0].astype(mx.float32))
            decoded = upto
            step_frames = req.chunk_frames
            if ttfa is None:
                ttfa = time.time() - t0
            n_out += len(pcm)
            return pcm

        cur = step(embeds, 0)
        if pipeline:
            mx.async_eval(*cur)
        for i in range(req.max_new_tokens):
            nxt = None
            if pipeline and i + 1 < req.max_new_tokens:
                nxt = step(cur[2], i + 1)
                mx.async_eval(*nxt)
            elif not pipeline:
                mx.eval(*cur)
            cb0, codes15, nin = cur
            if int(cb0.item()) == eos:
                ended = True
                break
            frames.append((cb0, codes15))
            if stream and len(frames) - decoded >= step_frames:
                yield Chunk(emit(len(frames)))
            if i + 1 >= req.max_new_tokens:
                break
            cur = nxt if pipeline else step(nin, i + 1)
        if stream:
            if len(frames) > decoded:
                yield Chunk(emit(len(frames)))
            dec.reset_streaming_state()
            pcm = np.zeros(0, np.float32)
        else:
            pcm = self._decode_whole(frames, req.voice if icl else None)
            n_out = len(pcm)
            ttfa = time.time() - t0
        info = SegmentInfo(frames=len(frames), ended_by_eos=ended, samples=n_out, ttfa_s=ttfa, wall_s=time.time() - t0,
                           extra={"peak_gb": round(mx.get_peak_memory() / 2**30, 3), "compile_cp": bool(self.opts.get("compile_cp")),
                                  "turbo": self.turbo, "pipeline": pipeline, "prefill_len": int(embeds.shape[1]),
                                  "mode": req.mode, "layout": prompt.layout})
        cache = None  # release the KV cache before clear_cache()
        mx.clear_cache()
        yield Chunk(pcm, info)

    def _prime_decoder(self, dec, voice):
        n = self.opts.get("prime_frames")
        T = voice.n_ref_frames
        n = T if n is None else max(0, min(int(n), T))
        if n == 0:
            return
        codes = self.mx.array(voice.ref_codes[T - n:].T[None].astype(np.int32))  # [1, 16, n]
        self.mx.eval(dec.streaming_step(codes))  # output discarded: fills conv buffers + decoder KV with ref context

    def _decode_whole(self, frames, voice=None):
        """voice set (icl): decode [ref ; gen] and drop int(T_ref / T_total * n_samples) leading samples, as qwen-tts.
        voice None (xvec, custom_voice, voice_design): decode the generated codes alone."""
        mx = self.mx
        if not frames:
            return np.zeros(0, np.float32)
        gen = mx.stack([mx.concatenate([c0, c15]) for c0, c15 in frames], axis=0).astype(mx.int32)  # [T_gen, 16]
        n_ref = voice.n_ref_frames if voice is not None else 0
        full = mx.concatenate([mx.array(voice.ref_codes.astype(np.int32)), gen], axis=0) if n_ref else gen
        full = full[None]  # [1, T, 16]
        audio, _ = self.model.speech_tokenizer.decode(full)
        audio = audio[0]
        # decoded length = all frames * 1920, as qwen-tts
        valid = int(full.shape[1]) * self.model.speech_tokenizer.decode_upsample_rate
        if 0 < valid < audio.shape[0]:
            audio = audio[:valid]
        cut = int(n_ref / max(full.shape[1], 1) * audio.shape[0])
        out = np.array(audio[cut:].astype(mx.float32))
        mx.clear_cache()
        return out


def check_model(model, spec) -> None:
    """The loaded modules must have the shapes and parts the ModelSpec describes."""
    talker = model.talker
    got = (talker.config.hidden_size, talker.code_predictor.config.hidden_size)
    want = (spec.talker_hidden_size, spec.code_predictor_hidden_size)
    if got != want:
        raise ValueError(f"{spec.model_dir}: loaded talker / code predictor hidden {got}, spec says {want}")
    proj = talker.code_predictor.small_to_mtp_projection
    if (proj is not None) != (got[0] != got[1]):
        raise ValueError(f"{spec.model_dir}: small_to_mtp_projection must exist exactly when the hidden sizes differ")
    enc = getattr(model, "speaker_encoder", None)
    if spec.variant == "base":
        if enc is None:
            raise ValueError(f"{spec.model_dir}: Base checkpoint without a speaker encoder")
        dim = model.config.speaker_encoder_config.enc_dim
        if dim != spec.talker_hidden_size:
            raise ValueError(f"{spec.model_dir}: speaker encoder dim {dim} != talker hidden {spec.talker_hidden_size}")


def _quantize(model, quant: str):
    import mlx.nn as nn
    if quant == "q8":
        spec = {"talker": 8, "cp": 8, "text": None}
    elif quant == "mix":
        spec = {"talker": 6, "cp": 8, "text": 4}
    else:
        raise ValueError(f"unknown quant {quant!r}: q8 | mix")

    def pred(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        if "speech_tokenizer" in path or "speaker_encoder" in path or "codec_embedding" in path:
            return False
        if "text_embedding" in path:
            return {"bits": spec["text"], "group_size": 64} if spec["text"] else False
        if "code_predictor" in path:
            return {"bits": spec["cp"], "group_size": 64}
        if path.startswith("talker"):
            return {"bits": spec["talker"], "group_size": 64}
        return False
    nn.quantize(model, class_predicate=pred)
