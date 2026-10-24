"""Gold reference "backend": the official qwen-tts generate(), wrapped in the core Backend interface so it runs through the
same runner, segmenter, seeds and sampling as every real backend. It follows the request's mode (icl, xvec, custom_voice,
voice_design) and layout (non_streaming_mode = layout == "non_streaming").

    python conformance/runner.py --backend ref_backend:QwenTTSRef --model /path/to/lure --out runs/ref

Needs a CUDA GPU and qwen-tts 0.1.1 (requirements-ref.txt). It defines the target distribution for WER / ECAPA / tag-read.
It is not a production backend: no streaming (each segment is generated whole, then decoded with the reference prefix and cut, exactly like
Qwen3TTSModel.generate_voice_clone).
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np

from lure_tts.backends.base import Chunk, GenerateRequest, SegmentInfo


class QwenTTSRef:
    name = "qwen-tts-ref"
    sample_rate = 24000

    def __init__(self, tts, device):
        self.tts, self.device = tts, device

    @classmethod
    def load(cls, model_dir: str, quant: Optional[str] = None, device: str = "cuda:0", dtype: str = "bfloat16", **_):
        import torch
        from qwen_tts import Qwen3TTSModel
        tts = Qwen3TTSModel.from_pretrained(model_dir, device_map=device, dtype=getattr(torch, dtype), attn_implementation="sdpa")
        tts.model.eval()
        return cls(tts, device)

    def encode_voice(self, ref_wav: np.ndarray, ref_text: str):
        it = self.tts.create_voice_clone_prompt(ref_audio=(np.asarray(ref_wav, np.float32), 24000), ref_text=ref_text)[0]
        return (it.ref_code.detach().cpu().numpy().astype(np.int32),
                it.ref_spk_embedding.detach().float().cpu().numpy().reshape(-1))

    def generate(self, req: GenerateRequest) -> Iterator[Chunk]:
        import torch
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
        t0 = time.time()
        tts, v, s = self.tts, req.voice, req.sampling
        icl = req.mode == "icl"
        vcp, speakers = None, None
        if req.mode in ("icl", "xvec"):
            item = VoiceClonePromptItem(
                ref_code=torch.from_numpy(v.ref_codes.astype(np.int64)).to(self.device) if icl else None,
                ref_spk_embedding=torch.from_numpy(v.spk_emb).to(self.device, tts.model.dtype),
                x_vector_only_mode=not icl, icl_mode=icl, ref_text=v.ref_text if icl else None)
            vcp = tts._prompt_items_to_voice_clone_prompt([item])
        elif req.mode == "custom_voice":
            speakers = [req.speaker]
        # ids straight from the core's PromptIds (identical to qwen-tts's own tokenization, see reference/README.md)
        p = req.prompt
        ids = [torch.tensor([p.input_ids], device=self.device)]
        ref_ids = [torch.tensor([p.ref_ids], device=self.device) if icl else None]
        ins = [torch.tensor([p.instruct_ids], device=self.device) if p.instruct_ids else None]
        gk = tts._merge_generate_kwargs(do_sample=s.do_sample, top_k=s.top_k, top_p=s.top_p, temperature=s.temperature,
                                        repetition_penalty=s.repetition_penalty, subtalker_dosample=s.subtalker_do_sample,
                                        subtalker_top_k=s.subtalker_top_k, subtalker_top_p=s.subtalker_top_p,
                                        subtalker_temperature=s.subtalker_temperature, max_new_tokens=req.max_new_tokens)
        if req.seed is not None:
            torch.manual_seed(int(req.seed))
        with torch.no_grad():
            codes, _ = tts.model.generate(input_ids=ids, instruct_ids=ins, ref_ids=ref_ids, voice_clone_prompt=vcp,
                                          languages=[req.language.capitalize() if req.language != "auto" else "Auto"],
                                          speakers=speakers, non_streaming_mode=req.layout == "non_streaming", **gk)
            c = codes[0]
            full = torch.cat([vcp["ref_code"][0].to(c.device), c], dim=0) if icl else c
            wavs, _ = tts.model.speech_tokenizer.decode([{"audio_codes": full}])
        wav = np.asarray(wavs[0], np.float32)
        cut = int(v.ref_codes.shape[0] / max(full.shape[0], 1) * wav.shape[0]) if icl else 0
        pcm = wav[cut:]
        n = int(c.shape[0])
        wall = time.time() - t0
        yield Chunk(pcm, SegmentInfo(frames=n, ended_by_eos=n < req.max_new_tokens, samples=len(pcm), ttfa_s=wall, wall_s=wall))

    def close(self):
        import torch
        self.tts = None
        torch.cuda.empty_cache()
