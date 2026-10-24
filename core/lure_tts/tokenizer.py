"""Pure-Python Qwen2 byte-level BPE tokenizer (vocab.json + merges.txt + tokenizer_config.json added tokens).

Matches transformers' Qwen2Tokenizer / Qwen2TokenizerFast for the texts the prompt spec uses: added tokens (<|im_start|>,
<|im_end|>, <tts_pad>, ...) are matched first and become single ids (never BPE-split), the remaining text is NFC-normalised,
pre-tokenised with the Qwen2 regex and byte-level BPE encoded. Only dependency: the `regex` module (for \\p{L} / \\p{N}).
Checked token-for-token against qwen-tts's own processor in conformance/reference (tests/test_prompt_reference.py).
"""
from __future__ import annotations

import json
import os
import unicodedata
from functools import lru_cache

import regex

# Qwen2Tokenizer PRETOKENIZE_REGEX (transformers/models/qwen2/tokenization_qwen2.py)
PRETOKENIZE_REGEX = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class QwenTokenizer:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]], added: dict[str, int]):
        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        self.added = dict(added)
        for k, v in self.added.items():
            self.decoder[v] = k
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = regex.compile(PRETOKENIZE_REGEX)
        # longest-first alternation so "<tts_text_bos_single>" wins over "<tts_text_bos>"
        alts = sorted(self.added, key=len, reverse=True)
        self.added_pat = regex.compile("(" + "|".join(regex.escape(a) for a in alts) + ")") if alts else None
        self._cache: dict[str, list[str]] = {}

    @classmethod
    def from_dir(cls, model_dir: str) -> "QwenTokenizer":
        with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        merges = []
        with open(os.path.join(model_dir, "merges.txt"), encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if (i == 0 and line.startswith("#version")) or not line:
                    continue
                a, b = line.split(" ")
                merges.append((a, b))
        added = {}
        tc = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(tc):
            with open(tc, encoding="utf-8") as f:
                cfg = json.load(f)
            for k, v in cfg.get("added_tokens_decoder", {}).items():
                added[v["content"]] = int(k)
        return cls(vocab, merges, added)

    def _bpe(self, token: str) -> list[str]:
        if token in self._cache:
            return self._cache[token]
        word = list(token)
        while len(word) > 1:
            rank, best = min((self.bpe_ranks.get((word[i], word[i + 1]), 1 << 60), i) for i in range(len(word) - 1))
            if rank == 1 << 60:
                break
            # merge every occurrence of the best-ranked pair, left to right
            a, b = word[best], word[best + 1]
            new, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = new
        if len(self._cache) < 50000:
            self._cache[token] = word
        return word

    def _encode_plain(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        ids = []
        for piece in self.pat.findall(text):
            mapped = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for t in self._bpe(mapped):
                ids.append(self.encoder[t])
        return ids

    def encode(self, text: str) -> list[int]:
        """No BOS/EOS is ever added (Qwen2 add_bos_token=False)."""
        if not self.added_pat:
            return self._encode_plain(text)
        ids: list[int] = []
        for i, part in enumerate(self.added_pat.split(text)):
            if not part:
                continue
            if i % 2 == 1:
                ids.append(self.added[part])
            else:
                ids.extend(self._encode_plain(part))
        return ids

    def decode(self, ids) -> str:
        out, buf = [], []
        for i in ids:
            tok = self.decoder.get(int(i), "")
            if tok in self.added:
                if buf:
                    out.append(bytearray(self.byte_decoder[c] for c in "".join(buf)).decode("utf-8", errors="replace"))
                    buf = []
                out.append(tok)
            else:
                buf.append(tok)
        if buf:
            out.append(bytearray(self.byte_decoder[c] for c in "".join(buf)).decode("utf-8", errors="replace"))
        return "".join(out)

    def id_of(self, token: str) -> int:
        return self.added[token] if token in self.added else self.encoder[token]
