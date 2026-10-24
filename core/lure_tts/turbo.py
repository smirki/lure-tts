"""Lure Turbo: the grouped code-predictor head (3 group passes + in-group sequential add-on per frame).

A Turbo model folder is the Lure HF layout (with the Turbo talker weights) plus `turbo.json` and `turbo_head.safetensors`.
Backends call `resolve(model_dir, turbo=..., turbo_head=...)` at load time:

    turbo=None / "auto"  -> Turbo iff model_dir/turbo.json exists (default)
    turbo=True / "1"     -> Turbo required (error if turbo.json is missing)
    turbo=False / "0"    -> original 15-pass code predictor even in a Turbo folder (A/B only: the Turbo talker expects
                            the head, so this pairing is unsupported; a warning is logged)
    turbo_head=PATH      -> another head file in safetensors format; groups etc. still come from turbo.json

With no turbo.json and turbo not forced, everything is the plain Lure path (resolve returns None).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

TURBO_JSON = "turbo.json"
DEFAULT_GROUPS = ((1, 3), (4, 7), (8, 15))


@dataclass(frozen=True)
class TurboConfig:
    model_dir: str
    head_path: str
    groups: tuple = DEFAULT_GROUPS
    hidden_size: int = 1024
    num_layers: int = 5
    addon_hidden: int = 512
    layernorm_eps: float = 1e-5
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def n_passes(self) -> int:
        return len(self.groups)

    def validate(self) -> "TurboConfig":
        pos = [k for first, last in self.groups for k in range(first, last + 1)]
        if pos != list(range(1, 16)):
            raise ValueError(f"turbo groups must tile codebooks 1..15 in order, got {self.groups}")
        if not os.path.isfile(self.head_path):
            raise FileNotFoundError(f"turbo head not found: {self.head_path}")
        return self


def parse_flag(v) -> Optional[bool]:
    """None/'auto'/'' -> None (auto); truthy -> True; falsy -> False."""
    if v is None or isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("", "auto"):
        return None
    if s in ("1", "true", "yes", "on", "turbo"):
        return True
    if s in ("0", "false", "no", "off", "none"):
        return False
    raise ValueError(f"turbo must be auto/1/0, got {v!r}")


def load_config(model_dir: str, turbo_head: Optional[str] = None) -> TurboConfig:
    p = os.path.join(model_dir, TURBO_JSON)
    with open(p) as f:
        raw = json.load(f)
    if raw.get("format") != "lure-turbo-v1":
        raise ValueError(f"{p}: unknown format {raw.get('format')!r}")
    s = raw.get("sampling") or {}
    a = raw.get("addon") or {}
    head = turbo_head or os.path.join(model_dir, raw.get("head_file", "turbo_head.safetensors"))
    return TurboConfig(model_dir=model_dir, head_path=head, groups=tuple(tuple(int(x) for x in g) for g in raw["groups"]),
                       hidden_size=int(raw.get("hidden_size", 1024)), num_layers=int(raw.get("num_layers", 5)),
                       addon_hidden=int(a.get("hidden", 512)), layernorm_eps=float(a.get("layernorm_eps", 1e-5)),
                       temperature=float(s.get("temperature", 0.9)), top_k=int(s.get("top_k", 50)), top_p=float(s.get("top_p", 1.0)),
                       raw=raw).validate()


def resolve(model_dir: str, turbo=None, turbo_head: Optional[str] = None) -> Optional[TurboConfig]:
    flag = parse_flag(turbo)
    present = os.path.isfile(os.path.join(model_dir, TURBO_JSON))
    if flag is False:
        if present:
            log.warning("turbo=0 on a Turbo model folder: running the original 15-pass code predictor with the Turbo talker "
                        "(unsupported pairing, A/B only)")
        return None
    if not present:
        if flag:
            raise FileNotFoundError(f"turbo=1 but {os.path.join(model_dir, TURBO_JSON)} does not exist")
        return None
    return load_config(model_dir, turbo_head)


def is_turbo_dir(model_dir: str) -> bool:
    return os.path.isfile(os.path.join(model_dir, TURBO_JSON))
