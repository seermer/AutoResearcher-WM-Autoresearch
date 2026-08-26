"""The evaluation case sets. Fixed and kernel-owned; agents cannot change them."""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from .config import EVAL, PATHS

NAVI_ACTIONS = {"W", "A", "S", "D", "left", "right", "up", "down",
                "forward", "backward", "cam_left", "cam_right", "cam_up", "cam_down"}


def _is_navi(case: dict) -> bool:
    return any(t.get("action") in NAVI_ACTIONS or t.get("type") == "navigation"
               for t in case.get("interactions", []))


@lru_cache(maxsize=1)
def all_cases() -> tuple[dict, ...]:
    out = []
    for f in sorted((PATHS.wbench / "data" / "cases").glob("case_*.json"),
                    key=lambda p: int(p.stem.split("_")[1])):
        out.append(json.loads(f.read_text()))
    return tuple(out)


@lru_cache(maxsize=1)
def navi_cases() -> tuple[dict, ...]:
    return tuple(c for c in all_cases() if _is_navi(c))


def _stratum(case: dict) -> tuple[str, str]:
    s = case["settings"]
    return s["scene"]["category"], s["perspective"]


@lru_cache(maxsize=8)
def proxy_cases(n: int = 0, seed: int = 0) -> tuple[dict, ...]:
    """A fixed, stratified subset of the navigation split used as the cheap rung.

    Stratified over (scene category x perspective) so the subset keeps the
    benchmark's coverage; deterministic given (n, seed).
    """
    n = n or EVAL.proxy_cases
    seed = seed or EVAL.proxy_seed
    pool = navi_cases()
    buckets: dict[tuple[str, str], list[dict]] = {}
    for c in pool:
        buckets.setdefault(_stratum(c), []).append(c)
    rng = random.Random(seed)
    for v in buckets.values():
        rng.shuffle(v)
    order = sorted(buckets, key=lambda k: (-len(buckets[k]), k))
    picked: list[dict] = []
    while len(picked) < n and any(buckets[k] for k in order):
        for k in order:
            if buckets[k] and len(picked) < n:
                picked.append(buckets[k].pop())
    return tuple(sorted(picked, key=lambda c: int(c["id"])))


def case_ids(cases) -> list[str]:
    return [str(c["id"]) for c in cases]
