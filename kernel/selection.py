"""Expansion policy: Thompson sampling over CMP posteriors."""
from __future__ import annotations

import random

from .archive import OK, Archive, Node
from .config import SELECTION


def expandable(archive: Archive) -> list[Node]:
    """Only evaluated, healthy nodes can be expanded."""
    return [n for n in archive.alive() if n.score is not None]


def posterior(node: Node) -> tuple[float, float]:
    """Beta posterior over the clade's normalized performance."""
    a = SELECTION.prior_alpha + node.clade_wx
    b = SELECTION.prior_beta + max(0.0, node.clade_w - node.clade_wx)
    return a, b


def sample_scores(archive: Archive, rng: random.Random | None = None) -> dict[str, float]:
    rng = rng or random
    out = {}
    for n in expandable(archive):
        a, b = posterior(n)
        out[n.id] = rng.betavariate(a, b)
    return out


def select(archive: Archive, rng: random.Random | None = None) -> Node | None:
    """Pick the node to expand next. Under-explored nodes keep wide posteriors."""
    draws = sample_scores(archive, rng)
    if not draws:
        return None
    return archive[max(draws, key=draws.get)]
