"""Clade-metaproductivity: MCTS-style backup of descendant performance."""
from __future__ import annotations

import math

from .archive import OK, TRASH, Archive, Node
from .config import SELECTION


def normalize(score: float | None, baseline: float | None) -> float:
    """Map a raw WBench score to [0,1], centred at 0.5 on the baseline."""
    if score is None:
        return 0.0
    if baseline is None:
        return 0.5
    return 1.0 / (1.0 + math.exp(-(score - baseline) / SELECTION.softness))


def _contribution(u: Node, root_depth: int, baseline: float | None) -> tuple[float, float]:
    """(weight, weight*x) of one clade member."""
    decay = SELECTION.clade_decay ** max(0, u.depth - root_depth)
    if u.status == TRASH:
        if not SELECTION.count_failures:
            return 0.0, 0.0
        return decay * SELECTION.failure_weight, 0.0
    if u.status != OK or u.score is None:
        return 0.0, 0.0
    return decay, decay * normalize(u.score, baseline)


def recompute(archive: Archive, node: Node, baseline: float | None) -> None:
    w = wx = 0.0
    n = 0
    for u in archive.clade(node.id):
        cw, cwx = _contribution(u, node.depth, baseline)
        w += cw
        wx += cwx
        if cw:
            n += 1
    node.clade_w, node.clade_wx, node.clade_n = w, wx, n
    archive.save(node)


def baseline_score(archive: Archive) -> float | None:
    root = archive.root_node()
    return root.score if root and root.score is not None else None


def backpropagate(archive: Archive, node: Node) -> None:
    """Push a new result up through every ancestor (MCTS backup)."""
    baseline = baseline_score(archive)
    recompute(archive, node, baseline)
    for anc in archive.ancestors(node.id):
        recompute(archive, anc, baseline)
