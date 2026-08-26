"""Tree archive of agent nodes. One directory per node, JSON metadata."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import PATHS

OK, TRASH, PENDING = "ok", "trash", "pending"


@dataclass
class Node:
    id: str
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    depth: int = 0
    status: str = PENDING
    agent_branch: str = ""
    sana_branch: str = ""
    # What this node changed, as reported by its own agents.
    self_edit: dict = field(default_factory=dict)     # summary + diffstat of edit_self
    recipe: dict = field(default_factory=dict)        # data recipe manifest
    train: dict = field(default_factory=dict)         # steps, lora config, wall-clock
    lora_path: str | None = None
    # Evaluation
    score: float | None = None                        # scalar objective (proxy average)
    metrics: dict = field(default_factory=dict)       # per-metric breakdown
    full_score: float | None = None                   # full 158-case eval, if promoted
    failure: str | None = None
    # CMP aggregates, maintained by backpropagate()
    clade_w: float = 0.0
    clade_wx: float = 0.0
    clade_n: int = 0
    created_at: float = field(default_factory=time.time)
    evaluated_at: float | None = None

    # Set by the owning Archive; not a dataclass field, so it stays out of node.json.
    _root: Path | None = None

    @property
    def dir(self) -> Path:
        return (self._root or PATHS.nodes) / self.id

    @property
    def cmp(self) -> float:
        """Clade-metaproductivity: weighted mean normalized score over the clade."""
        return self.clade_wx / self.clade_w if self.clade_w > 0 else 0.0


class Archive:
    def __init__(self, root: Path | None = None):
        self.root = root or PATHS.nodes
        self.root.mkdir(parents=True, exist_ok=True)
        self._nodes: dict[str, Node] = {}
        self.reload()

    def reload(self) -> None:
        self._nodes = {}
        for f in sorted(self.root.glob("*/node.json")):
            d = json.loads(f.read_text())
            node = Node(**{k: v for k, v in d.items() if not k.startswith("_")})
            node._root = self.root
            self._nodes[d["id"]] = node

    # ---- access ----
    def __contains__(self, nid: str) -> bool:
        return nid in self._nodes

    def __getitem__(self, nid: str) -> Node:
        return self._nodes[nid]

    def get(self, nid: str) -> Node | None:
        return self._nodes.get(nid)

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def alive(self) -> list[Node]:
        return [n for n in self._nodes.values() if n.status == OK]

    def root_node(self) -> Node | None:
        return next((n for n in self._nodes.values() if n.parent is None), None)

    def best(self) -> Node | None:
        scored = [n for n in self.alive() if n.score is not None]
        return max(scored, key=lambda n: n.score) if scored else None

    def clade(self, nid: str) -> list[Node]:
        """The node itself plus its entire descendant subtree."""
        out, stack = [], [nid]
        while stack:
            n = self._nodes.get(stack.pop())
            if n is None:
                continue
            out.append(n)
            stack.extend(n.children)
        return out

    def ancestors(self, nid: str) -> list[Node]:
        out, cur = [], self._nodes.get(nid)
        while cur is not None and cur.parent:
            cur = self._nodes.get(cur.parent)
            if cur:
                out.append(cur)
        return out

    # ---- mutation ----
    def new_id(self, parent: Node | None) -> str:
        n = len(self._nodes)
        stem = "n0000" if parent is None else f"n{n:04d}"
        while stem in self._nodes:
            n += 1
            stem = f"n{n:04d}"
        return stem

    def add(self, node: Node) -> Node:
        node._root = self.root
        self._nodes[node.id] = node
        if node.parent and node.parent in self._nodes:
            parent = self._nodes[node.parent]
            if node.id not in parent.children:
                parent.children.append(node.id)
                self.save(parent)
        self.save(node)
        return node

    def save(self, node: Node) -> None:
        node._root = node._root or self.root
        node.dir.mkdir(parents=True, exist_ok=True)
        tmp = node.dir / "node.json.tmp"
        payload = {k: v for k, v in asdict(node).items() if not k.startswith("_")}
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(node.dir / "node.json")

    def history_jsonl(self, nid: str, path: Path, limit: int = 60) -> Path:
        """Lineage + siblings + global bests, as a compact JSONL file for agents to read."""
        seen, records = set(), []
        chain = [self._nodes[nid]] + self.ancestors(nid) if nid in self._nodes else []
        for n in chain:
            seen.add(n.id)
            records.append(("lineage", n))
        for n in self.nodes:
            if n.id not in seen and n.parent in seen:
                seen.add(n.id)
                records.append(("sibling", n))
        rest = sorted((n for n in self.alive() if n.id not in seen),
                      key=lambda n: (n.score is None, -(n.score or 0)))
        for n in rest[: max(0, limit - len(records))]:
            records.append(("archive", n))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for rel, n in records:
                f.write(json.dumps({
                    "relation": rel, "id": n.id, "parent": n.parent, "depth": n.depth,
                    "status": n.status, "score": n.score, "cmp": round(n.cmp, 4),
                    "clade_n": n.clade_n, "failure": n.failure,
                    "self_edit": n.self_edit, "recipe": n.recipe, "train": n.train,
                    "metrics": n.metrics,
                }, default=str) + "\n")
        return path
