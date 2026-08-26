"""The outer loop: select -> self-modify -> evaluate -> backpropagate -> insert."""
from __future__ import annotations

import dataclasses
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import cmp as cmp_mod
from . import selection, vcs
from .archive import OK, PENDING, TRASH, Archive, Node
from .config import BUDGET, PATHS
from .contract import verify
from .evaluate import Evaluator
from .security import SecurityError, assert_disk_headroom, free_disk_gb
from .trace import Tracer

HOST = PATHS.repo / "kernel" / "runners" / "agent_host.py"


class Loop:
    def __init__(self, seed: int | None = None):
        self.archive = Archive()
        self.tracer = Tracer()
        self.evaluator = Evaluator(self.tracer)
        self.rng = random.Random(seed)

    # ---------- helpers ----------
    def _run_agent(self, phase: str, worktree: Path, ctx, writable, readable,
                   log_dir: Path, timeout: int) -> dict:
        log_dir.mkdir(parents=True, exist_ok=True)
        ctx_file, out_file = log_dir / f"{phase}_ctx.json", log_dir / f"{phase}_result.json"
        ctx_file.write_text(json.dumps({
            "ctx": {k: (str(v) if isinstance(v, Path) else v)
                    for k, v in dataclasses.asdict(ctx).items()},
            "writable": [str(p) for p in writable],
            "readable": [str(p) for p in readable],
            "log_dir": str(log_dir),
            "shell_timeout": min(timeout, BUDGET.improve_recipe_seconds),
        }, default=str))
        cmd = [sys.executable, str(HOST), "--worktree", str(worktree),
               "--kernel_root", str(PATHS.repo), "--phase", phase,
               "--ctx", str(ctx_file), "--out", str(out_file)]
        with (log_dir / f"{phase}.log").open("ab") as fh:
            try:
                subprocess.run(cmd, cwd=str(worktree), stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": f"{phase} exceeded {timeout}s"}
        if not out_file.exists():
            return {"ok": False, "error": f"{phase} produced no result"}
        return json.loads(out_file.read_text())

    def _trash(self, node: Node, ws: vcs.NodeWorkspace, reason: str) -> Node:
        node.status, node.failure = TRASH, reason
        self.archive.save(node)
        ws.trash()
        self.tracer.emit("node.trashed", node=node.id, reason=reason)
        cmp_mod.backpropagate(self.archive, node)
        return node

    # ---------- root ----------
    def bootstrap(self) -> Node:
        root = self.archive.root_node()
        if root is not None:
            return root
        node = Node(id="n0000", parent=None, depth=0, status=PENDING,
                    agent_branch=vcs.NODE_BRANCH.format(nid="n0000"),
                    sana_branch=vcs.NODE_BRANCH.format(nid="n0000"))
        self.archive.add(node)
        ws = vcs.NodeWorkspace(node.id, None).create()
        self.tracer.node_id = node.id
        with self.tracer.span("root.evaluate"):
            res = self.evaluator.run(node.id, ws.sana, node.dir, lora=None)
        if not res.ok:
            node.status, node.failure = TRASH, res.failure
            self.archive.save(node)
            raise RuntimeError(f"baseline evaluation failed: {res.failure}")
        node.status = OK
        node.score, node.metrics = res.score, res.summary()
        node.recipe = {"plan": "baseline: released SANA-WM stage-1 teacher, no LoRA"}
        node.evaluated_at = time.time()
        self.archive.save(node)
        cmp_mod.backpropagate(self.archive, node)
        self.tracer.emit("root.ready", score=node.score)
        return node

    # ---------- one iteration ----------
    def step(self) -> Node | None:
        assert_disk_headroom()
        parent = selection.select(self.archive, self.rng)
        if parent is None:
            return None
        nid = self.archive.new_id(parent)
        self.tracer = Tracer(nid)
        self.evaluator.tracer = self.tracer
        self.tracer.emit("select", parent=parent.id, parent_cmp=round(parent.cmp, 4),
                         parent_score=parent.score, clade_n=parent.clade_n,
                         free_disk_gb=round(free_disk_gb(), 1))

        node = Node(id=nid, parent=parent.id, depth=parent.depth + 1, status=PENDING,
                    agent_branch=vcs.NODE_BRANCH.format(nid=nid),
                    sana_branch=vcs.NODE_BRANCH.format(nid=nid))
        self.archive.add(node)
        ws = vcs.NodeWorkspace(nid, parent.id).create()
        logs = node.dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        history = self.archive.history_jsonl(parent.id, node.dir / "history.jsonl")

        # 2. SELF-MODIFY
        edited = self._self_modify(node, parent, ws, logs, history)
        if edited is None:
            return node

        # 3. EVALUATE (improve_recipe -> train -> WBench)
        scored = self._improve_and_score(node, parent, ws, logs, history)
        if scored is None:
            return node

        # 4. BACKPROPAGATE
        cmp_mod.backpropagate(self.archive, node)
        self.tracer.emit("node.done", score=node.score, parent_score=parent.score,
                         delta=None if parent.score is None else round(node.score - parent.score, 3))
        ws.destroy()
        return node

    def _self_modify(self, node: Node, parent: Node, ws, logs: Path, history: Path):
        from .api import EditSelfContext
        parent_report = parent.dir / "wbench_proxy" / parent.id / "evaluation" / "report.json"
        ctx = EditSelfContext(
            node_id=node.id, parent_node_id=parent.id, agents_dir=ws.agents,
            history_path=history, memory_dir=ws.agents / "agents" / "memory",
            logs_dir=logs, eval_report_path=parent_report if parent_report.exists() else None,
            parent_score=parent.score, parent_metrics=parent.metrics,
            budget_seconds=BUDGET.edit_self_seconds,
        )
        with self.tracer.span("edit_self"):
            res = self._run_agent("edit_self", ws.agents, ctx,
                                  writable=[ws.agents / "agents"],
                                  readable=[PATHS.nodes, PATHS.sana, PATHS.wbench],
                                  log_dir=logs, timeout=BUDGET.edit_self_seconds)
        if not res.get("ok", True):
            return self._trash(node, ws, f"edit_self: {res.get('error')}") and None

        sha = vcs.ensure_committed(ws.agents, f"[{node.id}] edit_self: {res.get('summary','')[:120]}")
        if sha is None:
            return self._trash(node, ws, "edit_self produced no diff") and None
        stat = vcs.diffstat(ws.agents)
        contract = verify(ws.agents, stat["files"])
        if not contract:
            return self._trash(node, ws, f"contract: {contract.reason}") and None

        node.self_edit = {"summary": res.get("summary", ""), "hypothesis": res.get("hypothesis", ""),
                          "sha": sha, **{k: stat[k] for k in ("files", "insertions", "deletions")}}
        self.archive.save(node)
        self.tracer.emit("edit_self.accepted", **{k: stat[k] for k in ("insertions", "deletions")},
                         files=len(stat["files"]))
        return node

    def _improve_and_score(self, node: Node, parent: Node, ws, logs: Path, history: Path):
        from .api import ImproveRecipeContext
        out_dir = node.dir / "work"
        out_dir.mkdir(parents=True, exist_ok=True)
        parent_report = parent.dir / "wbench_proxy" / parent.id / "evaluation" / "report.json"
        if parent_report.exists():
            shutil.copy(parent_report, out_dir / "parent_report.json")

        ctx = ImproveRecipeContext(
            node_id=node.id, agents_dir=ws.agents, sana_dir=ws.sana,
            datastore_dir=PATHS.datastore, out_dir=out_dir, history_path=history,
            memory_dir=ws.agents / "agents" / "memory", logs_dir=logs,
            baseline_lora=Path(parent.lora_path) if parent.lora_path else None,
            wbench_dir=PATHS.wbench, budget_seconds=BUDGET.improve_recipe_seconds,
            disk_gb=min(BUDGET.node_disk_gb, free_disk_gb() - BUDGET.min_free_disk_gb),
            gpus=BUDGET.gpus,
        )
        with self.tracer.span("improve_recipe"):
            res = self._run_agent("improve_recipe", ws.agents, ctx,
                                  writable=[ws.sana, PATHS.datastore, out_dir],
                                  readable=[PATHS.nodes, PATHS.wbench],
                                  log_dir=logs, timeout=BUDGET.improve_recipe_seconds)
        if not res.get("ok", True) or not res.get("lora_path"):
            return self._trash(node, ws, f"improve_recipe: {res.get('error')}") and None

        vcs.ensure_committed(ws.sana, f"[{node.id}] recipe: {res.get('summary','')[:120]}")
        lora = Path(res["lora_path"]).resolve()
        keep = node.dir / "lora"
        if keep.exists():
            shutil.rmtree(keep)
        shutil.copytree(lora, keep)
        node.lora_path, node.recipe, node.train = str(keep), res.get("recipe", {}), res.get("train", {})
        self.archive.save(node)

        with self.tracer.span("evaluate"):
            ev = self.evaluator.run(node.id, ws.sana, node.dir, lora=keep)
        if not ev.ok:
            return self._trash(node, ws, f"evaluate: {ev.failure}") and None

        node.status, node.score, node.metrics = OK, ev.score, ev.summary()
        node.evaluated_at = time.time()
        self.archive.save(node)
        return node

    # ---------- driver ----------
    def run(self, max_nodes: int | None = None) -> None:
        self.bootstrap()
        budget = max_nodes or BUDGET.max_nodes
        done = 0
        while done < budget:
            try:
                node = self.step()
            except SecurityError as e:
                self.tracer.emit("loop.halt", reason=str(e))
                print(f"halting: {e}")
                return
            if node is None:
                self.tracer.emit("loop.halt", reason="no expandable node")
                return
            done += 1
            best = self.archive.best()
            print(f"[{done}/{budget}] {node.id} status={node.status} score={node.score} "
                  f"best={best.id if best else '-'}@{best.score if best else '-'}")
