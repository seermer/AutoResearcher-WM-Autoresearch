"""The outer loop: select -> self-modify -> evaluate -> backpropagate -> insert."""
from __future__ import annotations

import dataclasses
import json
import os
import random
import shutil
import subprocess
import traceback
import sys
import time
from pathlib import Path

from . import cmp as cmp_mod
from . import datastore, diskguard, procs, weights
from . import selection, vcs
from .archive import OK, PENDING, TRASH, Archive, Node
from .config import BUDGET, PATHS
from .contract import verify
from .evaluate import Evaluator
from .security import SecurityError, assert_disk_headroom, free_disk_gb
from .trace import Tracer

HOST = PATHS.repo / "kernel" / "runners" / "agent_host.py"
MAX_CONSECUTIVE_TRASH = int(os.environ.get("AR_MAX_CONSECUTIVE_TRASH", 12))
TRASH_BACKOFF_CAP = 900


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
        node.checkpoint_path = None
        self.archive.save(node)
        shutil.rmtree(node.dir / "work", ignore_errors=True)
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
        node.shards = [datastore.register_base()]
        vcs.link_node_data(ws.sana, node.shards)
        self.archive.save(node)
        self.tracer.node_id = node.id
        with self.tracer.span("root.evaluate"):
            res = self.evaluator.run(node.id, ws.sana, node.dir, ckpt=None)
        if not res.ok:
            node.status, node.failure = TRASH, res.failure
            self.archive.save(node)
            raise RuntimeError(f"baseline evaluation failed: {res.failure}")
        node.status = OK
        node.score, node.metrics = res.score, res.summary()
        node.recipe = {"plan": "baseline: released SANA-WM stage-1 teacher, untrained"}
        node.evaluated_at = time.time()
        self.archive.save(node)
        cmp_mod.backpropagate(self.archive, node)
        self.tracer.emit("root.ready", score=node.score)
        return node

    # ---------- recovery ----------
    def recover(self) -> list[str]:
        """Clean up after a crashed run before doing anything else.

        A node left PENDING means the process died mid-iteration: its worktrees may
        still exist, its branches are half-built, and its work dir may hold a partial
        10 GB checkpoint. Trash them and prune both repos so the next step starts clean.
        """
        stale = [n for n in self.archive.nodes if n.status == PENDING]
        for node in stale:
            ws = vcs.NodeWorkspace(node.id, node.parent)
            self._trash(node, ws, "abandoned by a previous run")
        for repo in (PATHS.repo, PATHS.sana):
            vcs.git(repo, "worktree", "prune", check=False)
        # Worktree directories with no live node behind them.
        live = {n.id for n in self.archive.nodes if n.status != TRASH}
        if PATHS.worktrees.exists():
            for d in PATHS.worktrees.iterdir():
                if d.is_dir() and d.name not in live:
                    vcs.NodeWorkspace(d.name, None).destroy()
        if stale:
            self.tracer.emit("loop.recovered", nodes=[n.id for n in stale])
        return [n.id for n in stale]

    # ---------- one iteration ----------
    def step(self) -> Node | None:
        """One iteration. Never raises for a node-level failure; the node is trashed."""
        try:
            return self._step()
        except SecurityError:
            raise
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # noqa: BLE001 - the outer loop must survive anything
            self.tracer.emit("step.crashed", error=f"{type(e).__name__}: {e}",
                             trace=traceback.format_exc()[-4000:])
            self.archive.reload()
            for node in [n for n in self.archive.nodes if n.status == PENDING]:
                self._trash(node, vcs.NodeWorkspace(node.id, node.parent),
                            f"kernel crash: {type(e).__name__}: {e}")
            return None

    def _step(self) -> Node | None:
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
        node.shards = list(parent.shards)
        vcs.link_node_data(ws.sana, node.shards)
        self.archive.save(node)
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
            logs_dir=logs, parent_logs_dir=(parent.dir / "logs" if (parent.dir / "logs").is_dir() else None),
            eval_report_path=parent_report if parent_report.exists() else None,
            parent_score=parent.score, parent_metrics=parent.metrics,
            budget_seconds=BUDGET.edit_self_seconds,
        )
        # Runs from the frozen parent snapshot, writes into the child's draft. An agent
        # rewriting its own package therefore cannot pull a half-written module or a
        # truncated prompt into the interpreter that is executing it.
        with self.tracer.span("edit_self"):
            res = self._run_agent("edit_self", ws.agents_frozen, ctx,
                                  writable=[ws.agents / "agents"],
                                  readable=[PATHS.nodes, PATHS.sana, PATHS.wbench,
                                            ws.agents_frozen, ws.agents],
                                  log_dir=logs, timeout=BUDGET.edit_self_seconds)
        ws.release_frozen()
        if not res.get("ok", True):
            self._trash(node, ws, f"edit_self: {res.get('error')}")
            return None

        sha = vcs.ensure_committed(ws.agents, f"[{node.id}] edit_self: {res.get('summary','')[:120]}")
        if sha is None:
            self._trash(node, ws, "edit_self produced no diff")
            return None
        stat = vcs.diffstat(ws.agents)
        contract = verify(ws.agents, stat["files"])
        if not contract:
            self._trash(node, ws, f"contract: {contract.reason}")
            return None

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
            base_checkpoint=weights.ensure_stage1()["dit"],
            wbench_dir=PATHS.wbench, budget_seconds=BUDGET.improve_recipe_seconds,
            deadline_ts=time.time() + BUDGET.improve_recipe_seconds,
            disk_gb=min(BUDGET.node_disk_gb, free_disk_gb() - BUDGET.min_free_disk_gb),
            gpus=BUDGET.gpus,
        )
        # Training saves ~30 GB of sharded state per checkpoint that nothing prunes;
        # without this a long run fills the disk and loses the node hours in.
        with self.tracer.span("improve_recipe"), diskguard.DiskGuard(ws.sana, self.tracer):
            res = self._run_agent("improve_recipe", ws.agents, ctx,
                                  writable=[ws.sana, PATHS.datastore, out_dir, ctx.memory_dir],
                                  readable=[PATHS.nodes, PATHS.wbench, ws.agents],
                                  log_dir=logs, timeout=BUDGET.improve_recipe_seconds)
        # Lessons and sources written during the recipe phase land in the agent
        # worktree after edit_self already committed. Without this they would be
        # dropped by `worktree remove --force` and never reach any descendant.
        vcs.ensure_committed(ws.agents, f"[{node.id}] memory from improve_recipe")
        salvaged = None
        if not res.get("ok", True) or not res.get("checkpoint_path"):
            # The agent is gone, but anything it launched with nohup is detached and
            # still holding every GPU. Give training that is genuinely still working a
            # bounded extension, then reap whatever is left either way.
            log = procs.newest_log(ws.sana)
            if log and procs.progressing(log) and procs.pids_under(ws.sana):
                self.tracer.emit("improve_recipe.grace",
                                 seconds=BUDGET.train_grace_seconds, log=str(log))
                drained = procs.wait_for_exit(ws.sana, BUDGET.train_grace_seconds)
                self.tracer.emit("improve_recipe.grace_end", drained=drained)
            if killed := procs.reap(ws.sana):
                self.tracer.emit("improve_recipe.reaped", pids=killed)
            # Training may well have finished; only the report was late. Look for the
            # weights before throwing away hours of GPU.
            salvaged = diskguard.newest_merged(ws.sana)
            if salvaged is None:
                self._trash(node, ws, f"improve_recipe: {res.get('error')}")
                return None
            self.tracer.emit("improve_recipe.salvaged", path=str(salvaged),
                             agent_error=str(res.get("error"))[:300])
            res = {**res, "checkpoint_path": str(salvaged)}

        vcs.ensure_committed(ws.sana, f"[{node.id}] recipe: {res.get('summary','')[:120]}")
        try:
            new_shards = datastore.seal_staging(
                ws.sana / "data" / "staging", node.id,
                {"recipe": (res.get("recipe") or {}).get("plan", "")[:200]})
        except Exception as e:  # noqa: BLE001 - a bad shard must not kill the run
            self.tracer.emit("datastore.seal_failed", error=f"{type(e).__name__}: {e}")
            new_shards = []
        if new_shards:
            node.shards = list(node.shards) + new_shards
            self.tracer.emit("datastore.sealed", shards=new_shards)
        ckpt = self._retain(node, Path(res["checkpoint_path"]))
        if ckpt is None:
            self._trash(node, ws, f"checkpoint missing: {res['checkpoint_path']}")
            return None
        node.recipe, node.train = res.get("recipe", {}), res.get("train", {})
        if salvaged is not None:
            node.train = {**node.train, "salvaged": True,
                          "agent_error": str(res.get("error"))[:300]}
        self.archive.save(node)

        with self.tracer.span("evaluate"):
            ev = self.evaluator.run(node.id, ws.sana, node.dir, ckpt=ckpt)
        if not ev.ok:
            self._trash(node, ws, f"evaluate: {ev.failure}")
            return None

        node.status, node.score, node.metrics = OK, ev.score, ev.summary()
        procs.reap(ws.sana)              # nothing the agent launched outlives its node
        node.evaluated_at = time.time()
        self.archive.save(node)
        return node

    def _retain(self, node: Node, produced: Path) -> Path | None:
        """Keep exactly one trained checkpoint: this node's. Nodes always train from
        base, so every older checkpoint is dead weight on a disk that has none to spare."""
        produced = produced.resolve()
        if not produced.is_file():
            return None
        PATHS.current.mkdir(parents=True, exist_ok=True)
        dest = PATHS.current / f"{node.id}.pth"
        shutil.move(str(produced), dest)
        for old in PATHS.current.iterdir():
            if old != dest:
                old.unlink(missing_ok=True) if old.is_file() else shutil.rmtree(old, ignore_errors=True)
        for other in self.archive.nodes:
            if other.id != node.id and other.checkpoint_path:
                other.checkpoint_path, other.checkpoint_evicted = None, True
                self.archive.save(other)
        node.checkpoint_path = str(dest)
        self.archive.save(node)
        # The sharded FSDP state is only useful for resuming, and we never resume.
        shutil.rmtree(produced.with_suffix(""), ignore_errors=True)
        self.tracer.emit("checkpoint.retained", path=str(dest),
                         gb=round(dest.stat().st_size / 2**30, 2))
        return dest

    # ---------- driver ----------
    def run(self, max_nodes: int | None = None) -> None:
        self.recover()
        self.bootstrap()
        budget = max_nodes or BUDGET.max_nodes
        done = crashes = consecutive_trash = 0
        while done < budget:
            try:
                node = self.step()
            except SecurityError as e:
                self.tracer.emit("loop.halt", reason=str(e))
                print(f"halting: {e}")
                return
            if node is None:
                if not selection.expandable(self.archive):
                    self.tracer.emit("loop.halt", reason="no expandable node")
                    return
                crashes += 1
                if crashes >= 5:
                    self.tracer.emit("loop.halt", reason="5 consecutive kernel crashes")
                    print("halting: 5 consecutive kernel crashes; see the trace")
                    return
                continue
            crashes = 0
            done += 1
            # A dead LLM endpoint or a full GPU trashes every node without ever
            # crashing the loop. Back off so a transient outage heals itself, and
            # stop only once it is clearly not transient.
            consecutive_trash = consecutive_trash + 1 if node.status == TRASH else 0
            if consecutive_trash:
                if consecutive_trash >= MAX_CONSECUTIVE_TRASH:
                    self.tracer.emit("loop.halt", reason=f"{consecutive_trash} nodes trashed in a row",
                                     last_failure=node.failure)
                    print(f"halting: {consecutive_trash} nodes trashed in a row; last: {node.failure}")
                    return
                backoff = min(TRASH_BACKOFF_CAP, 30 * 2 ** (consecutive_trash - 1))
                self.tracer.emit("loop.backoff", seconds=backoff,
                                 consecutive=consecutive_trash, failure=node.failure)
                time.sleep(backoff)
            best = self.archive.best()
            print(f"[{done}/{budget}] {node.id} status={node.status} score={node.score} "
                  f"best={best.id if best else '-'}@{best.score if best else '-'}")
