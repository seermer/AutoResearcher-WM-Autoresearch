"""Reads the archive and trace streams. Nothing here mutates state."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ..archive import Archive
from ..config import BUDGET, EVAL, PATHS

# Spans the UI shows as phases, in the order they run within a node.
PHASES = ("edit_self", "improve_recipe", "evaluate", "root.evaluate")
FATAL = ("node.trashed", "step.crashed", "loop.halt", "datastore.seal_failed",
         "llm.error", "tool.error")


def _flat_metrics(summary: dict) -> dict:
    """node.metrics is the evaluator's summary: {score, dimensions, metrics, n_cases}.
    Flatten it for the comparison table, dimensions first since those are the objective."""
    if not isinstance(summary, dict):
        return {}
    if "metrics" not in summary and "dimensions" not in summary:
        return {k: v for k, v in summary.items() if isinstance(v, (int, float))}
    out = {f"[dim] {k}": v for k, v in (summary.get("dimensions") or {}).items()}
    out.update({k: v for k, v in (summary.get("metrics") or {}).items()
                if isinstance(v, (int, float))})
    return out


def _read_jsonl(path: Path, offset: int = 0) -> tuple[list[dict], int]:
    if not path.is_file():
        return [], offset
    size = path.stat().st_size
    if offset > size:
        offset = 0                       # file rotated or truncated
    out = []
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    for line in data[:end].split(b"\n"):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out, offset + end + 1


def run_files() -> list[Path]:
    return sorted(PATHS.traces.glob("run-*.jsonl"))


def latest_run() -> Path | None:
    files = run_files()
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def events(run: str = "", offset: int = 0, tail: int = 2000, file: str = "") -> dict:
    """Incremental read of one run stream. `run='*'` concatenates every run once.

    `file` is the stream the caller last read. When the loop restarts it opens a new
    run file, and the caller's offset would land mid-way through it; say so instead.
    """
    if run == "*" and offset == 0:
        merged = []
        for f in run_files():
            merged += _read_jsonl(f)[0]
        merged.sort(key=lambda e: e.get("ts", 0))
        newest = latest_run()
        return {"run": "*", "file": newest.name if newest else "",
                "offset": newest.stat().st_size if newest else 0,
                "events": merged[-tail:], "dropped": max(0, len(merged) - tail)}
    path = (PATHS.traces / run) if run and run != "*" else latest_run()
    if path is None or not path.is_file():
        return {"run": run, "file": "", "offset": 0, "events": [], "dropped": 0}
    reset = bool(file) and file != path.name
    evs, new_off = _read_jsonl(path, 0 if reset else offset)
    dropped = max(0, len(evs) - tail)
    return {"run": run or path.name, "file": path.name, "offset": new_off, "reset": reset,
            "events": evs[-tail:], "dropped": dropped}


class Aggregator:
    """Token and call totals per node, accumulated by tailing each node's stream."""

    def __init__(self):
        self.offsets: dict[str, int] = {}
        self.stats: dict[str, dict] = {}

    def _blank(self) -> dict:
        return {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0, "tool_calls": 0,
                "errors": 0, "by_role": {}, "by_tool": {}, "llm_seconds": 0.0}

    def refresh(self) -> dict[str, dict]:
        for d in sorted(PATHS.nodes.glob("*/trace.jsonl")):
            nid = d.parent.name
            st = self.stats.setdefault(nid, self._blank())
            evs, off = _read_jsonl(d, self.offsets.get(nid, 0))
            self.offsets[nid] = off
            for e in evs:
                ev = e.get("event", "")
                if ev == "llm.end":
                    role = e.get("role") or "agent"
                    r = st["by_role"].setdefault(role, {"in": 0, "out": 0, "calls": 0})
                    r["in"] += e.get("input_tokens", 0)
                    r["out"] += e.get("output_tokens", 0)
                    r["calls"] += 1
                    st["input_tokens"] += e.get("input_tokens", 0)
                    st["output_tokens"] += e.get("output_tokens", 0)
                    st["llm_calls"] += 1
                    st["llm_seconds"] += e.get("elapsed", 0) or 0
                elif ev == "tool.start":
                    st["tool_calls"] += 1
                    st["by_tool"][e.get("name", "?")] = st["by_tool"].get(e.get("name", "?"), 0) + 1
                elif ev in FATAL or ev.endswith(".error"):
                    st["errors"] += 1
        return self.stats


def _open_spans(path: Path) -> list[dict]:
    """Spans started but not ended: what the loop is doing right now."""
    live: dict[str, dict] = {}
    for e in _read_jsonl(path)[0] if path and path.is_file() else []:
        ev, span = e.get("event", ""), e.get("span")
        if not span:
            continue
        if ev.endswith(".start"):
            live[span] = {"name": ev[:-6], "node": e.get("node"), "ts": e.get("ts")}
        elif ev.endswith((".end", ".error")):
            live.pop(span, None)
    return sorted(live.values(), key=lambda s: s.get("ts") or 0)


def loop_pids() -> list[int]:
    """PIDs of the outer loop itself. A silent trace means one of two very different
    things — the loop is busy inside a long span, or it is gone."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if ("cli.py" in cmd and " run" in cmd) or "kernel.loop" in cmd:
            out.append(int(d.name))
    return out


def _budget_for(name: str) -> int:
    return {"edit_self": BUDGET.edit_self_seconds,
            "improve_recipe": BUDGET.improve_recipe_seconds,
            "evaluate": BUDGET.eval_seconds,
            "root.evaluate": BUDGET.eval_seconds}.get(name, 0)


def summary(agg: Aggregator) -> dict:
    a = Archive()
    stats = agg.refresh()
    nodes = []
    for n in sorted(a.nodes, key=lambda n: n.id):
        nodes.append({
            "id": n.id, "parent": n.parent, "children": n.children, "depth": n.depth,
            "status": n.status, "score": n.score, "full_score": n.full_score,
            "cmp": round(n.cmp, 4), "clade_n": n.clade_n, "clade_w": round(n.clade_w, 3),
            "agent_branch": n.agent_branch, "sana_branch": n.sana_branch,
            "self_edit": n.self_edit, "recipe": n.recipe, "train": n.train,
            "shards": n.shards, "metrics": _flat_metrics(n.metrics),
            "metrics_raw": n.metrics, "n_cases": (n.metrics or {}).get("n_cases"),
            "failure": n.failure,
            "checkpoint_path": n.checkpoint_path, "checkpoint_evicted": n.checkpoint_evicted,
            "created_at": n.created_at, "evaluated_at": n.evaluated_at,
            "spend": stats.get(n.id, {}),
        })
    newest = latest_run()
    live = []
    for s in _open_spans(newest) if newest else []:
        b = _budget_for(s["name"])
        live.append({**s, "elapsed": round(time.time() - (s["ts"] or time.time())),
                     "budget": b})
    selected = None
    for e in reversed(_read_jsonl(newest)[0] if newest else []):
        if e.get("event") == "select":
            selected = {"parent": e.get("parent"), "child": e.get("node"),
                        "parent_cmp": e.get("parent_cmp"), "ts": e.get("ts")}
            break
    # A node's trace stream is opened before its node.json is written. Nothing that
    # was captured should ever be invisible just because the archive entry is younger.
    known = {n.id for n in a.nodes}
    orphans = sorted(set(stats) - known)
    last_ts = 0.0
    if newest:
        tailed = _read_jsonl(newest)[0]
        last_ts = tailed[-1].get("ts", 0) if tailed else newest.stat().st_mtime
    best = a.best()
    total = shutil.disk_usage(PATHS.archive)
    return {
        "now": time.time(),
        "archive": str(PATHS.archive),
        "nodes": nodes,
        "runs": [p.name for p in run_files()][::-1],
        "latest_run": newest.name if newest else "",
        "live": live,
        "selected": selected,
        "orphan_traces": orphans,
        "last_event_ts": last_ts,
        "loop_pids": loop_pids(),
        "cycles": sum(1 for n in a.nodes if n.parent is not None),
        "counts": {"total": len(a.nodes), "ok": len(a.alive()),
                   "trash": sum(1 for n in a.nodes if n.status == "trash"),
                   "pending": sum(1 for n in a.nodes if n.status == "pending")},
        "best": {"id": best.id, "score": best.score} if best else None,
        "disk": {"free_gb": round(total.free / 2**30, 1), "total_gb": round(total.total / 2**30, 1),
                 "floor_gb": BUDGET.min_free_disk_gb},
        "budget": {"max_nodes": BUDGET.max_nodes, "gpus": BUDGET.gpus,
                   "edit_self_s": BUDGET.edit_self_seconds,
                   "improve_recipe_s": BUDGET.improve_recipe_seconds,
                   "eval_s": BUDGET.eval_seconds, "grace_s": BUDGET.train_grace_seconds},
        "eval": {"proxy_cases": EVAL.proxy_cases, "split": EVAL.split,
                 "vlm": EVAL.vlm_enabled},
        # USD per million tokens; unset means the UI shows token counts only, because
        # a made-up price is worse than no price.
        "price": {"in": float(os.environ.get("AR_PRICE_IN", 0) or 0),
                  "out": float(os.environ.get("AR_PRICE_OUT", 0) or 0)},
    }


def node_artifacts(nid: str) -> dict:
    """Videos, reports and logs a human may want to open for one node."""
    root = PATHS.nodes / nid
    if not root.is_dir():
        return {"error": f"unknown node {nid}"}
    vids, logs, reports = [], [], []
    proxy = root / "wbench_proxy" / nid
    # `**` also matches zero directories, so the two globs overlap; dedupe by path.
    for v in sorted({*proxy.glob("videos/*.mp4"), *proxy.glob("videos/**/*.mp4")}):
        vids.append({"name": v.name, "path": str(v), "mb": round(v.stat().st_size / 2**20, 2)})
    for r in sorted(root.rglob("report.json")):
        reports.append({"name": str(r.relative_to(root)), "path": str(r)})
    for d in (root / "logs", proxy):
        if d.is_dir():
            for f in sorted(d.glob("*.log")) + sorted(d.glob("*.json")):
                logs.append({"name": str(f.relative_to(root)), "path": str(f),
                             "kb": round(f.stat().st_size / 1024, 1),
                             "mtime": f.stat().st_mtime})
    return {"node": nid, "videos": vids[:200], "reports": reports, "logs": logs[:120]}


def node_diff(nid: str) -> str:
    """The actual patch a node's meta agent applied to the agent layer."""
    a = Archive()
    n = a.get(nid)
    sha = (n.self_edit or {}).get("sha") if n else None
    if not sha:
        return "(this node recorded no self-edit commit)"
    try:
        out = subprocess.run(["git", "-C", str(PATHS.repo), "show", "--stat", "-p", sha],
                             capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr or "(empty)")[:400_000]
    except Exception as e:  # noqa: BLE001
        return f"(git show failed: {type(e).__name__}: {e})"


_SYS_CACHE: dict = {"ts": 0.0, "value": {}}


def system(ttl: float = 8.0) -> dict:
    if time.time() - _SYS_CACHE["ts"] < ttl:
        return _SYS_CACHE["value"]
    gpus = []
    try:
        q = ("index,name,utilization.gpu,memory.used,memory.total,temperature.gpu")
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=8)
        for line in out.stdout.strip().splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 6:
                gpus.append({"index": f[0], "name": f[1], "util": f[2],
                             "mem_used": f[3], "mem_total": f[4], "temp": f[5]})
    except Exception:  # noqa: BLE001
        pass
    procs_out = []
    try:
        from .. import procs as procmod
        for wt in sorted(PATHS.worktrees.glob("*/sana")):
            for pid in procmod.pids_under(wt):
                try:
                    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                except OSError:
                    cmd = "?"
                procs_out.append({"pid": pid, "worktree": wt.parent.name, "cmd": cmd[:200]})
    except Exception:  # noqa: BLE001
        pass
    val = {"gpus": gpus, "procs": procs_out[:64], "ts": time.time()}
    _SYS_CACHE.update(ts=time.time(), value=val)
    return val
