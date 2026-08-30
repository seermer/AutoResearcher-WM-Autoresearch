"""AutoResearcher CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel import selection  # noqa: E402
from kernel.archive import Archive  # noqa: E402
from kernel.cases import navi_cases, proxy_cases  # noqa: E402
from kernel.config import BUDGET, EVAL, PATHS  # noqa: E402
from kernel.evaluate import SANA_PY, WBENCH_PY  # noqa: E402
from kernel.security import free_disk_gb  # noqa: E402
from kernel import weights  # noqa: E402


def cmd_status(args) -> None:
    a = Archive()
    if not a.nodes:
        print("archive is empty; run `bootstrap`")
        return
    order = {n.id: n for n in a.nodes}

    def walk(nid: str, indent: int = 0):
        n = order[nid]
        d = "" if n.score is None or not n.parent or order[n.parent].score is None \
            else f" ({n.score - order[n.parent].score:+.2f})"
        print(f"{'  ' * indent}{n.id} [{n.status}] score={n.score}{d} "
              f"cmp={n.cmp:.3f} clade={n.clade_n} {(n.recipe.get('plan') or '')[:60]}")
        for c in n.children:
            walk(c, indent + 1)

    root = a.root_node()
    if root:
        walk(root.id)
    best = a.best()
    print(f"\nnodes={len(a.nodes)} alive={len(a.alive())} "
          f"expandable={len(selection.expandable(a))} best={best.id if best else '-'}@{best.score if best else '-'}")


def _busy_gpus() -> list[str]:
    """GPUs already holding someone else's work. Queries nvidia-smi; runs nothing on them."""
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        uuids = {l.strip() for l in r.stdout.splitlines() if l.strip()}
        if not uuids:
            return []
        idx = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        busy = []
        for line in idx.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[1] in uuids:
                busy.append(parts[0])
        return sorted(busy)
    except Exception:  # noqa: BLE001 - a preflight probe must not abort the preflight
        return []


def _dead_online_tools() -> list[tuple[str, bool]]:
    from kernel.tools import web
    probes = [("web_search", web.web_search, {"query": "huggingface datasets", "max_results": 3}),
              ("arxiv_search", web.arxiv_search, {"query": "diffusion", "max_results": 3}),
              ("hf_search", web.hf_search, {"query": "egocentric video camera pose", "limit": 5}),
              ("hf_info", web.hf_info, {"repo_id": "MuteApo/RealCam-Vid", "kind": "dataset"}),
              ("fetch_url", web.fetch_url, {"url": "https://export.arxiv.org/abs/2504.08181"})]
    out = []
    for name, fn, arg in probes:
        try:
            r = fn.invoke(arg)
        except Exception as e:  # noqa: BLE001 - a probe must not abort the preflight
            r = f"ERROR: {type(e).__name__}: {e}"
        out.append((name, r.startswith("ERROR") or r.startswith("(no results")))
    return out


def cmd_doctor(args) -> None:
    ok = True
    checks = [
        ("sana python", Path(SANA_PY).exists()),
        ("wbench python", Path(WBENCH_PY).exists()),
        ("Sana repo", (PATHS.sana / ".git").exists()),
        ("WBench repo", (PATHS.wbench / "main.py").exists()),
        ("WBench cases", len(navi_cases()) == 158),
        ("base corpus", (PATHS.sana / "data" / "sekai_game_train_961frames_16fps_ovl640").exists()),
        ("vae cache", (PATHS.sana / "data" / "vae_cache").exists()),
        ("stage-1 trainer", (PATHS.sana / "train_video_scripts" / "train_sana_wm_stage1.py").exists()),
        ("recipe config", (PATHS.sana / "configs" / "sana_wm" / "stage1" / "sana_wm_stage1_recipe_base.yaml").exists()),
        ("small-GPU recipes", all((PATHS.sana / "configs" / "sana_wm" / "stage1" / f).exists()
                                  for f in ("sana_wm_stage1_recipe_5x24gb.yaml",
                                            "sana_wm_stage1_recipe_4x24gb.yaml"))),
        ("LLM endpoint set", bool(__import__("os").environ.get("OPENAI_BASE_URL"))),
        ("stage-1 weights bundle", (weights.bundle_dir() / weights.DIT_FILE).exists()),
        ("caches off $HOME", not str(__import__("os").environ.get("HF_HOME", "")).startswith(str(Path.home()))),
        (f"disk headroom (>{BUDGET.min_free_disk_gb:.0f} GB)", free_disk_gb() > BUDGET.min_free_disk_gb),
    ]
    for name, good in checks:
        print(f"  {'PASS' if good else 'FAIL'}  {name}")
        ok &= bool(good)
    # A search tool that answers "(no results)" to everything reads to an agent as
    # "this data does not exist", and it spends the node's whole budget concluding
    # that. Each one is asked a question with a known answer.
    # A node launches training on every GPU it is given. Someone else's job sitting on
    # them does not fail the run, it just makes both slower and can push it into OOM.
    busy = _busy_gpus()
    wanted = [g.strip() for g in BUDGET.gpus.split(",") if g.strip()]
    clash = sorted(set(busy) & set(wanted))
    print(f"  {'FAIL' if clash else 'PASS'}  GPUs free"
          + (f" (in use by another process: {','.join(clash)})" if clash else ""))
    ok &= not clash
    for name, dead in _dead_online_tools():
        print(f"  {'FAIL' if dead else 'PASS'}  online tool: {name}")
        ok &= not dead
    print(f"\n  proxy cases: {len(proxy_cases())}  full navi: {len(navi_cases())}")
    print(f"  VLM metrics: {'enabled' if EVAL.vlm_enabled else 'SKIPPED (no VLM_API_KEY)'}")
    print(f"  free disk: {free_disk_gb():.1f} GB   gpus: {BUDGET.gpus}")
    print(f"  archive: {PATHS.archive}")
    sys.exit(0 if ok else 1)


def cmd_bootstrap(args) -> None:
    from kernel.loop import Loop
    node = Loop().bootstrap()
    print(f"root {node.id} score={node.score}")
    print(json.dumps(node.metrics, indent=2)[:2000])


def cmd_run(args) -> None:
    from kernel.loop import Loop
    Loop(seed=args.seed).run(max_nodes=args.max_nodes)


def cmd_eval(args) -> None:
    from kernel.evaluate import Evaluator
    from kernel.vcs import NodeWorkspace
    a = Archive()
    node = a[args.node]
    ws = NodeWorkspace(node.id, node.parent)
    sana = ws.sana if ws.sana.exists() else PATHS.sana
    # Regenerates by default: resuming would score the videos the previous checkpoint
    # left behind and report them as this one's. --resume opts out, for continuing a
    # part-finished rung or re-running the same weights.
    res = Evaluator().run(node.id, sana, node.dir,
                          ckpt=Path(node.checkpoint_path) if node.checkpoint_path else None,
                          full=args.full, all_cases=args.all_cases, resume=args.resume)
    print(json.dumps(res.summary(), indent=2))
    if res.ok and args.full:
        node.full_score = res.score
        a.save(node)


def cmd_stop(args) -> None:
    """Free the GPUs after an interrupted run.

    Training is launched detached (start_new_session), so Ctrl+C on the loop leaves
    torchrun holding every GPU. The kernel only reaps inside its own iteration, so an
    interrupted run needs this.
    """
    from kernel import procs
    from kernel.monitor.data import loop_pids

    if pids := loop_pids():
        print(f"the outer loop is still running (pid {', '.join(map(str, pids))}); "
              f"stop it first:  kill {' '.join(map(str, pids))}")
    killed = []
    for root in sorted(PATHS.worktrees.glob("*/sana")) + sorted(PATHS.worktrees.glob("*/agents")):
        killed += procs.reap(root)
    print(f"reaped {len(killed)} process(es) under node worktrees"
          + (f": {killed}" if killed else ""))
    print("the next `run` will trash any half-finished node and prune the worktrees")


def cmd_monitor(args) -> None:
    from kernel.monitor.server import serve
    serve(args.host, args.port)


def main() -> None:
    p = argparse.ArgumentParser(prog="autoresearcher")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    sub.add_parser("bootstrap").set_defaults(fn=cmd_bootstrap)
    r = sub.add_parser("run"); r.add_argument("--max-nodes", type=int, default=None)
    r.add_argument("--seed", type=int, default=None); r.set_defaults(fn=cmd_run)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    m = sub.add_parser("monitor"); m.add_argument("--port", type=int, default=8787)
    m.add_argument("--host", default="127.0.0.1"); m.set_defaults(fn=cmd_monitor)
    e = sub.add_parser("eval"); e.add_argument("--node", required=True)
    e.add_argument("--full", action="store_true")
    e.add_argument("--all-cases", action="store_true", dest="all_cases",
                   help="with --full: the whole 289-case benchmark, not just the 158 navigation cases")
    # Off by default: resuming would score whatever videos a previous checkpoint left
    # behind and report them as this one's. Safe to pass when re-running the same
    # weights, and the only way to continue a part-finished multi-hour rung.
    e.add_argument("--resume", action="store_true",
                   help="keep videos already generated in the work dir instead of regenerating")
    e.set_defaults(fn=cmd_eval)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
