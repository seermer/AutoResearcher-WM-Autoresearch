"""End-to-end loop mechanics test with stubbed agents and a stubbed evaluator.

Exercises the real archive, VCS, contract, security, CMP and selection code; only
the LLM calls and GPU work are faked. Run before any real launch.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Same filesystem as the real archive, so the disk-headroom check sees real numbers.
ARCHIVE = Path(tempfile.mkdtemp(prefix="ar-selftest-", dir=str(ROOT.parent)))
os.environ["AR_ARCHIVE_DIR"] = str(ARCHIVE)
sys.path.insert(0, str(ROOT))

from kernel import selection, vcs  # noqa: E402
from kernel.archive import OK, TRASH  # noqa: E402
from kernel.evaluate import EvalResult  # noqa: E402
from kernel.loop import Loop  # noqa: E402

SCRIPT = {}   # node_id -> behaviour for this fake node


def fake_eval(self, node_id, sana_root, out_dir, ckpt=None, full=False, **kw):
    beh = SCRIPT.get(node_id, {})
    if beh.get("eval_fail"):
        return EvalResult(ok=False, failure="stub: generation crashed")
    score = beh.get("score", 70.0)
    return EvalResult(ok=True, score=score, dimensions={"quality": score},
                      metrics={"aesthetic_quality": score}, n_cases=32)


def fake_agent(self, phase, worktree, ctx, writable, readable, log_dir, timeout):
    beh = SCRIPT.get(ctx.node_id, {})
    log_dir.mkdir(parents=True, exist_ok=True)
    if beh.get("kernel_crash") and phase == "improve_recipe":
        raise RuntimeError("simulated kernel-level explosion")
    if beh.get("timeout_after_training") and phase == "improve_recipe":
        # Training finished; only the report was late. The kernel must find the weights.
        ck = Path(ctx.sana_dir) / "output" / "run" / "checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        f = ck / "epoch_1_step_77.pth"          # sparse: real size, no real bytes
        f.touch()
        os.truncate(f, 10 * 2**30)
        return {"ok": False, "error": "improve_recipe exceeded 10800s"}
    if beh.get(f"{phase}_fail"):
        return {"ok": False, "error": f"stub: {phase} refused"}
    if phase == "edit_self":
        assert Path(worktree).name == "agents_frozen", (
            f"edit_self must run from the frozen snapshot, got {worktree}")
        assert all("agents_frozen" not in str(w) for w in writable), (
            "the frozen snapshot must never be writable")
        assert Path(ctx.agents_dir).name == "agents", (
            f"edit_self must write to the draft, got {ctx.agents_dir}")
        if beh.get("no_diff"):
            return {"ok": True, "summary": "no-op"}
        draft = Path(ctx.agents_dir)
        target = draft / "agents" / "prompts" / "analyst.md"
        if beh.get("break_contract"):
            (draft / "agents" / "entrypoints.py").write_text("def edit_self():\n    pass\n")
        elif beh.get("touch_kernel"):
            (draft / "kernel" / "config.py").write_text("BROKEN = 1\n")
        else:
            target.write_text(target.read_text() + f"\n<!-- edit by {ctx.node_id} -->\n")
        return {"ok": True, "summary": f"edit by {ctx.node_id}", "hypothesis": "h", "files": []}
    ckpt_dir = Path(ctx.out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "epoch_1_step_400.pth"
    ckpt.write_bytes(b"stub-checkpoint")
    (Path(ctx.sana_dir) / "configs" / f"{ctx.node_id}.yaml").write_text("stub: 1\n")
    if beh.get("new_data"):
        staged = Path(ctx.sana_dir) / "data" / "staging" / f"corpus_{ctx.node_id}"
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "clips.zip").write_bytes(b"y" * 4096)
    return {"ok": True, "checkpoint_path": str(ckpt),
            "recipe": {"plan": beh.get("plan", f"plan for {ctx.node_id}")},
            "train": {"steps": "300"}}


def check(name: str, cond: bool) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        raise SystemExit(1)


def main() -> None:
    from kernel.evaluate import Evaluator
    Evaluator.run = fake_eval
    Loop._run_agent = fake_agent

    SCRIPT["n0000"] = {"score": 70.0}
    for i, beh in enumerate([
        {"score": 73.0}, {"score": 68.0}, {"break_contract": True}, {"touch_kernel": True},
        {"no_diff": True}, {"eval_fail": True}, {"score": 75.5}, {"improve_recipe_fail": True},
        {"score": 74.0}, {"kernel_crash": True}, {"score": 76.0, "new_data": True},
        {"score": 77.0, "timeout_after_training": True},
    ], start=1):
        SCRIPT[f"n{i:04d}"] = beh

    loop = Loop(seed=7)
    root = loop.bootstrap()
    check("root created and scored", root.status == OK and root.score == 70.0)
    check("root worktrees exist", (ARCHIVE / "worktrees" / "n0000" / "sana").exists())

    for _ in range(12):
        loop.step()

    a = loop.archive
    by = {n.id: n for n in a.nodes}
    check("13 nodes in archive", len(a.nodes) == 13)
    # A late report must not discard a finished training run: 2.5 h of GPU was lost
    # this way before the kernel learned to look for the weights itself.
    salv = by["n0012"]
    check("timed-out node salvaged its checkpoint",
          salv.status == OK and salv.train.get("salvaged") is True)
    check("salvage records why the agent failed",
          "exceeded" in (salv.train.get("agent_error") or ""))
    check("contract violation trashed", by["n0003"].status == TRASH and "contract" in by["n0003"].failure)
    check("kernel edit trashed", by["n0004"].status == TRASH and "protected" in by["n0004"].failure)
    check("empty diff trashed", by["n0005"].status == TRASH and "no diff" in by["n0005"].failure)
    check("eval failure trashed", by["n0006"].status == TRASH and "evaluate" in by["n0006"].failure)
    check("improve_recipe failure trashed",
          by["n0008"].status == TRASH and "improve_recipe" in by["n0008"].failure)
    check("healthy nodes scored", all(by[n].score is not None for n in ("n0001", "n0002", "n0007")))

    for nid in ("n0003", "n0004", "n0005", "n0006", "n0008"):
        check(f"{nid} branch moved to trash namespace",
              vcs.branch_exists(ROOT, vcs.TRASH_BRANCH.format(nid=nid))
              and not vcs.branch_exists(ROOT, vcs.NODE_BRANCH.format(nid=nid)))
        check(f"{nid} worktree removed", not (ARCHIVE / "worktrees" / nid).exists())

    from kernel.config import PATHS
    kept = sorted(PATHS.current.glob("*.pth")) if PATHS.current.exists() else []
    check(f"exactly one checkpoint retained (found {len(kept)})", len(kept) == 1)
    newest = max((n for n in a.alive() if n.evaluated_at), key=lambda n: n.evaluated_at)
    check("retained checkpoint belongs to the newest node", kept[0].stem == newest.id)
    check("older nodes marked evicted",
          all(n.checkpoint_path is None and n.checkpoint_evicted
              for n in a.alive() if n.id != newest.id and n.evaluated_at and n.id != "n0000"))

    # CMP was backed up through every ancestor
    root = by["n0000"]
    check("root clade covers the tree", root.clade_n >= 4)
    check("root cmp in range", 0.0 <= root.cmp <= 1.0)
    best = a.best()
    check("best is the highest scorer", best.score == max(n.score for n in a.alive() if n.score))
    exp = {n.id for n in selection.expandable(a)}
    check("trash is not expandable", not exp & {"n0003", "n0004", "n0005", "n0006", "n0008"})

    # Selection prefers the productive clade but still explores
    rng = random.Random(1)
    picks = [selection.select(a, rng).id for _ in range(2000)]
    top = max(set(picks), key=picks.count)
    check("selection explores >1 node", len(set(picks)) > 1)
    # Thompson sampling is stochastic, so assert the intent -- the modal pick sits in
    # the better half of the expandable set -- rather than an exact rank.
    ranked = sorted(exp, key=lambda i: by[i].cmp, reverse=True)
    rank = ranked.index(top) + 1
    check(f"selection favours the productive half (picked {top}, cmp rank "
          f"{rank}/{len(ranked)})", rank <= max(1, len(ranked) // 2))

    # Regression guard: --weights_root once clobbered --model_path, which would have
    # evaluated every node on base weights and made all scores identical.
    from kernel.evaluate import gen_cmd
    c = gen_cmd(ROOT, ARCHIVE / "v", ARCHIVE / "ids.json", 0, 8, 2.0, 30,
                Path("/node/ckpt.pth"), None, None, True)
    check("node checkpoint reaches the generator", "--model_path" in c and
          c[c.index("--model_path") + 1] == "/node/ckpt.pth")
    check("weights bundle still passed alongside it", "--weights_root" in c)
    base = gen_cmd(ROOT, ARCHIVE / "v", ARCHIVE / "ids.json", 0, 8, 2.0, 30, None, None, None, True)
    check("root node falls back to base weights", "--model_path" not in base)

    # A kernel-level crash inside one node must not take the loop down.
    check("kernel crash trashed its node, loop survived",
          by["n0010"].status == TRASH and "kernel crash" in (by["n0010"].failure or ""))
    check("loop kept going after the crash", by["n0011"].status == OK)

    # Data a node produced is sealed into the immutable store and inherited by name.
    from kernel import datastore as ds
    check("new data sealed into the store", len(by["n0011"].shards) > len(by["n0000"].shards))
    sealed = [x for x in by["n0011"].shards if x not in by["n0000"].shards]
    shard = ds.shard_path(sealed[0])
    check("sealed shard exists", shard.is_dir())
    victim = next(f for f in shard.rglob("*") if f.is_file())
    try:
        victim.write_bytes(b"tamper")
        check("sealed shard is immutable", False)
    except PermissionError:
        check("sealed shard is immutable", True)
    check("parent did not inherit the child's data", by["n0000"].shards != by["n0011"].shards)

    # Recovery: a node abandoned mid-flight is cleaned up, not resumed.
    from kernel.archive import Node, PENDING
    a.add(Node(id="n9999", parent="n0000", depth=1, status=PENDING))
    recovered = loop.recover()
    check("abandoned node recovered", "n9999" in recovered and a["n9999"].status == TRASH)

    print("\n  tree:")
    for n in sorted(a.nodes, key=lambda x: x.id):
        print(f"    {n.id} {n.status:5s} score={n.score} cmp={n.cmp:.3f} "
              f"clade={n.clade_n} {(n.failure or '')[:40]}")
    print(f"\n  selection over 2000 draws: "
          f"{ {k: picks.count(k) for k in sorted(set(picks))} }")
    print("\nALL CHECKS PASSED")


def cleanup() -> None:
    from kernel.config import PATHS
    for repo in (ROOT, PATHS.sana):
        for nid in list(SCRIPT) + ["n0000"]:
            vcs.remove_worktree(repo, ARCHIVE / "worktrees" / nid / ("agents" if repo == ROOT else "sana"))
            for br in (f"node/{nid}", f"trash/{nid}"):
                vcs.git(repo, "branch", "-D", br, check=False)
        vcs.git(repo, "worktree", "prune", check=False)
    shutil.rmtree(ARCHIVE, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
