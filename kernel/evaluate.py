"""Evaluation access. Kernel-owned: agents can read results but never run or change this."""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import weights
from .cases import case_ids, navi_cases, proxy_cases
from .config import BUDGET, CACHE_ENV, EVAL, PATHS
from .trace import Tracer

SANA_PY = os.environ.get("AR_SANA_PYTHON", "/home/zhantaoy/miniforge3/envs/sana/bin/python")
WBENCH_PY = os.environ.get("AR_WBENCH_PYTHON", "/home/zhantaoy/miniforge3/envs/wbench/bin/python")

DIMENSIONS = {
    "quality": ["aesthetic_quality", "imaging_quality", "temporal_flickering",
                "dynamic_degree", "motion_smoothness", "hpsv3_quality"],
    "consistency": ["background_consistency", "segment_continuity", "perspective_consistency",
                    "subject_consistency", "geometric_consistency", "photometric_consistency",
                    "spatial_consistency", "gated_spatial_consistency"],
    "interaction": ["navigation_trajectory", "event_edit_adherence",
                    "subject_action_adherence", "perspective_switch_adherence"],
    "setting": ["scene_adherence", "subject_adherence"],
    "physical": ["visual_plausibility", "causal_fidelity"],
}
MIN_CASE_SUCCESS = 0.9


@dataclass
class EvalResult:
    ok: bool
    score: float | None = None
    dimensions: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    n_cases: int = 0
    failure: str | None = None
    report_path: str | None = None
    seconds: float = 0.0

    def summary(self) -> dict:
        return {"score": self.score, "dimensions": self.dimensions,
                "metrics": self.metrics, "n_cases": self.n_cases}


def _env_for(conda_python: str, gpus: str) -> dict:
    prefix = str(Path(conda_python).parent.parent)
    return {**CACHE_ENV, "CUDA_VISIBLE_DEVICES": gpus,
            "LD_LIBRARY_PATH": f"{prefix}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
            "TOKENIZERS_PARALLELISM": "false"}


def metric_set_path() -> Path:
    return PATHS.archive / "metric_set.json"


def pinned_metrics() -> list[str] | None:
    """The metric set the archive scores on, fixed by the root node.

    A metric can fail on one node and succeed on another (a missing package, a
    flaky model load). Averaging over whatever happened to run would let a node
    score higher just because a metric that would have dragged it down crashed,
    so the set is pinned and a node missing any of it is failed instead.
    """
    p = metric_set_path()
    return json.loads(p.read_text()) if p.exists() else None


def pin_metrics(metrics: dict) -> list[str]:
    names = sorted(metrics)
    metric_set_path().parent.mkdir(parents=True, exist_ok=True)
    metric_set_path().write_text(json.dumps(names, indent=1))
    return names


def score_from_report(report: dict, split: str = "navi",
                      pinned: list[str] | None = None) -> tuple[float | None, dict, dict, list[str]]:
    """Leaderboard-style scalar: mean over dimensions, restricted to the pinned set."""
    table = report.get(split) or {}
    metrics = {k: round(v["mean"] * 100, 3) for k, v in table.items()
               if isinstance(v, dict) and "mean" in v and v.get("n")}
    missing = sorted(set(pinned) - set(metrics)) if pinned else []
    scored = {k: v for k, v in metrics.items() if pinned is None or k in pinned}
    dims = {}
    for dim, names in DIMENSIONS.items():
        vals = [scored[n] for n in names if n in scored]
        if vals:
            dims[dim] = round(sum(vals) / len(vals), 3)
    score = round(sum(dims.values()) / len(dims), 3) if dims else None
    return score, dims, metrics, missing


def gen_cmd(sana_root: Path, videos: Path, ids_path: Path, shard: int, num_shards: int,
            duration: float, step: int, ckpt: Path | None, base_ckpt: str | None,
            config: str | None, resume: bool) -> list[str]:
    """Build one generation shard's argv. Pure, so the weight selection is testable."""
    cmd = [SANA_PY, str(PATHS.repo / "kernel" / "runners" / "wbench_generate.py"),
           "--sana_root", str(sana_root), "--wbench_root", str(PATHS.wbench),
           "--out_dir", str(videos), "--cases", str(ids_path),
           "--shard", str(shard), "--num_shards", str(num_shards),
           "--duration", str(duration), "--step", str(step),
           "--weights_root", str(weights.bundle_dir())]
    if ckpt:
        cmd += ["--model_path", str(ckpt)]
    elif base_ckpt:
        cmd += ["--model_path", base_ckpt]
    if config:
        cmd += ["--config", config]
    if resume:
        cmd += ["--resume"]
    if EVAL.offload_vae:
        cmd += ["--offload_vae"]
    return cmd


class Evaluator:
    """Runs the fixed WBench protocol against a node's LoRA delta."""

    def __init__(self, tracer: Tracer | None = None):
        self.tracer = tracer or Tracer()

    # ---- generation ----
    def generate(self, node_id: str, sana_root: Path, work_dir: Path, cases,
                 ckpt: Path | None, gpus: list[str], timeout: int,
                 base_ckpt: str | None = None, config: str | None = None,
                 step: int = 60, duration: float = 4.0, resume: bool = True) -> dict:
        videos = work_dir / node_id / "videos"
        videos.mkdir(parents=True, exist_ok=True)
        ids_path = work_dir / node_id / "cases.json"
        ids_path.write_text(json.dumps(case_ids(cases)))

        procs = []
        for shard, gpu in enumerate(gpus):
            cmd = gen_cmd(sana_root, videos, ids_path, shard, len(gpus), duration, step,
                          ckpt, base_ckpt, config, resume)
            log = (work_dir / node_id / f"generate_gpu{gpu}.log").open("wb")
            procs.append((subprocess.Popen(cmd, cwd=str(sana_root), stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           env={**os.environ, **_env_for(SANA_PY, gpu)}), log))

        deadline = time.time() + timeout
        for p, log in procs:
            try:
                p.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(p.pid), 9)
            log.close()

        produced = sorted(videos.glob("case_*_combined.mp4"))
        return {"expected": len(cases), "produced": len(produced),
                "videos_dir": str(videos)}

    # ---- metrics ----
    def score(self, node_id: str, work_dir: Path, gpus: str, timeout: int) -> dict:
        phases = ["precompute", "gpu"]
        if EVAL.vlm_enabled:
            phases.append("vlm")
        phases.append("report")
        log = work_dir / node_id / "evaluate.log"
        for phase in phases:
            cmd = [WBENCH_PY, "main.py", "--model", node_id, "--work_dir", str(work_dir),
                   "--gpus", gpus, "--phase", phase]
            with log.open("ab") as fh:
                fh.write(f"\n===== phase {phase} =====\n".encode())
                p = subprocess.run(cmd, cwd=str(PATHS.wbench), stdout=fh,
                                   stderr=subprocess.STDOUT, timeout=timeout,
                                   env={**os.environ, **_env_for(WBENCH_PY, gpus)})
            if p.returncode != 0:
                return {"error": f"WBench phase {phase} exited {p.returncode}", "log": str(log)}
        report_path = work_dir / node_id / "evaluation" / "report.json"
        if not report_path.exists():
            return {"error": "no report.json produced", "log": str(log)}
        return {"report": json.loads(report_path.read_text()), "report_path": str(report_path)}

    # ---- public ----
    def run(self, node_id: str, sana_root: Path, out_dir: Path, ckpt: Path | None,
            full: bool = False, base_ckpt: str | None = None, config: str | None = None) -> EvalResult:
        t0 = time.time()
        cases = navi_cases() if full else proxy_cases()
        work_dir = out_dir / ("wbench_full" if full else "wbench_proxy")
        work_dir.mkdir(parents=True, exist_ok=True)
        gpus = BUDGET.gpus.split(",")
        weights.ensure_stage1()

        with self.tracer.span("eval.generate", node=node_id, n_cases=len(cases), full=full):
            gen = self.generate(
                node_id, sana_root, work_dir, cases, ckpt, gpus,
                timeout=BUDGET.eval_seconds, base_ckpt=base_ckpt, config=config,
                step=EVAL.full_step if full else EVAL.proxy_step,
                duration=EVAL.turn_duration_s if full else EVAL.proxy_turn_duration_s)
        self.tracer.emit("eval.generated", **gen)
        if gen["produced"] < MIN_CASE_SUCCESS * gen["expected"]:
            return EvalResult(ok=False, seconds=time.time() - t0,
                              failure=f"generation produced {gen['produced']}/{gen['expected']} videos")

        with self.tracer.span("eval.metrics", node=node_id):
            res = self.score(node_id, work_dir, BUDGET.gpus, timeout=BUDGET.eval_seconds)
        if "error" in res:
            return EvalResult(ok=False, seconds=time.time() - t0, failure=res["error"])

        pinned = pinned_metrics()
        score, dims, metrics, missing = score_from_report(res["report"], EVAL.split, pinned)
        if score is None:
            return EvalResult(ok=False, seconds=time.time() - t0,
                              failure="report contained no usable metrics")
        if pinned is None:
            pinned = pin_metrics(metrics)
            self.tracer.emit("eval.metrics_pinned", metrics=pinned)
        elif missing:
            return EvalResult(ok=False, seconds=time.time() - t0,
                              failure=f"metrics missing vs the pinned set: {missing}")
        return EvalResult(ok=True, score=score, dimensions=dims, metrics=metrics,
                          n_cases=gen["produced"], report_path=res["report_path"],
                          seconds=time.time() - t0)
