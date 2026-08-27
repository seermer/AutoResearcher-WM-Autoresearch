"""Kernel configuration. Not editable by agents."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent
load_dotenv(REPO_ROOT / ".env")


def _env_path(key: str, default: Path) -> Path:
    return Path(os.environ.get(key, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class Paths:
    repo: Path = REPO_ROOT
    workspace: Path = WORKSPACE
    sana: Path = field(default_factory=lambda: _env_path("AR_SANA_DIR", WORKSPACE / "Sana"))
    wbench: Path = field(default_factory=lambda: _env_path("AR_WBENCH_DIR", WORKSPACE / "WBench"))
    archive: Path = field(default_factory=lambda: _env_path("AR_ARCHIVE_DIR", WORKSPACE / "archive"))

    @property
    def nodes(self) -> Path:
        return self.archive / "nodes"

    @property
    def worktrees(self) -> Path:
        return self.archive / "worktrees"

    @property
    def datastore(self) -> Path:
        """Content-addressed store shared across nodes (latents, videos, captions)."""
        return self.archive / "datastore"

    @property
    def traces(self) -> Path:
        return self.archive / "traces"

    @property
    def current(self) -> Path:
        """Holds exactly one checkpoint: the newest node's. Nodes train from base,
        never from each other, so nothing older is ever needed again."""
        return self.archive / "current"

    @property
    def cache(self) -> Path:
        """All model/dataset caches live on the workspace disk, never in $HOME."""
        return _env_path("AR_CACHE_DIR", WORKSPACE / "cache")


@dataclass(frozen=True)
class Budget:
    max_nodes: int = int(os.environ.get("AR_MAX_NODES", 1000))
    edit_self_seconds: int = int(os.environ.get("AR_EDIT_SELF_SECONDS", 3600))
    improve_recipe_seconds: int = int(os.environ.get("AR_IMPROVE_SECONDS", 12 * 3600))
    eval_seconds: int = int(os.environ.get("AR_EVAL_SECONDS", 6 * 3600))
    # Agents choose their own training steps; the kernel caps only wall-clock and disk.
    # Running late is not the same as failing. If the deadline passes while training
    # is still writing to its log, the kernel waits this much longer for it to finish
    # rather than throwing away the GPU hours already spent.
    train_grace_seconds: int = int(os.environ.get("AR_TRAIN_GRACE_SECONDS", 4 * 3600))
    node_disk_gb: float = float(os.environ.get("AR_NODE_DISK_GB", 25))
    min_free_disk_gb: float = float(os.environ.get("AR_MIN_FREE_DISK_GB", 40))
    gpus: str = os.environ.get("AR_GPUS", "0,1,2,3,4,5,6,7")


@dataclass(frozen=True)
class Selection:
    """Thompson sampling over clade-metaproductivity."""
    clade_decay: float = 0.9        # descendant weight per level below the node
    count_failures: bool = True     # failed children contribute x=0 at reduced weight
    failure_weight: float = 0.5
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    softness: float = 2.0           # score points mapping to ~1 logit of normalized gain


@dataclass(frozen=True)
class EvalCfg:
    split: str = "navi"
    proxy_cases: int = int(os.environ.get("AR_PROXY_CASES", 32))
    proxy_seed: int = 20260826
    # The proxy rung only has to *rank* nodes, so it generates shorter clips with
    # fewer sampling steps. The full 158-case promotion eval uses canonical settings.
    turn_duration_s: float = 4.0
    proxy_turn_duration_s: float = float(os.environ.get("AR_PROXY_TURN_SECONDS", 2.0))
    proxy_step: int = int(os.environ.get("AR_PROXY_STEP", 30))
    # Stage-1 inference peaks around 33 GB on a full-length case; set this on hosts
    # whose GPUs are smaller than that.
    offload_vae: bool = os.environ.get("AR_OFFLOAD_VAE", "0") not in ("0", "", "false")
    full_step: int = int(os.environ.get("AR_FULL_STEP", 60))
    # VLM metrics need a Doubao/ARK key; skipped (not removed) when absent.
    gpu_metrics: tuple[str, ...] = ("quality", "consistency")
    vlm_metrics: tuple[str, ...] = ("setting", "interaction", "physical")

    @property
    def vlm_enabled(self) -> bool:
        return bool(os.environ.get("VLM_API_KEY"))

    def vlm_metrics_names(self) -> tuple[str, ...]:
        """Metric names that only a VLM can produce; expected to be absent without a key."""
        if self.vlm_enabled:
            return ()
        return ("event_edit_adherence", "subject_action_adherence", "perspective_switch_adherence",
                "scene_adherence", "subject_adherence", "visual_plausibility", "causal_fidelity")


PATHS = Paths()


def use_workspace_caches() -> dict:
    """Force HF, torch and temp files onto the workspace disk.

    The root filesystem here is shared and full, so anything that defaults to
    $HOME/.cache or /tmp will fail mid-download. Set before any heavy import and
    inherited by every subprocess the kernel launches.
    """
    root = PATHS.cache
    env = {
        "HF_HOME": str(root / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(root / "huggingface" / "hub"),
        "TORCH_HOME": str(root / "torch"),
        "XDG_CACHE_HOME": str(root / "xdg"),
        "TMPDIR": str(root / "tmp"),
        "TRITON_CACHE_DIR": str(root / "triton"),
    }
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    for key, value in env.items():
        os.environ.setdefault(key, value)
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    return env


CACHE_ENV = use_workspace_caches()

BUDGET = Budget()
SELECTION = Selection()
EVAL = EvalCfg()

# Paths agents may never write to.
# Relative names, enforced inside agent checkouts only.
PROTECTED_RELATIVE = ("kernel", ".git", ".env")
# Whole trees that are off limits everywhere: the benchmark.
PROTECTED_ABSOLUTE = (PATHS.wbench,)
# Sana subtrees agents may not change. Improvement must come from data, so the
# training path is editable and the inference path is not.
PROTECTED_SANA_SUBPATHS = ("inference_video_scripts",)
