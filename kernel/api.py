"""The kernel<->agent contract surface.

The kernel constructs these; agent code consumes them. Everything the agent needs
is passed as a *path*, not as inline content, so the agent decides what to read
and its context stays under control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EditSelfContext:
    """Given to edit_self: rewrite your own agent codebase."""
    node_id: str
    parent_node_id: str | None
    agents_dir: Path          # writable: this node's checkout of the agent layer
    history_path: Path        # JSONL: lineage, siblings, archive bests, failures
    memory_dir: Path          # inherited cross-loop memory (inside agents_dir)
    logs_dir: Path            # this node's logs
    eval_report_path: Path | None   # the parent's WBench report.json, if any
    parent_score: float | None
    parent_metrics: dict = field(default_factory=dict)
    budget_seconds: int = 3600
    notes: str = ""


@dataclass
class EditSelfResult:
    summary: str = ""                       # what was changed and why
    hypothesis: str = ""                    # what this change is expected to buy
    changed: list[str] = field(default_factory=list)
    ok: bool = True
    error: str | None = None


@dataclass
class ImproveRecipeContext:
    """Given to improve_recipe: improve the WM training data and produce a checkpoint."""
    node_id: str
    agents_dir: Path          # readable: your own code
    sana_dir: Path            # writable: the WM training codebase for this node
    datastore_dir: Path       # writable, shared, content-addressed: new data lives here
    out_dir: Path             # writable: where the LoRA adapter and manifest go
    history_path: Path
    memory_dir: Path
    logs_dir: Path
    base_checkpoint: Path           # the released stage-1 weights; ALWAYS train from these
    wbench_dir: Path                # READ ONLY: benchmark definition, for understanding tasks
    budget_seconds: int = 12 * 3600
    disk_gb: float = 25.0
    gpus: str = "0,1,2,3,4,5,6,7"


@dataclass
class ImproveRecipeResult:
    checkpoint_path: str | None = None      # merged .pth produced by training
    recipe: dict = field(default_factory=dict)   # manifest: what data, from where, why
    train: dict = field(default_factory=dict)    # steps, config path, wall-clock
    summary: str = ""
    ok: bool = True
    error: str | None = None
