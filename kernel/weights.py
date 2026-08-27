"""Provision exactly the weights stage-1-only inference needs, and nothing else.

`SANA-WM_bidirectional` is ~96 GB, but 84 GB of that is the LTX-2 refiner and its
Gemma-3-12B text encoder, which we never load (the baseline is stage-1 without the
refiner). Letting diffusers resolve the repo id at load time pulls the whole thing,
so the kernel materialises a local bundle up front and hands out filesystem paths.
"""
from __future__ import annotations

from pathlib import Path

from .config import PATHS

REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
TEXT_REPO = "Efficient-Large-Model/gemma-2-2b-it"
STAGE1_PATTERNS = ["config.yaml", "dit/*", "vae/*"]
DIT_FILE = "dit/sana_wm_1600m_720p.safetensors"

# WBench's MegaSAM stage pulls this at run time, once per worker. Eight workers
# racing a cold cache all fail, MegaSAM writes no poses, and navigation_trajectory
# plus the spatial metrics silently report zero applicable cases. Warm it serially.
UNIDEPTH_REPO = "lpiccinelli/unidepth-v2-vitl14"
UNIDEPTH_REVISION = "1d0d3c52f60b5164629d279bb9a7546458e6dcc4"


def bundle_dir() -> Path:
    return PATHS.cache / "weights" / "sana_wm_stage1"


def ensure_stage1(force: bool = False) -> dict:
    """Download config + DiT + VAE only. Returns local paths."""
    from huggingface_hub import snapshot_download

    root = bundle_dir()
    if force or not (root / DIT_FILE).exists() or not (root / "vae").is_dir():
        snapshot_download(repo_id=REPO, local_dir=str(root), allow_patterns=STAGE1_PATTERNS)
    text = PATHS.cache / "weights" / "gemma-2-2b-it"
    if force or not any(text.glob("*.safetensors")):
        snapshot_download(repo_id=TEXT_REPO, local_dir=str(text))
    return {"root": root, "config": root / "config.yaml",
            "dit": root / DIT_FILE, "text_encoder": text}


def ensure_metric_models() -> dict:
    """Pre-download the metric weights WBench fetches from the Hub."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=UNIDEPTH_REPO, revision=UNIDEPTH_REVISION,
                             allow_patterns=["config.json", "model.safetensors"])
    return {"unidepth": Path(root)}


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**30
