"""Reclaim sharded FSDP checkpoint state while a node trains.

Sana's `checkpoint_total_limit` prunes only `epoch_*.pth` files; the sibling
`epoch_*_step_*/` directories holding the sharded optimizer+model state are never
removed and cost ~30 GB each. Nothing in this system ever resumes from them --
`Loop._retain` deletes the winner's copy outright -- so on a disk with tens of GB
of headroom they are simply a way to lose a node several hours in.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from .security import free_disk_gb
from .trace import Tracer

QUIET_SECONDS = 180        # leave a save alone until it has settled
KEEP_MERGED = 2            # newest two .pth: the one in flight and the one being reported


def sharded_dirs(root: Path) -> list[Path]:
    """Sharded state directories whose merged .pth already exists."""
    out = []
    for d in Path(root).glob("**/checkpoints/epoch_*_step_*"):
        if d.is_dir() and d.with_suffix(".pth").is_file():
            out.append(d)
    return out


def stale_merged(root: Path, keep: int = KEEP_MERGED) -> list[Path]:
    """Superseded merged checkpoints. `checkpoint_total_limit` does not prune these
    on the FSDP path, so at ~10 GB each a frequent save schedule fills the disk.
    Keep the newest few: the engineer may be about to report one of them."""
    out = []
    for ck in {p.parent for p in Path(root).glob("**/checkpoints/epoch_*_step_*.pth")}:
        pths = sorted(ck.glob("epoch_*_step_*.pth"), key=lambda p: p.stat().st_mtime)
        out.extend(pths[:-keep] if keep > 0 else pths)
    return out


def sweep(root: Path, quiet: float = QUIET_SECONDS) -> list[str]:
    now = time.time()
    freed = []
    for d in sharded_dirs(root):
        try:
            if now - d.stat().st_mtime < quiet:
                continue
            shutil.rmtree(d, ignore_errors=True)
            freed.append(d.name)
        except OSError:
            continue
    for f in stale_merged(root):
        try:
            if now - f.stat().st_mtime < quiet:
                continue
            f.unlink()
            freed.append(f.name)
        except OSError:
            continue
    return freed


def newest_merged(root: Path, min_gb: float = 2.0) -> Path | None:
    """The newest merged checkpoint under a node's training tree, if any.

    Training can succeed and the agent still miss its deadline reporting it. The
    weights are the expensive part, so the kernel looks for them itself rather than
    discarding hours of GPU because a polling loop was late.
    """
    best, best_mtime = None, 0.0
    for f in Path(root).glob("**/checkpoints/epoch_*_step_*.pth"):
        try:
            st = f.stat()
        except OSError:
            continue
        if st.st_size < min_gb * 2**30:
            continue
        if st.st_mtime > best_mtime:
            best, best_mtime = f, st.st_mtime
    return best


class DiskGuard:
    """Background sweeper, started for the duration of one training phase."""

    def __init__(self, root: Path, tracer: Tracer | None = None, interval: float = 60.0):
        self.root, self.interval = Path(root), interval
        self.tracer = tracer or Tracer()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                if freed := sweep(self.root):
                    self.tracer.emit("diskguard.pruned", dirs=freed,
                                     free_gb=round(free_disk_gb(), 1))
            except Exception as e:  # noqa: BLE001 - a guard must never kill the run
                self.tracer.emit("diskguard.error", error=f"{type(e).__name__}: {e}")

    def __enter__(self) -> "DiskGuard":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
