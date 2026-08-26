"""Security layer: what agent code is allowed to touch, and how it runs."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

from .config import (BUDGET, PATHS, PROTECTED_ABSOLUTE, PROTECTED_RELATIVE,
                     PROTECTED_SANA_SUBPATHS)


class SecurityError(RuntimeError):
    pass


def _resolve(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def is_protected(path: str | Path) -> bool:
    """True if `path` lies inside a region agents must never modify."""
    p = _resolve(path)
    for root in PROTECTED_ABSOLUTE:
        if p == root or root in p.parents:
            return True
    # Sana inference is off limits in the base checkout and in every node worktree.
    for base in (PATHS.sana, *(_resolve(w) / "sana" for w in PATHS.worktrees.glob("*"))):
        if base in p.parents:
            rel = p.relative_to(base)
            if rel.parts and rel.parts[0] in PROTECTED_SANA_SUBPATHS:
                return True
    # Relative names (kernel/, .git, .env) are protected only inside agent checkouts,
    # so an unrelated directory named "kernel" elsewhere is not swept up.
    for base in (PATHS.repo, PATHS.worktrees):
        if p == base or base in p.parents:
            rel = p.relative_to(base) if base in p.parents else Path()
            if any(part in PROTECTED_RELATIVE for part in rel.parts):
                return True
    return False


def assert_writable(path: str | Path, roots: list[Path]) -> Path:
    """Path must sit under one of `roots` and outside every protected region."""
    p = _resolve(path)
    if is_protected(p):
        raise SecurityError(f"protected path: {p}")
    if not any(p == r or r in p.parents for r in (_resolve(r) for r in roots)):
        raise SecurityError(f"outside writable roots {roots}: {p}")
    return p


def free_disk_gb(path: Path | None = None) -> float:
    target = Path(path or PATHS.archive)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / 2**30


def assert_disk_headroom(need_gb: float | None = None) -> None:
    need = BUDGET.min_free_disk_gb if need_gb is None else need_gb
    free = free_disk_gb()
    if free < need:
        raise SecurityError(f"insufficient disk: {free:.1f} GB free, need {need:.1f} GB")


def run(cmd: list[str] | str, cwd: Path, timeout: int, env: dict | None = None,
        log_path: Path | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess in its own process group so a timeout kills the whole tree."""
    full_env = {**os.environ, **(env or {})}
    shell = isinstance(cmd, str)
    sink = open(log_path, "ab") if log_path else subprocess.PIPE
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=full_env, shell=shell,
            stdout=sink, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.communicate()
            raise
        out = ""
        if log_path:
            out = Path(log_path).read_text(errors="replace")[-20000:]
        return subprocess.CompletedProcess(cmd, proc.returncode, out, "")
    finally:
        if log_path:
            sink.close()
