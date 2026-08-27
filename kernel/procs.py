"""Find and reap the training processes belonging to one node.

`run_shell` starts every command in its own session so a timeout kills the tree,
but a job the agent launched with nohup outlives the agent host. Without this a
timed-out node leaves training running on every GPU while the loop starts the
next one.
"""
from __future__ import annotations

import os
import signal
import time
from pathlib import Path


def _cwd_of(pid: str) -> Path | None:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return None


def pids_under(root: Path) -> list[int]:
    """PIDs whose working directory is inside `root`, excluding this process."""
    root = Path(root).resolve()
    me = os.getpid()
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        cwd = _cwd_of(entry)
        if cwd is None:
            continue
        try:
            if cwd == root or root in cwd.parents:
                out.append(int(entry))
        except OSError:
            continue
    return out


def progressing(log: Path, within: float = 600.0) -> bool:
    """Has the training log been written to recently?"""
    try:
        return (time.time() - Path(log).stat().st_mtime) < within
    except OSError:
        return False


def newest_log(root: Path) -> Path | None:
    logs = list(Path(root).glob("**/output/**/train_log.log"))
    return max(logs, key=lambda p: p.stat().st_mtime, default=None)


def wait_for_exit(root: Path, timeout: float, poll: float = 30.0) -> bool:
    """Block until nothing under `root` is running. True if it drained in time."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pids_under(root):
            return True
        time.sleep(poll)
    return not pids_under(root)


def reap(root: Path, grace: float = 10.0) -> list[int]:
    """Terminate, then kill, everything still running under `root`."""
    killed = pids_under(root)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = pids_under(root)
        if not alive:
            break
        for pid in alive:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        if sig is signal.SIGTERM:
            time.sleep(grace)
    return killed
