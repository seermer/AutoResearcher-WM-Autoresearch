"""Core shell tool. Every command runs through the security layer."""
from __future__ import annotations

import time

from langchain_core.tools import tool

from ..config import BUDGET, PATHS
from ..security import SecurityError, free_disk_gb, is_protected, run
from . import context

DENY = ("rm -rf /", "mkfs", ":(){", "shutdown", "reboot", "dd if=/dev/zero of=/dev")


@tool
def run_shell(command: str, cwd: str = "", timeout: int = 1800) -> str:
    """Run a shell command and return its combined output (tail-truncated).

    Use this for training, data processing, downloads and git. It cannot write to
    WBench or to the kernel. Long jobs should be run with nohup + a log file, then
    polled, so a single call does not block for hours.
    """
    ctx = context.get()
    free = free_disk_gb()
    if free < BUDGET.min_free_disk_gb:
        return (f"ERROR: refused: only {free:.1f} GB free (floor {BUDGET.min_free_disk_gb:.0f} GB). "
                f"Delete raw footage you have already encoded, or pick a smaller source.")
    low = command.lower()
    if any(d in low for d in DENY):
        return "ERROR: refused by security policy"
    writes = any(w in low for w in (" > ", ">>", "rm ", "mv ", "sed -i", "tee ", "truncate", "chmod "))
    if str(PATHS.wbench) in command and writes:
        return "ERROR: refused: WBench is read-only (evaluation code and data may not be modified)"
    # The trace stream is the run's only record of what every agent did. Agents have
    # no reason to write there and the file tools already forbid it; this closes the
    # one path that does not go through them. Note out_dir lives under nodes/<id>/work,
    # so only the record files themselves are named here, never the node tree.
    if writes and (str(PATHS.traces) in command
                   or any(f in command for f in ("node.json", "trace.jsonl"))):
        return ("ERROR: refused: the run's trace and node records are read-only. "
                "Write your own outputs under out_dir instead.")
    work = (ctx.writable[0] if ctx.writable else PATHS.archive) if not cwd else cwd
    from pathlib import Path
    work = Path(work).expanduser().resolve()
    if is_protected(work):
        return f"ERROR: refused: protected cwd {work}"
    log = (ctx.log_dir / "shell.log") if ctx.log_dir else None
    try:
        res = run(command, cwd=work, timeout=min(timeout, ctx.shell_timeout), log_path=log)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
    tail = (res.stdout or "")[-12000:]
    return f"exit={res.returncode}\n{tail}"


@tool
def wait_for_training(log_path: str, timeout: int = 43200, quiet_seconds: int = 900) -> str:
    """Block until the training job finishes, then return the tail of its log.

    Use this ONCE after launching training instead of polling. Polling costs one LLM
    turn per check and each turn resends the whole transcript, so a job watched for an
    hour can cost millions of tokens; this call costs one. Returns when no process is
    left in your Sana worktree, or when the log has been silent for `quiet_seconds`
    (a crashed job), or at `timeout`.
    """
    ctx = context.get()
    from pathlib import Path

    from .. import procs
    root = ctx.writable[0] if ctx.writable else PATHS.archive
    log = Path(log_path).expanduser()
    limit = min(timeout, ctx.shell_timeout)
    deadline = time.time() + limit
    started = time.time()
    while time.time() < deadline:
        alive = procs.pids_under(root)
        if not alive:
            break
        if log.exists() and not procs.progressing(log, quiet_seconds):
            tail = log.read_text(errors="replace")[-6000:]
            return (f"STALLED: no output for {quiet_seconds}s while {len(alive)} process(es) "
                    f"were still up. Treat as a hang.\n{tail}")
        time.sleep(30)
    elapsed = time.time() - started
    tail = log.read_text(errors="replace")[-6000:] if log.exists() else "(no log)"
    state = "FINISHED" if not procs.pids_under(root) else f"STILL RUNNING after {limit}s"
    return f"{state} (waited {elapsed:.0f}s)\n{tail}"


TOOLS = [run_shell, wait_for_training]
