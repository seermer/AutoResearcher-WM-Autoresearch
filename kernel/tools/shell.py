"""Core shell tool. Every command runs through the security layer."""
from __future__ import annotations

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
    if str(PATHS.wbench) in command and any(w in low for w in (" > ", ">>", "rm ", "mv ", "sed -i", "tee ")):
        return "ERROR: refused: WBench is read-only (evaluation code and data may not be modified)"
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


TOOLS = [run_shell]
