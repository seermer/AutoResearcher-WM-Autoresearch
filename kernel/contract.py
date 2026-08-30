"""The agent contract: every agent codebase must expose edit_self and improve_recipe.

Verified programmatically after every edit_self, before the candidate becomes a node.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import PATHS
from .security import is_protected

ENTRYPOINTS = ("edit_self", "improve_recipe")
# Agents SDK tools that run locally or reach a third party. None of them go through
# the sandbox in kernel/tools, so a role holding one could write anywhere on the box
# or ship the workspace to a hosted service. Every tool an agent uses must come from
# the kernel; these names are refused in the agent layer outright.
BANNED_TOOLS = ("LocalShellTool", "ShellTool", "ComputerTool", "CodeInterpreterTool",
                "ApplyPatchTool", "ProgrammaticToolCallingTool", "HostedMCPTool",
                "WebSearchTool", "FileSearchTool", "ImageGenerationTool", "ToolSearchTool")
ENTRY_MODULE = "roles.entrypoints"

_PROBE = r'''
import importlib, inspect, json, sys
sys.path.insert(0, sys.argv[1])
out = {"ok": False, "entrypoints": {}, "error": None}
try:
    m = importlib.import_module("roles.entrypoints")
    for name in ("edit_self", "improve_recipe"):
        fn = getattr(m, name, None)
        if fn is None or not callable(fn):
            out["entrypoints"][name] = "missing"
            continue
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        required = [p for p in params if p.default is inspect.Parameter.empty]
        out["entrypoints"][name] = "ok" if len(required) == 1 else \
            f"expects 1 required positional arg, found {len(required)}"
    out["ok"] = all(v == "ok" for v in out["entrypoints"].values())
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"
print("__PROBE__" + json.dumps(out))
'''


@dataclass
class ContractResult:
    ok: bool
    reason: str = ""
    detail: dict | None = None

    def __bool__(self) -> bool:
        return self.ok


def check_diff(files: list[str], worktree: Path) -> ContractResult:
    if not files:
        return ContractResult(False, "empty diff: edit_self changed nothing")
    bad = [f for f in files if is_protected(worktree / f)]
    if bad:
        return ContractResult(False, f"diff touches protected paths: {bad[:5]}")
    outside = [f for f in files if not f.startswith("roles/")]
    if outside:
        return ContractResult(False, f"edit_self may only change roles/: {outside[:5]}")
    for f in files:
        path = worktree / f
        if path.suffix != ".py" or not path.is_file():
            continue
        try:
            body = path.read_text(errors="replace")
        except OSError:
            continue
        hits = sorted({t for t in BANNED_TOOLS if t in body})
        if hits:
            return ContractResult(
                False, f"{f} reaches for a tool outside the sandbox: {hits}. "
                       f"Every tool must come from kernel/tools.")
    return ContractResult(True)


def check_compiles(worktree: Path, timeout: int = 120) -> ContractResult:
    p = subprocess.run([sys.executable, "-m", "compileall", "-q", str(worktree / "roles")],
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        return ContractResult(False, "compile failed", {"stderr": p.stderr[-4000:]})
    return ContractResult(True)


def check_entrypoints(worktree: Path, timeout: int = 180) -> ContractResult:
    """Import the candidate agent package in a fresh interpreter and introspect."""
    p = subprocess.run([sys.executable, "-c", _PROBE, str(worktree)],
                       capture_output=True, text=True, timeout=timeout,
                       cwd=str(worktree), env=_probe_env())
    line = next((l for l in p.stdout.splitlines() if l.startswith("__PROBE__")), None)
    if line is None:
        return ContractResult(False, "entrypoint probe produced no result",
                              {"stdout": p.stdout[-2000:], "stderr": p.stderr[-4000:]})
    data = json.loads(line[len("__PROBE__"):])
    if data.get("error"):
        return ContractResult(False, f"import failed: {data['error']}", data)
    if not data["ok"]:
        return ContractResult(False, f"entrypoint contract violated: {data['entrypoints']}", data)
    return ContractResult(True, detail=data)


def _probe_env() -> dict:
    import os
    env = dict(os.environ)
    env["AR_CONTRACT_PROBE"] = "1"          # agent code can short-circuit heavy init
    env["PYTHONPATH"] = str(PATHS.repo)     # kernel always comes from main, never the candidate
    return env


def verify(worktree: Path, changed_files: list[str]) -> ContractResult:
    for check in (lambda: check_diff(changed_files, worktree),
                  lambda: check_compiles(worktree),
                  lambda: check_entrypoints(worktree)):
        res = check()
        if not res:
            return res
    return ContractResult(True, "contract satisfied")
