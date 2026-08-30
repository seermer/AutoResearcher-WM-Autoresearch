"""The planner's working plan, kept as a file rather than a final message.

A role's answer is one message at the end of its loop, so anything that stops the
loop early loses it: in the first live run one planner hit the step limit and the
node trained on an error string, and another wrote 40k characters of deliberation
that contained none of the fields the graph parses. A plan recorded as it develops
survives both.

Kernel-owned so the fields cannot drift from what the graph reads.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from . import context

FILENAME = "plan.json"


def plan_path(ctx=None) -> Path | None:
    ctx = ctx or context.get()
    return Path(ctx.log_dir) / FILENAME if ctx.log_dir else None


def read_plan(log_dir: Path) -> dict:
    p = Path(log_dir) / FILENAME
    try:
        return json.loads(p.read_text()) if p.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


@tool
def record_plan(plan: str, mechanism: str = "", needs_external_data: bool = False,
                risk: str = "") -> str:
    """Record the current plan. Call this as soon as you have a candidate and again
    whenever it changes: the recorded plan is what the engineer receives, not your
    final message. `plan` is the intervention, `mechanism` is which WBench metric
    moves and why, `risk` is the most likely way it fails."""
    p = plan_path()
    if p is None:
        return "ERROR: no log_dir on this tool context; the plan cannot be recorded"
    body = {"plan": (plan or "").strip(), "mechanism": (mechanism or "").strip(),
            "needs_external_data": bool(needs_external_data), "risk": (risk or "").strip()}
    if not body["plan"]:
        return "ERROR: refused: an empty plan is not a plan"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body, indent=1))
    return f"recorded ({len(body['plan'])} chars). Call again if the plan changes."


TOOLS = [record_plan]
