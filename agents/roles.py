"""Role agents. Each is a ReAct loop over the core tools plus whatever the meta
agent has added to tools_ext. Editable."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from kernel.tools import CORE_TOOLS

from . import memory
from .llm import chat

PROMPTS = Path(__file__).resolve().parent / "prompts"
SKILLS = Path(__file__).resolve().parent / "skills"
STEP_LIMIT = 40
OUTPUT_CHARS = 4000


def extra_tools() -> list:
    """Tools the meta agent has added over time. Never let a broken one kill a run."""
    try:
        from . import tools_ext
        return list(getattr(tools_ext, "TOOLS", []))
    except Exception as e:  # noqa: BLE001
        print(f"[roles] tools_ext unavailable: {type(e).__name__}: {e}")
        return []


def skill_index() -> str:
    """Filename + first line of each playbook. Bodies are read on demand, not inlined."""
    rows = []
    for f in sorted(SKILLS.glob("*.md")):
        if f.name == "README.md":
            continue
        first = next((l.strip() for l in f.read_text().splitlines() if l.strip()), "")
        rows.append(f"- `{f}` — {first[:110]}")
    if not rows:
        return ""
    return ("## Skills available\nRead one with read_file when it applies; "
            "add a new one when a procedure has worked twice.\n" + "\n".join(rows))


def system_prompt(role: str, memory_dir: Path | None = None, extra: str = "") -> str:
    parts = [(PROMPTS / "common.md").read_text(), (PROMPTS / f"{role}.md").read_text()]
    if index := skill_index():
        parts.append(index)
    if memory_dir:
        digest = memory.digest(Path(memory_dir))
        if digest:
            parts.append("## Memory from earlier loops\n" + digest)
    if extra:
        parts.append(extra)
    return "\n\n---\n\n".join(parts)


def _account(role: str, messages: list, elapsed: float) -> None:
    """Token accounting. The API is billed per token, so every role call is logged."""
    tin = tout = 0
    for m in messages:
        u = getattr(m, "usage_metadata", None) or {}
        tin += u.get("input_tokens", 0)
        tout += u.get("output_tokens", 0)
    rec = {"ts": time.time(), "role": role, "input_tokens": tin, "output_tokens": tout,
           "turns": len(messages), "elapsed_s": round(elapsed, 1)}
    print(f"[spend] {role}: in={tin} out={tout} turns={len(messages)} {elapsed:.0f}s", flush=True)
    log = os.environ.get("AR_SPEND_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")


def run(role: str, task: str, memory_dir: Path | None = None, extra: str = "",
        steps: int = STEP_LIMIT, temperature: float = 0.3) -> str:
    """Run one role to completion and return only its final message."""
    agent = create_react_agent(chat(role, temperature=temperature), CORE_TOOLS + extra_tools())
    msgs = [SystemMessage(system_prompt(role, memory_dir, extra)), HumanMessage(task)]
    t0 = time.time()
    try:
        out = agent.invoke({"messages": msgs}, {"recursion_limit": steps})
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {role} failed: {type(e).__name__}: {e}"
    _account(role, out["messages"], time.time() - t0)
    text = out["messages"][-1].content
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    return (text or "")[:OUTPUT_CHARS]


def failed(text: str) -> str:
    """A role that crashed returns its error as its text; callers must not treat
    that as an answer. Returns the error, or "" when the run was healthy."""
    return text if (text or "").startswith("ERROR:") else ""


def field(text: str, name: str, default: str = "") -> str:
    """Pull a `NAME: value` block out of a role's structured output."""
    m = re.search(rf"^{name}:\s*(.*?)(?=^[A-Z][A-Z_]+:|\Z)", text or "", re.M | re.S)
    return m.group(1).strip() if m else default
