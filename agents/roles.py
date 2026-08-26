"""Role agents. Each is a ReAct loop over the core tools plus whatever the meta
agent has added to tools_ext. Editable."""
from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from kernel.tools import CORE_TOOLS

from . import memory
from .llm import chat

PROMPTS = Path(__file__).resolve().parent / "prompts"
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


def system_prompt(role: str, memory_dir: Path | None = None, extra: str = "") -> str:
    parts = [(PROMPTS / "common.md").read_text(), (PROMPTS / f"{role}.md").read_text()]
    if memory_dir:
        digest = memory.digest(Path(memory_dir))
        if digest:
            parts.append("## Memory from earlier loops\n" + digest)
    if extra:
        parts.append(extra)
    return "\n\n---\n\n".join(parts)


def run(role: str, task: str, memory_dir: Path | None = None, extra: str = "",
        steps: int = STEP_LIMIT, temperature: float = 0.3) -> str:
    """Run one role to completion and return only its final message."""
    agent = create_react_agent(chat(role, temperature=temperature), CORE_TOOLS + extra_tools())
    msgs = [SystemMessage(system_prompt(role, memory_dir, extra)), HumanMessage(task)]
    try:
        out = agent.invoke({"messages": msgs}, {"recursion_limit": steps})
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {role} failed: {type(e).__name__}: {e}"
    text = out["messages"][-1].content
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    return (text or "")[:OUTPUT_CHARS]


def field(text: str, name: str, default: str = "") -> str:
    """Pull a `NAME: value` block out of a role's structured output."""
    m = re.search(rf"^{name}:\s*(.*?)(?=^[A-Z][A-Z_]+:|\Z)", text or "", re.M | re.S)
    return m.group(1).strip() if m else default
