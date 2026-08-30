"""Role agents. Each is a ReAct loop over the core tools plus whatever the meta
agent has added to tools_ext. Editable."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from agents import Agent, FunctionTool, RunConfig, Runner
from agents.exceptions import MaxTurnsExceeded

from kernel.tools import CORE_TOOLS

from . import memory
from .llm import model_for, settings_for

PROMPTS = Path(__file__).resolve().parent / "prompts"
SKILLS = Path(__file__).resolve().parent / "skills"
STEP_LIMIT = 40
OUTPUT_CHARS = 4000
# A tool loop resends its whole transcript every step, so cost grows with the SQUARE
# of the turn count. Polling a training log for an hour cost 2.5M input tokens over
# 66 turns before this was bounded. The old cap dropped whole messages, which meant a
# role could not see that it had already read a file and read it again -- one engineer
# made 48 reads over 29 distinct files. Trimming the OUTPUT while keeping the call
# itself visible fixes that: the role still sees `read_file(x)` and a preview of what
# came back.
KEEP_RECENT = 4          # turns kept at full fidelity
TOOL_OUTPUT_CHARS = 2000  # older tool results are cut to this
TOOL_PREVIEW_CHARS = 400


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


def environment() -> str:
    """Which interpreter runs what. Read from the kernel so it cannot drift.

    The shell tool runs in the harness environment, which cannot import Sana; a role
    that forgets this spends its whole budget rediscovering it.
    """
    from kernel.evaluate import SANA_PY
    sana_bin = Path(SANA_PY).parent
    return (f"## Environment\n"
            f"Your shell is NOT the training environment. Sana runs in its own env:\n"
            f"- python: `{SANA_PY}`\n"
            f"- launcher: `{sana_bin / 'torchrun'} --nproc_per_node=<N> "
            f"train_video_scripts/train_sana_wm_stage1.py --config_path <cfg>`, run from the Sana worktree\n"
            f"Cache variables (HF_HOME, TORCH_HOME, TMPDIR) are already exported; keep them.\n"
            f"You never run WBench yourself — the kernel evaluates your checkpoint.")


def system_prompt(role: str, memory_dir: Path | None = None, extra: str = "") -> str:
    parts = [(PROMPTS / "common.md").read_text(), (PROMPTS / f"{role}.md").read_text(),
             environment()]
    if index := skill_index():
        parts.append(index)
    if memory_dir:
        digest = memory.digest(Path(memory_dir))
        if digest:
            parts.append("## Memory from earlier loops\n" + digest)
    if extra:
        parts.append(extra)
    return "\n\n---\n\n".join(parts)


def _as_sdk_tool(lc_tool) -> FunctionTool:
    """Expose a kernel tool to the SDK without rewriting it.

    The kernel's tools are LangChain StructuredTools and stay that way: they carry
    the sandbox checks, they are covered by the selftests, and agents may not edit
    them. Only the calling convention is adapted.
    """
    schema = lc_tool.args_schema.model_json_schema()
    schema.setdefault("type", "object")
    schema.pop("title", None)

    async def _invoke(_ctx, args_json: str) -> str:
        try:
            kwargs = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as e:
            return f"ERROR: could not parse arguments: {e}"
        try:
            return str(lc_tool.invoke(kwargs))
        except Exception as e:  # noqa: BLE001 - a tool must report, never abort the run
            return f"ERROR: {lc_tool.name} raised {type(e).__name__}: {e}"

    return FunctionTool(name=lc_tool.name,
                        description=(lc_tool.description or "").strip(),
                        params_json_schema=schema,
                        on_invoke_tool=_invoke,
                        strict_json_schema=False)


def _tools() -> list[FunctionTool]:
    return [_as_sdk_tool(t) for t in (CORE_TOOLS + extra_tools())]


def _account(role: str, usage, turns: int, elapsed: float) -> None:
    """Token accounting. The API is billed per token, so every role call is logged."""
    tin = getattr(usage, "input_tokens", 0) or 0
    tout = getattr(usage, "output_tokens", 0) or 0
    rec = {"ts": time.time(), "role": role, "input_tokens": tin, "output_tokens": tout,
           "turns": turns, "elapsed_s": round(elapsed, 1)}
    print(f"[spend] {role}: in={tin} out={tout} turns={turns} {elapsed:.0f}s", flush=True)
    log = os.environ.get("AR_SPEND_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")


def run(role: str, task: str, memory_dir: Path | None = None, extra: str = "",
        steps: int = STEP_LIMIT, temperature: float = 0.3) -> str:
    """Run one role to completion and return only its final message."""
    from agents.extensions import ToolOutputTrimmer

    agent = Agent(name=role,
                  instructions=system_prompt(role, memory_dir, extra),
                  model=model_for(role),
                  model_settings=settings_for(role, temperature),
                  tools=_tools())
    cfg = RunConfig(call_model_input_filter=ToolOutputTrimmer(
        recent_turns=KEEP_RECENT, max_output_chars=TOOL_OUTPUT_CHARS,
        preview_chars=TOOL_PREVIEW_CHARS))
    t0 = time.time()
    try:
        out = Runner.run_sync(agent, task, max_turns=steps, run_config=cfg)
    except MaxTurnsExceeded as e:
        # Loud on purpose. The previous harness returned the runner's own
        # "Sorry, need more steps to process this request." as if it were the role's
        # answer, and a node wrote that string into its recipe as the plan.
        return f"ERROR: {role} failed: MaxTurnsExceeded: {e}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {role} failed: {type(e).__name__}: {e}"
    _account(role, out.context_wrapper.usage, len(out.new_items), time.time() - t0)
    text = out.final_output
    if not isinstance(text, str):
        text = str(text)
    return (text or "")[:OUTPUT_CHARS]


def failed(text: str) -> str:
    """A role that crashed returns its error as its text; callers must not treat
    that as an answer. Returns the error, or "" when the run was healthy."""
    return text if (text or "").startswith("ERROR:") else ""


def field(text: str, name: str, default: str = "") -> str:
    """Pull a `NAME: value` block out of a role's structured output."""
    m = re.search(rf"^{name}:\s*(.*?)(?=^[A-Z][A-Z_]+:|\Z)", text or "", re.M | re.S)
    return m.group(1).strip() if m else default
