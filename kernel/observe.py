"""Kernel-side capture of every LLM call and tool call.

Installed as a LangChain *inheritable* callback before any agent code is imported,
so it attaches itself to every model and tool the agent layer runs regardless of how
that layer is written. Agents may rewrite roles.py, the graph, or their prompts; they
cannot detach this, because it lives in the kernel and is bound to the context, not
to the objects they construct.

Message bodies are content-addressed into traces/blobs so the JSONL stream stays
small while the UI can still show any prompt or tool output in full.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler

from .config import PATHS
from .trace import Tracer

PREVIEW = 200
_ROLE_RX = re.compile(r"^You are the ([A-Z][A-Z_ ]+?)[.\s]", re.M)


def blobs_dir():
    return PATHS.traces / "blobs"


def put(text: str) -> str:
    """Store text by sha256, return the digest. Identical prompts are stored once."""
    data = (text or "").encode("utf-8", "replace")
    sha = hashlib.sha256(data).hexdigest()
    p = blobs_dir() / sha[:2] / f"{sha}.txt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
    return sha


def get(sha: str) -> str | None:
    if not re.fullmatch(r"[0-9a-f]{64}", sha or ""):
        return None
    p = blobs_dir() / sha[:2] / f"{sha}.txt"
    return p.read_text(errors="replace") if p.is_file() else None


def _text(content: Any) -> str:
    """Flatten a message body, which may be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                out.append(part.get("text") or part.get("thinking") or f"[{part.get('type','block')}]")
        return "\n".join(out)
    return "" if content is None else str(content)


def _ref(content: Any) -> dict:
    """A blob reference plus enough inline text to render a collapsed row."""
    t = _text(content)
    return {"sha": put(t), "chars": len(t), "preview": t[:PREVIEW]}


class Observer(BaseCallbackHandler):
    """Writes llm.* and tool.* records into the node and run trace streams."""

    raise_error = False

    def __init__(self, node_id: str | None, phase: str = ""):
        self.tracer = Tracer(node_id)
        self.phase = phase
        self.t0: dict[str, float] = {}
        self.roles: dict[str, str] = {}

    def _emit(self, event: str, **fields) -> None:
        try:
            self.tracer.emit(event, phase=self.phase, **fields)
        except Exception:  # noqa: BLE001 - observation must never break the run
            pass

    def _start(self, run_id) -> str:
        rid = str(run_id)
        self.t0[rid] = time.time()
        return rid

    def _elapsed(self, run_id) -> float:
        return round(time.time() - self.t0.pop(str(run_id), time.time()), 3)

    # ---- model ----
    def on_chat_model_start(self, serialized, messages, *, run_id=None, parent_run_id=None,
                            **kw) -> None:
        try:
            turn = messages[0] if messages else []
            refs = []
            role = ""
            for m in turn:
                r = getattr(m, "type", "") or m.__class__.__name__
                ref = {"role": r, **_ref(getattr(m, "content", ""))}
                calls = getattr(m, "tool_calls", None)
                if calls:
                    ref["tool_calls"] = [c.get("name") for c in calls]
                if r == "system" and not role:
                    body = _text(getattr(m, "content", ""))
                    hit = _ROLE_RX.search(body)
                    role = (hit.group(1).strip().lower() if hit else "agent")
                refs.append(ref)
            rid = self._start(run_id)
            self.roles[rid] = role
            params = kw.get("invocation_params") or {}
            self._emit("llm.start", call=rid, parent=str(parent_run_id) if parent_run_id else None,
                       role=role, model=params.get("model") or params.get("model_name", ""),
                       n_messages=len(refs), messages=refs)
        except Exception:  # noqa: BLE001
            pass

    def on_llm_end(self, response, *, run_id=None, **kw) -> None:
        try:
            gen = (response.generations or [[]])[0]
            msg = getattr(gen[0], "message", None) if gen else None
            usage = (getattr(msg, "usage_metadata", None) or {}) if msg else {}
            calls = [{"name": c.get("name"), **_ref(str(c.get("args")))}
                     for c in (getattr(msg, "tool_calls", None) or [])]
            self._emit("llm.end", call=str(run_id), role=self.roles.pop(str(run_id), ""),
                       elapsed=self._elapsed(run_id), tool_calls=calls,
                       input_tokens=usage.get("input_tokens", 0),
                       output_tokens=usage.get("output_tokens", 0),
                       **_ref(getattr(msg, "content", "") if msg else ""))
        except Exception:  # noqa: BLE001
            pass

    def on_llm_error(self, error, *, run_id=None, **kw) -> None:
        self._emit("llm.error", call=str(run_id), role=self.roles.pop(str(run_id), ""),
                   elapsed=self._elapsed(run_id), error=f"{type(error).__name__}: {error}"[:2000])

    # ---- tools ----
    def on_tool_start(self, serialized, input_str, *, run_id=None, parent_run_id=None,
                      inputs=None, **kw) -> None:
        try:
            name = (serialized or {}).get("name") or kw.get("name") or "tool"
            body = inputs if inputs is not None else input_str
            if isinstance(body, dict):
                body = "\n".join(f"{k}: {v}" for k, v in body.items())
            self._emit("tool.start", call=self._start(run_id),
                       parent=str(parent_run_id) if parent_run_id else None,
                       name=name, **_ref(body))
        except Exception:  # noqa: BLE001
            pass

    def on_tool_end(self, output, *, run_id=None, **kw) -> None:
        try:
            body = getattr(output, "content", output)
            text = _text(body)
            self._emit("tool.end", call=str(run_id), elapsed=self._elapsed(run_id),
                       failed=text.startswith("ERROR:"), **_ref(text))
        except Exception:  # noqa: BLE001
            pass

    def on_tool_error(self, error, *, run_id=None, **kw) -> None:
        self._emit("tool.error", call=str(run_id), elapsed=self._elapsed(run_id),
                   error=f"{type(error).__name__}: {error}"[:2000])


_VAR = None


def install(node_id: str | None, phase: str = "") -> Observer:
    """Bind an Observer to this process's context for every chain it later runs.

    Idempotent: registering the hook twice would attach two handlers and double every
    record, so the variable is created once and only its value is replaced.
    """
    global _VAR
    from contextvars import ContextVar

    from langchain_core.tracers.context import register_configure_hook

    if _VAR is None:
        _VAR = ContextVar("ar_observer", default=None)
        register_configure_hook(_VAR, True)
    obs = Observer(node_id, phase)
    _VAR.set(obs)
    return obs
