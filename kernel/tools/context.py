"""Per-invocation sandbox context for the core tools."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PATHS


@dataclass
class ToolContext:
    node_id: str
    writable: list[Path] = field(default_factory=list)
    readable: list[Path] = field(default_factory=list)
    shell_timeout: int = 3600
    log_dir: Path | None = None

    def read_roots(self) -> list[Path]:
        return self.readable + self.writable + [PATHS.wbench, PATHS.sana, PATHS.datastore]


_CTX: ContextVar[ToolContext | None] = ContextVar("ar_tool_ctx", default=None)


def get() -> ToolContext:
    ctx = _CTX.get()
    if ctx is None:
        raise RuntimeError("no ToolContext active; the kernel must open one before running agents")
    return ctx


@contextmanager
def using(ctx: ToolContext):
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)
