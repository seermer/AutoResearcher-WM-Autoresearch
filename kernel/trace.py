"""Structured JSONL tracing. One stream per node plus a global run stream."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .config import PATHS

_RUN_ID = time.strftime("%Y%m%d-%H%M%S")


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


class Tracer:
    def __init__(self, node_id: str | None = None):
        self.node_id = node_id
        self.run_id = _RUN_ID

    @property
    def _paths(self) -> list[Path]:
        out = [PATHS.traces / f"run-{self.run_id}.jsonl"]
        if self.node_id:
            out.append(PATHS.nodes / self.node_id / "trace.jsonl")
        return out

    def emit(self, event: str, **fields) -> None:
        rec = {"ts": time.time(), "iso": time.strftime("%F %T"), "run": self.run_id,
               "node": self.node_id, "event": event, **fields}
        for p in self._paths:
            _write(p, rec)

    @contextmanager
    def span(self, name: str, **fields):
        span_id = uuid.uuid4().hex[:8]
        t0 = time.time()
        self.emit(f"{name}.start", span=span_id, **fields)
        try:
            yield span_id
        except BaseException as e:
            self.emit(f"{name}.error", span=span_id, elapsed=time.time() - t0,
                      error=f"{type(e).__name__}: {e}")
            raise
        else:
            self.emit(f"{name}.end", span=span_id, elapsed=time.time() - t0)
