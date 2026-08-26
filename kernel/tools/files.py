"""Core file tools. Reads are broad; writes are confined to the node's worktrees."""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from ..security import SecurityError, assert_writable, is_protected
from . import context

MAX_READ = 200_000


def _readable(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    roots = context.get().read_roots()
    if not any(p == r or r in p.parents for r in roots):
        raise SecurityError(f"outside readable roots: {p}")
    return p


@tool
def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a text file. `offset`/`limit` are line numbers, for large files."""
    p = _readable(path)
    if not p.is_file():
        return f"ERROR: not a file: {p}"
    lines = p.read_text(errors="replace").splitlines()
    chunk = lines[offset: offset + limit]
    body = "\n".join(f"{offset + i + 1}\t{l}" for i, l in enumerate(chunk))[:MAX_READ]
    more = "" if offset + limit >= len(lines) else f"\n... ({len(lines) - offset - limit} more lines)"
    return body + more


@tool
def write_file(path: str, content: str) -> str:
    """Write a text file, creating parent directories. Confined to writable roots."""
    ctx = context.get()
    p = assert_writable(path, ctx.writable)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {p}"


@tool
def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
    """Replace an exact substring in a file. `old` must be unique unless replace_all."""
    ctx = context.get()
    p = assert_writable(path, ctx.writable)
    text = p.read_text(errors="replace")
    n = text.count(old)
    if n == 0:
        return "ERROR: `old` not found"
    if n > 1 and not replace_all:
        return f"ERROR: `old` occurs {n} times; pass replace_all or use a longer anchor"
    p.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1))
    return f"edited {p} ({n if replace_all else 1} replacement(s))"


@tool
def list_dir(path: str, pattern: str = "*", depth: int = 1) -> str:
    """List a directory. `pattern` is a glob; `depth` >1 recurses."""
    p = _readable(path)
    if not p.is_dir():
        return f"ERROR: not a directory: {p}"
    glob = pattern if depth <= 1 else f"**/{pattern}"
    out = []
    for c in sorted(p.glob(glob))[:500]:
        try:
            size = c.stat().st_size if c.is_file() else 0
        except OSError:
            size = 0
        out.append(f"{'d' if c.is_dir() else '-'} {size:>12} {c.relative_to(p)}")
    return "\n".join(out) or "(empty)"


@tool
def search_files(path: str, pattern: str, glob: str = "*.py", max_hits: int = 80) -> str:
    """Grep for a regex across files under `path`."""
    import re
    p = _readable(path)
    rx = re.compile(pattern)
    hits = []
    for f in sorted(p.rglob(glob)):
        if not f.is_file() or is_protected(f):
            continue
        try:
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f}:{i}: {line.strip()[:200]}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits) + "\n... (truncated)"
        except OSError:
            continue
    return "\n".join(hits) or "(no matches)"


TOOLS = [read_file, write_file, edit_file, list_dir, search_files]
