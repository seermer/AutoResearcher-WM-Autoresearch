"""Cross-loop memory. Lives inside the agent checkout, so it forks with the lineage.

Deliberately small and capped: memory is carried in every prompt, so it must stay
short enough that it never crowds out the working context.
"""
from __future__ import annotations

from pathlib import Path

FILES = {
    "lessons": "What worked and what did not, one bullet each.",
    "recipes": "Data recipes tried, with their score deltas.",
    "sources": "Data sources found: URL, size, licence, how to ingest.",
}
MAX_CHARS = 6000       # per file; older lines are dropped first
MAX_PROMPT_CHARS = 9000


def path(memory_dir: Path, name: str) -> Path:
    return Path(memory_dir) / f"{name}.md"


def read(memory_dir: Path, name: str) -> str:
    p = path(memory_dir, name)
    return p.read_text() if p.exists() else ""


def append(memory_dir: Path, name: str, line: str) -> None:
    """Append one bullet, trimming the oldest lines to stay under the cap."""
    p = path(memory_dir, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = (p.read_text() if p.exists() else f"# {name}\n\n{FILES.get(name,'')}\n\n")
    body += f"- {line.strip()}\n"
    if len(body) > MAX_CHARS:
        head, _, rest = body.partition("\n\n")
        lines = rest.strip().splitlines()
        while lines and len(head) + sum(len(l) + 1 for l in lines) > MAX_CHARS:
            lines.pop(0)
        body = head + "\n\n" + "\n".join(lines) + "\n"
    p.write_text(body)


def digest(memory_dir: Path) -> str:
    """The whole memory, capped, for injection into a system prompt."""
    parts = []
    for name in FILES:
        text = read(memory_dir, name).strip()
        if text:
            parts.append(text)
    out = "\n\n".join(parts)
    return out[-MAX_PROMPT_CHARS:] if len(out) > MAX_PROMPT_CHARS else out
