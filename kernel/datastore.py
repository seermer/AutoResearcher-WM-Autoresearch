"""Immutable, content-addressed data shards.

A node must never be able to change what its parent trained on. Shards are written
once, fingerprinted by content, then made read-only; a node's data recipe is a
manifest of shard ids, and its `data/` directory is a farm of symlinks into the
store. New data is produced in a node-private staging area and only enters the
store when the node succeeds, at which point it is immutable too.

Layout:
    datastore/shards/<shard_id>/        the shard's files (read-only)
    datastore/shards/<shard_id>.json    provenance: origin, tags, node, files
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path

from .config import PATHS

BASE_SHARD = "base-sekai"


def shards_dir() -> Path:
    d = PATHS.datastore / "shards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def shard_path(shard_id: str) -> Path:
    return shards_dir() / shard_id


def meta_path(shard_id: str) -> Path:
    return shards_dir() / f"{shard_id}.json"


def fingerprint(directory: Path) -> str:
    """Content hash over relative paths plus file sizes and head/tail bytes.

    Full hashing of a 185 GB zip is not worth the hours; path + size + edges is
    enough to detect a different shard while staying cheap.
    """
    h = hashlib.sha256()
    for f in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        h.update(str(f.relative_to(directory)).encode())
        size = f.stat().st_size
        h.update(str(size).encode())
        with f.open("rb") as fh:
            h.update(fh.read(1 << 20))
            if size > 1 << 21:
                fh.seek(-(1 << 20), os.SEEK_END)
                h.update(fh.read(1 << 20))
    return h.hexdigest()[:16]


def freeze(path: Path) -> None:
    """Drop write bits so even a stray shell command cannot corrupt a shard."""
    for p in [path, *path.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        except OSError:
            pass


def thaw(path: Path) -> None:
    """Only the kernel uses this, and only to delete a shard it is replacing."""
    for p in [path, *path.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass


def register(source: Path, tags: dict, node_id: str = "", move: bool = True) -> str:
    """Seal a directory into the store. Returns its shard id (idempotent by content)."""
    source = Path(source)
    if not source.is_dir() or not any(source.iterdir()):
        raise ValueError(f"not a usable shard directory: {source}")
    shard_id = f"s{fingerprint(source)}"
    dest = shard_path(shard_id)
    if dest.exists():
        return shard_id
    staged = dest.with_suffix(".partial")
    if staged.exists():
        thaw(staged)
        shutil.rmtree(staged)
    # Nest under the source's own name so the dataset directory name — which the
    # training config references — survives sealing.
    staged.mkdir(parents=True)
    inner = staged / source.name
    if move:
        shutil.move(str(source), inner)
    else:
        shutil.copytree(source, inner, symlinks=True)
    staged.rename(dest)
    freeze(dest)
    meta_path(shard_id).write_text(json.dumps({
        "id": shard_id, "node": node_id, "tags": tags, "sealed_at": time.time(),
        "files": sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())[:200],
        "bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()),
    }, indent=2))
    return shard_id


def register_base() -> str:
    """Register the pre-existing Sekai corpus as the root shard, in place.

    It is ~220 GB, so it is referenced by symlink rather than copied, and frozen
    so nothing downstream can write through to it.
    """
    dest = shard_path(BASE_SHARD)
    if dest.exists() or dest.is_symlink():
        return BASE_SHARD
    src = PATHS.sana / "data"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(src.resolve())
    freeze(src.resolve())
    meta_path(BASE_SHARD).write_text(json.dumps({
        "id": BASE_SHARD, "node": "n0000", "tags": {"origin": "Sekai-Game, shipped with Sana"},
        "by_reference": str(src.resolve()), "sealed_at": time.time(),
    }, indent=2))
    return BASE_SHARD


def materialize(manifest: list[str], data_dir: Path) -> list[str]:
    """Build a node's `data/` as symlinks into immutable shards, plus a writable
    `staging/` for anything the node produces."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    linked = []
    for shard_id in manifest:
        src = shard_path(shard_id).resolve()
        if not src.exists():
            continue
        # Every shard holds named dataset directories; link them under their own names.
        for child in sorted(src.iterdir()):
            link = data_dir / child.name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(child.resolve())
                linked.append(child.name)
    (data_dir / "staging").mkdir(exist_ok=True)
    return linked


def seal_staging(staging: Path, node_id: str, tags: dict) -> list[str]:
    """Seal every subdirectory a node produced under `data/staging/` into the store."""
    staging = Path(staging)
    if not staging.is_dir():
        return []
    out = []
    for child in sorted(staging.iterdir()):
        if child.is_dir() and any(child.iterdir()):
            out.append(register(child, {**tags, "name": child.name}, node_id=node_id))
    return out
