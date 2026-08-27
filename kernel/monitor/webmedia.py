"""Serve browser-playable copies of the run's videos.

The generator writes mp4s through cv2's `mp4v` fourcc, which is MPEG-4 Part 2.
No browser decodes it, so every video pane in the monitor stayed black. Neither
the generator nor the benchmark may be touched, so the monitor transcodes to
H.264 on first play and caches the result next to the other run caches.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from ..config import PATHS

CACHE_MB = 2048
CRF = "23"
# Containers a browser can open directly; anything else is converted regardless
# of what codec is inside it.
WEB_CONTAINERS = (".mp4", ".m4v", ".webm")
WEB_CODECS = ("h264", "hevc", "av1", "vp8", "vp9")
MEDIA_SUFFIXES = (".mp4", ".m4v", ".webm", ".avi", ".mov", ".mkv", ".gif")

# Decisions are remembered in this process only: a bad ffmpeg invocation must not
# leave a marker on disk that makes the failure permanent. Restarting the monitor
# re-probes, and probing costs one short subprocess.
_DECIDED: dict[str, bool] = {}
_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()
# The GPUs are busy with a real run; converting a handful of clips must not turn
# into a CPU storm because a page opened several players at once.
_SLOTS = threading.Semaphore(2)


def _cache() -> Path:
    d = PATHS.cache / "webvideo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ffmpeg() -> str | None:
    """Any ffmpeg on this machine. Nothing installs one system-wide here, but the
    imageio-ffmpeg wheel in the Sana env ships a static build, so look for that
    before giving up."""
    if exe := os.environ.get("AR_FFMPEG"):
        return exe if Path(exe).is_file() else None
    if exe := shutil.which("ffmpeg"):
        return exe
    roots = [Path(sys.prefix), *sorted(Path(sys.prefix).parent.glob("*"))]
    for root in roots:
        for p in sorted(root.glob("lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")):
            if os.access(p, os.X_OK):
                return str(p)
    return None


def _codec(exe: str, src: Path) -> str:
    try:
        r = subprocess.run([exe, "-hide_banner", "-i", str(src)],
                           capture_output=True, text=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        return ""
    m = re.search(r"Video: (\w+)", r.stderr)
    return m.group(1) if m else ""


def _prune(limit_mb: int = CACHE_MB) -> None:
    files = sorted(_cache().glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    while files and total > limit_mb * 2**20:
        victim = files.pop(0)
        total -= victim.stat().st_size
        victim.unlink(missing_ok=True)


def playable_copy(src: Path) -> Path | None:
    """An H.264 copy of `src`, converted once and cached.

    None means serve the original: it is already web-playable, no ffmpeg exists,
    or the conversion failed. A failure is remembered so a broken file does not
    fork ffmpeg on every request; the cache key includes mtime and size, so a
    regenerated video is converted again.
    """
    if src.suffix.lower() not in MEDIA_SUFFIXES:
        return None
    exe = ffmpeg()
    if exe is None:
        return None
    try:
        st = src.stat()
        key = hashlib.sha1(f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}"
                           .encode()).hexdigest()[:16]
    except OSError:
        return None
    out = _cache() / f"{key}.mp4"
    if out.is_file():
        os.utime(out, None)          # keep what is being watched out of the prune
        return out
    if _DECIDED.get(key) is False:
        return None
    with _GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:
        if out.is_file():
            return out
        if _DECIDED.get(key) is False:
            return None
        if src.suffix.lower() in WEB_CONTAINERS and _codec(exe, src) in WEB_CODECS:
            _DECIDED[key] = False
            return None
        tmp = out.with_suffix(".part")
        with _SLOTS:
            # -f mp4 is not optional: the temporary name has no extension ffmpeg
            # recognises, and without it the muxer refuses to open.
            cmd = [exe, "-v", "error", "-y", "-threads", "4", "-i", str(src),
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", CRF,
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                   "-f", "mp4", str(tmp)]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=900)
            except subprocess.CalledProcessError as e:
                err = (e.stderr or b"").decode("utf-8", "replace").strip()[-400:]
                print(f"monitor: transcode failed for {src.name}: {err}", flush=True)
                tmp.unlink(missing_ok=True)
                _DECIDED[key] = False
                return None
            except (subprocess.SubprocessError, OSError) as e:
                print(f"monitor: transcode failed for {src.name}: {e}", flush=True)
                tmp.unlink(missing_ok=True)
                _DECIDED[key] = False
                return None
        tmp.replace(out)
    _prune()
    return out
