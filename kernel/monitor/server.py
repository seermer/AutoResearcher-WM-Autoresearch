"""Read-only HTTP server for the monitor UI. Standard library only."""
from __future__ import annotations

import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..config import PATHS
from . import data, webmedia
from .. import observe

APP = Path(__file__).resolve().parent / "app.html"
# Everything the browser may open. All read-only, all inside the run's own trees.
SERVABLE = ("archive", "sana", "wbench")
MAX_TAIL = 400_000


def _allowed(p: Path) -> bool:
    roots = [getattr(PATHS, n) for n in SERVABLE]
    try:
        p = p.resolve()
    except OSError:
        return False
    return any(p == r or r in p.parents for r in roots)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    agg = data.Aggregator()

    def log_message(self, *a):        # keep the console clean for the loop's own output
        pass

    # ---- plumbing ----
    def _send(self, body: bytes, ctype: str, code: int = 200, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj, code: int = 200):
        self._send(json.dumps(obj, default=str).encode(), "application/json", code)

    def _text(self, s: str, code: int = 200):
        self._send(s.encode("utf-8", "replace"), "text/plain; charset=utf-8", code)

    def _file(self, path: Path):
        """Serves media with Range support so the browser can seek in a video."""
        if not _allowed(path) or not path.is_file():
            return self._json({"error": "not servable"}, 404)
        # The source is what gets authorised; the substitute is a cached H.264
        # copy of it, because the generator's mp4v stream will not play.
        path = webmedia.playable_copy(path) or path
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        if m and (m.group(1) or m.group(2)):
            start = int(m.group(1)) if m.group(1) else max(0, size - int(m.group(2)))
            end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
            end = min(end, size - 1)
            with path.open("rb") as f:
                f.seek(start)
                chunk = f.read(end - start + 1)
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            try:
                self.wfile.write(chunk)
            except BrokenPipeError:
                pass
            return
        self._send(path.read_bytes(), ctype, extra={"Accept-Ranges": "bytes"})

    # ---- routes ----
    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            self.route(u.path, q)
        except Exception as e:  # noqa: BLE001 - a broken panel must not kill the server
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def route(self, path: str, q: dict):
        if path in ("/", "/index.html"):
            return self._send(APP.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/summary":
            return self._json(data.summary(self.agg))
        if path == "/api/events":
            return self._json(data.events(q.get("run", ""), int(q.get("off", 0)),
                                          int(q.get("tail", 2000)), q.get("file", "")))
        if path == "/api/blob":
            body = observe.get(q.get("sha", ""))
            return self._text(body if body is not None else "(blob not found)",
                              200 if body is not None else 404)
        if path == "/api/node":
            return self._json(data.node_artifacts(q.get("id", "")))
        if path == "/api/diff":
            return self._text(data.node_diff(q.get("id", "")))
        if path == "/api/system":
            return self._json(data.system())
        if path == "/api/tail":
            p = Path(q.get("path", "")).expanduser()
            if not _allowed(p) or not p.is_file():
                return self._text("(not readable)", 404)
            n = min(int(q.get("bytes", 60_000)), MAX_TAIL)
            with p.open("rb") as f:
                f.seek(max(0, p.stat().st_size - n))
                return self._text(f.read().decode("utf-8", "replace"))
        if path == "/api/file":
            return self._file(Path(q.get("path", "")).expanduser())
        return self._json({"error": "no such route"}, 404)


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    print(f"monitor: http://{host}:{port}  (archive {PATHS.archive})", flush=True)
    srv.serve_forever()
