"""python -m kernel.monitor --port 8787"""
from __future__ import annotations

import argparse

from .server import serve

ap = argparse.ArgumentParser(prog="kernel.monitor")
ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only behind a trusted network")
ap.add_argument("--port", type=int, default=8787)
a = ap.parse_args()
serve(a.host, a.port)
