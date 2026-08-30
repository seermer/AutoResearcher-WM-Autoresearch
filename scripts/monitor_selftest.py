"""Monitor test: real agent host, real observer, canned LLM endpoint, no GPU, no spend.

scripts/selftest.py stubs Loop._run_agent, so it never crosses the process boundary
where capture actually happens. This one does: it runs kernel/runners/agent_host.py
for real against a local OpenAI-compatible server, then asserts that every prompt,
response and tool result reached the trace and comes back out of the HTTP API whole.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  ' + detail}")
    if not cond:
        FAILURES.append(name)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------- canned LLM ----------------
class LLM(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    n = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        LLM.n += 1
        # first turn of a role calls a tool; the turn after it answers.
        if not any(m.get("role") == "tool" for m in body.get("messages", [])):
            msg = {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{LLM.n}", "type": "function",
                "function": {"name": "list_dir", "arguments": '{"path": "/nope"}'}}]}
            fin = "tool_calls"
        else:
            msg = {"role": "assistant", "content": "CHANGE: none\nHYPOTHESIS: none\nFILES:\n"}
            fin = "stop"
        raw = json.dumps({"id": "c", "object": "chat.completion", "created": 0,
                          "model": body.get("model", "fake"),
                          "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                          "usage": {"prompt_tokens": 1000, "completion_tokens": 20,
                                    "total_tokens": 1020}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve(handler, port):
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def get(url: str) -> bytes:
    return urllib.request.urlopen(url, timeout=20).read()


def code(url: str) -> int:
    try:
        return urllib.request.urlopen(url, timeout=20).getcode()
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ar-monitor-"))
    llm_port, mon_port = free_port(), free_port()
    serve(LLM, llm_port)

    arch, work = tmp / "arch", tmp / "work"
    (work / "roles" / "memory").mkdir(parents=True)
    logs = tmp / "logs"
    logs.mkdir()
    (work / "history.jsonl").write_text('{"relation":"lineage","id":"n0000","score":83.7}\n')
    ctx = tmp / "ctx.json"
    ctx.write_text(json.dumps({
        "ctx": {"node_id": "n0001", "parent_node_id": "n0000", "agents_dir": str(work),
                "history_path": str(work / "history.jsonl"),
                "memory_dir": str(work / "roles" / "memory"), "logs_dir": str(logs),
                "parent_logs_dir": None, "eval_report_path": None, "parent_score": 83.7,
                "parent_metrics": {}, "budget_seconds": 300, "notes": ""},
        "writable": [str(work)], "readable": [str(tmp), str(REPO)],
        "log_dir": str(logs), "shell_timeout": 60}))

    # The loop writes node.json before it launches the agent; mirror that here so the
    # test exercises the same archive state the monitor sees in production.
    nd = arch / "nodes" / "n0001"
    nd.mkdir(parents=True)
    (nd / "node.json").write_text(json.dumps(
        {"id": "n0001", "parent": None, "children": [], "depth": 0, "status": "pending",
         "agent_branch": "node/n0001", "sana_branch": "node/n0001"}))

    env = {**os.environ, "AR_ARCHIVE_DIR": str(arch), "AR_RUN_ID": "test-run",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{llm_port}", "OPENAI_API_KEY": "fake",
           "PYTHONUNBUFFERED": "1"}
    # A node runs out of its own checkout, which carries its own copy of `kernel`
    # next to its own `roles`. Passing REPO as the worktree -- as this test used
    # to -- makes the two indistinguishable and hides which one wins. This stand-in
    # makes the shadow copy loud: importing it writes a marker and then fails.
    wt = tmp / "wt"
    (wt / "kernel").mkdir(parents=True)
    (wt / "roles").symlink_to(REPO / "roles")
    (wt / "kernel" / "__init__.py").write_text(
        "import pathlib\n"
        "pathlib.Path(__file__).with_name('SHADOWED').write_text('x')\n"
        "raise ImportError('the node checkout shadowed the kernel')\n")

    print("running the real agent host against a canned endpoint…")
    proc = subprocess.run([PY, str(REPO / "kernel" / "runners" / "agent_host.py"),
                    "--worktree", str(wt), "--kernel_root", str(REPO),
                    "--phase", "edit_self", "--ctx", str(ctx), "--out", str(tmp / "out.json")],
                   cwd=str(wt), env=env, capture_output=True, timeout=600)

    check("the node checkout cannot shadow the kernel package",
          not (wt / "kernel" / "SHADOWED").exists(),
          proc.stderr.decode("utf-8", "replace")[-400:])
    check("no shadow archive under the node checkout",
          not (wt / "archive").exists() and not (wt / "cache").exists(),
          str(sorted(q.name for q in wt.iterdir())))

    trace = arch / "nodes" / "n0001" / "trace.jsonl"
    check("agent host produced a node trace", trace.is_file(), str(trace))
    if not trace.is_file():
        return finish(tmp)
    recs = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    ev = [r["event"] for r in recs]
    roles = {r.get("role") for r in recs if r["event"] == "llm.start"}

    check("captured LLM calls", ev.count("llm.start") >= 2, str(ev))
    check("every LLM call was paired with a result",
          ev.count("llm.start") == ev.count("llm.end") + ev.count("llm.error"))
    check("captured tool calls", ev.count("tool.start") >= 1)
    check("every tool call was paired", ev.count("tool.start") == ev.count("tool.end") + ev.count("tool.error"))
    check("roles identified from the prompt, not from agent code",
          {"analyst", "meta"} <= roles, str(roles))
    check("token usage recorded",
          all(r.get("input_tokens") for r in recs if r["event"] == "llm.end"))
    check("a refused tool result is flagged as failed",
          any(r.get("failed") for r in recs if r["event"] == "tool.end"))
    check("phase recorded on every record", all(r.get("phase") == "edit_self" for r in recs))
    sysmsg = [m for r in recs if r["event"] == "llm.start"
              for m in r["messages"] if m["role"] == "system"]
    check("system prompt captured", bool(sysmsg) and sysmsg[0]["chars"] > 1000)

    # ---- the API must hand back exactly what was captured ----
    serve_env = dict(os.environ, AR_ARCHIVE_DIR=str(arch))
    mon = subprocess.Popen([PY, "-m", "kernel.monitor", "--port", str(mon_port)],
                           cwd=str(REPO), env=serve_env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{mon_port}"
    for _ in range(60):
        try:
            get(base + "/api/summary")
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.25)
    try:
        check("index page served", get(base + "/").startswith(b"<!doctype html"))
        summ = json.loads(get(base + "/api/summary"))
        check("summary reports the node", any(n["id"] == "n0001" for n in summ["nodes"]))
        check("no captured trace is orphaned", summ["orphan_traces"] == [], str(summ["orphan_traces"]))
        spend = next(n["spend"] for n in summ["nodes"] if n["id"] == "n0001")
        check("spend aggregated per role", spend["input_tokens"] > 0 and spend["by_role"])
        stream = json.loads(get(base + "/api/events"))
        check("event stream served in order", len(stream["events"]) >= len(recs))
        tail = f"{base}/api/events?off={stream['offset']}&file={stream['file']}"
        check("incremental tail returns nothing new", not json.loads(get(tail))["events"])
        subprocess.run([PY, "-c", "from kernel.trace import Tracer;"
                        "Tracer('n0001').emit('node.done', score=84.0)"],
                       cwd=str(REPO), env=env, check=True)
        fresh = json.loads(get(tail))
        check("live tail picks up new events",
              any(e["event"] == "node.done" for e in fresh["events"]))
        check("a rotated run file resets the caller instead of seeking into it",
              json.loads(get(f"{base}/api/events?off=999999&file=gone.jsonl")).get("reset") is True)

        bad = 0
        refs = []
        for r in recs:
            if r["event"] == "llm.start":
                refs += [(m["sha"], m["chars"]) for m in r["messages"]]
            elif r["event"] == "llm.end":
                refs += [(r["sha"], r["chars"])] + [(c["sha"], c["chars"]) for c in r["tool_calls"]]
            elif r["event"] in ("tool.start", "tool.end"):
                refs += [(r["sha"], r["chars"])]
        for sha, chars in refs:
            body = get(f"{base}/api/blob?sha={sha}").decode()
            if len(body) != chars or hashlib.sha256(body.encode()).hexdigest() != sha:
                bad += 1
        check(f"all {len(refs)} blobs round-trip byte-exact (no truncation)", bad == 0, f"{bad} bad")

        check("unknown blob is a 404", code(base + "/api/blob?sha=" + "0" * 64) == 404)
        check("path traversal refused", code(base + "/api/file?path=/etc/passwd") == 404
              and code(base + "/api/tail?path=/etc/passwd") == 404)
        check("unknown route is a 404", code(base + "/nope") == 404)
        check("system panel answers", "gpus" in json.loads(get(base + "/api/system")))
        check("node artifacts answer", "videos" in json.loads(get(base + "/api/node?id=n0001")))
    finally:
        mon.terminate()
    finish(tmp)


def finish(tmp: Path) -> None:
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL CHECKS PASSED" if not FAILURES else f"\nFAILED: {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
