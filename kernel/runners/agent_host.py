"""Run one agent entry point out of a node's own checkout, in an isolated process.

The node's agent code goes first on sys.path so the candidate — not the kernel's
baseline copy — is what actually runs; the kernel package still resolves from the
main checkout, so agents cannot shadow it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--kernel_root", required=True)
    ap.add_argument("--phase", required=True, choices=["edit_self", "improve_recipe"])
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.kernel_root)
    sys.path.insert(0, args.worktree)

    out_path = Path(args.out)
    try:
        from kernel import api
        from kernel.tools import context as tool_ctx

        payload = json.loads(Path(args.ctx).read_text())
        cls = api.EditSelfContext if args.phase == "edit_self" else api.ImproveRecipeContext
        fields = {f.name: f.type for f in dataclasses.fields(cls)}
        kwargs = {}
        for k, v in payload["ctx"].items():
            if k not in fields:
                continue
            kwargs[k] = Path(v) if (v is not None and "Path" in str(fields[k])) else v
        ctx = cls(**kwargs)

        sandbox = tool_ctx.ToolContext(
            node_id=ctx.node_id,
            writable=[Path(p) for p in payload["writable"]],
            readable=[Path(p) for p in payload["readable"]],
            shell_timeout=payload.get("shell_timeout", 3600),
            log_dir=Path(payload["log_dir"]),
        )
        with tool_ctx.using(sandbox):
            import agents.entrypoints as ep
            result = getattr(ep, args.phase)(ctx)
        out_path.write_text(json.dumps(dataclasses.asdict(result), default=str))
    except BaseException as e:  # noqa: BLE001
        out_path.write_text(json.dumps(
            {"ok": False, "error": f"{type(e).__name__}: {e}",
             "trace": traceback.format_exc()[-4000:]}))
        raise


if __name__ == "__main__":
    main()
