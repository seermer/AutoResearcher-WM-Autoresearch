"""Print a WBench report.json the way the leaderboard reads it.

The kernel's own scorer (`evaluate.score_from_report`) restricts to the metric set
pinned by the root node, on purpose: a node must not score higher because a metric
that would have dragged it down happened to crash. That pinned set was fixed on a
run with no VLM key, so it covers 14 GPU metrics only. A one-off benchmark run wants
the opposite view -- every metric that produced a number -- so this reports both.

    python scripts/report_full.py <report.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel.evaluate import DIMENSIONS, pinned_metrics, score_from_report  # noqa: E402


def table(report: dict, split: str, pinned: list[str] | None) -> None:
    score, dims, metrics, missing = score_from_report(report, split, pinned)
    scored = {k: v for k, v in metrics.items() if pinned is None or k in pinned}
    n = report.get("n_navi") if split == "navi" else report.get("n_cases")
    label = "pinned 14-metric set" if pinned else "every metric present"
    print(f"\n  split={split}  cases={n}  ({label})")
    for dim, names in DIMENSIONS.items():
        head = f"{dims[dim]:.3f}" if dim in dims else "  --  "
        print(f"    {dim:<12s} {head}")
        for m in names:
            raw = report.get(split, {}).get(m) or {}
            cases = f"n={raw.get('n')}" if raw.get("n") else ""
            if m in scored:
                print(f"        {m:<32s} {scored[m]:7.3f}  {cases}")
            else:
                why = "not in pinned set" if m in metrics else "no case produced it"
                print(f"        {m:<32s}     --    {why}")
    print(f"    {'SCORE':<12s} {score if score is not None else '--'}"
          f"   (mean of {len(dims)} dimensions)")
    if missing:
        print(f"    MISSING vs pinned set: {missing}")


def main() -> None:
    path = Path(sys.argv[1])
    report = json.loads(path.read_text())
    print(f"report: {path}")
    print(f"model={report.get('model')} n_cases={report.get('n_cases')} n_navi={report.get('n_navi')}")
    for split in ("full", "navi"):
        table(report, split, pinned=None)
    print("\n" + "=" * 70)
    print("  archive-comparable view (same metric set as every node's proxy score)")
    print("=" * 70)
    table(report, "navi", pinned=pinned_metrics())


if __name__ == "__main__":
    main()
