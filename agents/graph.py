"""The multi-agent loop, as an explicit LangGraph. This file is editable by the
meta agent: adding a node, an edge or a retry path is a legitimate self-edit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from kernel.api import EditSelfContext, ImproveRecipeContext

from . import memory, roles

MAX_ENGINEER_RETRIES = 2
MAX_META_RETRIES = 1


def _last(a, b):
    return b if b is not None else a


# ─────────────────────────── improve_recipe ───────────────────────────

class RecipeState(TypedDict, total=False):
    ctx: ImproveRecipeContext
    weaknesses: Annotated[str, _last]
    already_tried: Annotated[str, _last]
    plan: Annotated[str, _last]
    mechanism: Annotated[str, _last]
    needs_data: Annotated[bool, _last]
    sources: Annotated[str, _last]
    actions: Annotated[str, _last]
    config: Annotated[str, _last]
    steps: Annotated[str, _last]
    adapter: Annotated[str, _last]
    attempts: Annotated[int, _last]
    error: Annotated[str, _last]


def n_analyze(state: RecipeState) -> dict:
    c = state["ctx"]
    out = roles.run("analyst", (
        f"Node {c.node_id}. The parent's WBench report is at {c.out_dir}/parent_report.json "
        f"(absent if this is the root). The archive history is at {c.history_path}. "
        f"Read them and diagnose."), memory_dir=c.memory_dir)
    return {"weaknesses": roles.field(out, "WEAKNESSES", out),
            "already_tried": roles.field(out, "ALREADY_TRIED")}


def n_plan(state: RecipeState) -> dict:
    c = state["ctx"]
    out = roles.run("planner", (
        f"Weaknesses:\n{state.get('weaknesses','')}\n\n"
        f"Already tried in this archive:\n{state.get('already_tried','none')}\n\n"
        f"The Sana training codebase is at {c.sana_dir}; the baseline config is "
        f"configs/sana_wm/stage1/sana_wm_stage1_lora_base.yaml. Inspect the data layout "
        f"under {c.sana_dir}/data before deciding. Choose one intervention."),
        memory_dir=c.memory_dir)
    return {"plan": roles.field(out, "PLAN", out),
            "mechanism": roles.field(out, "MECHANISM"),
            "needs_data": roles.field(out, "NEEDS_EXTERNAL_DATA").lower().startswith("y")}


def n_scout(state: RecipeState) -> dict:
    c = state["ctx"]
    out = roles.run("scout", (
        f"The plan is:\n{state.get('plan','')}\n\n"
        f"Find and verify data that makes this possible. Disk budget: {c.disk_gb:.0f} GB. "
        f"The loader layout is described in your instructions; say how each candidate maps to it."),
        memory_dir=c.memory_dir)
    if src := roles.field(out, "RECOMMENDATION"):
        memory.append(c.memory_dir, "sources", src.replace("\n", " ")[:300])
    return {"sources": out}


def n_engineer(state: RecipeState) -> dict:
    c = state["ctx"]
    attempt = state.get("attempts", 0) + 1
    retry = ""
    if state.get("error"):
        retry = f"\n\nYour previous attempt failed: {state['error']}\nFix the cause; do not repeat it."
    out = roles.run("engineer", (
        f"Node {c.node_id}, attempt {attempt}.\n"
        f"PLAN:\n{state.get('plan','')}\n\n"
        f"SOURCES:\n{state.get('sources','(none needed)')}\n\n"
        f"Paths: sana={c.sana_dir}  datastore={c.datastore_dir}  out={c.out_dir}  "
        f"logs={c.logs_dir}  baseline_lora={c.baseline_lora}\n"
        f"GPUs={c.gpus}. Wall-clock budget {c.budget_seconds//3600}h, disk {c.disk_gb:.0f} GB.\n"
        f"Implement the plan, train, and report the adapter path.{retry}"),
        memory_dir=c.memory_dir, steps=roles.STEP_LIMIT + 20)
    return {"actions": roles.field(out, "ACTIONS", out),
            "config": roles.field(out, "CONFIG"),
            "steps": roles.field(out, "STEPS"),
            "adapter": roles.field(out, "ADAPTER"),
            "attempts": attempt}


def n_verify(state: RecipeState) -> dict:
    """Pure check, no LLM: did training actually produce a loadable adapter?"""
    raw = (state.get("adapter") or "").strip().split()
    path = Path(raw[0]) if raw and raw[0].upper() != "NONE" else None
    if path is None:
        return {"error": "engineer reported no adapter"}
    if not path.exists():
        return {"error": f"adapter path does not exist: {path}"}
    if not (path / "adapter_model.safetensors").exists():
        return {"error": f"no adapter_model.safetensors in {path}"}
    return {"error": "", "adapter": str(path.resolve())}


def _after_plan(state: RecipeState) -> str:
    return "scout" if state.get("needs_data") else "engineer"


def _after_verify(state: RecipeState) -> str:
    if not state.get("error"):
        return END
    return "engineer" if state.get("attempts", 0) < MAX_ENGINEER_RETRIES else END


def recipe_graph():
    g = StateGraph(RecipeState)
    g.add_node("analyze", n_analyze)
    g.add_node("plan", n_plan)
    g.add_node("scout", n_scout)
    g.add_node("engineer", n_engineer)
    g.add_node("verify", n_verify)
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "plan")
    g.add_conditional_edges("plan", _after_plan, {"scout": "scout", "engineer": "engineer"})
    g.add_edge("scout", "engineer")
    g.add_edge("engineer", "verify")
    g.add_conditional_edges("verify", _after_verify, {"engineer": "engineer", END: END})
    return g.compile()


# ───────────────────────────── edit_self ─────────────────────────────

class EditState(TypedDict, total=False):
    ctx: EditSelfContext
    review: Annotated[str, _last]
    change: Annotated[str, _last]
    hypothesis: Annotated[str, _last]
    files: Annotated[str, _last]
    attempts: Annotated[int, _last]
    error: Annotated[str, _last]


def n_review(state: EditState) -> dict:
    c = state["ctx"]
    out = roles.run("analyst", (
        f"Node {c.node_id} is about to rewrite its own agent code. Its parent scored "
        f"{c.parent_score}. Read the archive history at {c.history_path} and this node's "
        f"logs at {c.logs_dir}. Do not diagnose the world model here — diagnose the AGENTS: "
        f"where did the agent process itself waste effort, repeat a mistake, or fail to "
        f"execute its own plan?"), memory_dir=c.memory_dir)
    return {"review": out}


def n_meta(state: EditState) -> dict:
    c = state["ctx"]
    retry = f"\n\nYour previous edit was rejected: {state['error']}\nFix it." if state.get("error") else ""
    out = roles.run("meta", (
        f"Node {c.node_id}. Agent code to edit: {c.agents_dir}/agents (writable).\n"
        f"Process review:\n{state.get('review','')}\n\n"
        f"History: {c.history_path}. Parent metrics: {json.dumps(c.parent_metrics)[:1200]}\n"
        f"Make ONE coherent improvement to the agent layer.{retry}"),
        memory_dir=c.memory_dir, steps=roles.STEP_LIMIT + 10)
    return {"change": roles.field(out, "CHANGE", out),
            "hypothesis": roles.field(out, "HYPOTHESIS"),
            "files": roles.field(out, "FILES"),
            "attempts": state.get("attempts", 0) + 1}


def n_selfcheck(state: EditState) -> dict:
    """Local pre-check. The kernel re-verifies authoritatively afterwards."""
    from kernel.contract import check_compiles, check_entrypoints
    c = state["ctx"]
    for check in (check_compiles, check_entrypoints):
        res = check(Path(c.agents_dir))
        if not res:
            return {"error": res.reason}
    return {"error": ""}


def _after_selfcheck(state: EditState) -> str:
    if not state.get("error"):
        return END
    return "meta" if state.get("attempts", 0) < MAX_META_RETRIES + 1 else END


def edit_graph():
    g = StateGraph(EditState)
    g.add_node("review", n_review)
    g.add_node("meta", n_meta)
    g.add_node("selfcheck", n_selfcheck)
    g.add_edge(START, "review")
    g.add_edge("review", "meta")
    g.add_edge("meta", "selfcheck")
    g.add_conditional_edges("selfcheck", _after_selfcheck, {"meta": "meta", END: END})
    return g.compile()
