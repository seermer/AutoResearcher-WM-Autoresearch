"""THE CONTRACT. Every agent codebase must export exactly these two functions,
each taking one positional argument. The kernel verifies this after every self-edit;
a candidate that breaks it is discarded to the trash branch.

Imports are lazy so the kernel's contract probe stays cheap.
"""
from __future__ import annotations

from kernel.api import (EditSelfContext, EditSelfResult, ImproveRecipeContext,
                        ImproveRecipeResult)


def edit_self(ctx: EditSelfContext) -> EditSelfResult:
    """Rewrite this agent codebase in place under ctx.agents_dir."""
    from .graph import edit_graph
    try:
        out = edit_graph().invoke({"ctx": ctx, "attempts": 0}, {"recursion_limit": 60})
    except Exception as e:  # noqa: BLE001
        return EditSelfResult(ok=False, error=f"{type(e).__name__}: {e}")
    if out.get("error"):
        return EditSelfResult(ok=False, error=out["error"], summary=out.get("change", ""))
    return EditSelfResult(
        summary=out.get("change", ""), hypothesis=out.get("hypothesis", ""),
        changed=[f.strip() for f in (out.get("files") or "").splitlines() if f.strip()],
    )


def improve_recipe(ctx: ImproveRecipeContext) -> ImproveRecipeResult:
    """Improve the WM data recipe, train, and return the LoRA adapter path."""
    from . import memory
    from .graph import recipe_graph
    try:
        out = recipe_graph().invoke({"ctx": ctx, "attempts": 0}, {"recursion_limit": 80})
    except Exception as e:  # noqa: BLE001
        return ImproveRecipeResult(ok=False, error=f"{type(e).__name__}: {e}")
    if out.get("error") or not out.get("adapter"):
        return ImproveRecipeResult(ok=False, error=out.get("error") or "no adapter produced",
                                   summary=out.get("plan", ""))
    recipe = {"plan": out.get("plan", ""), "mechanism": out.get("mechanism", ""),
              "weaknesses": out.get("weaknesses", ""), "sources": (out.get("sources") or "")[:1500],
              "actions": out.get("actions", "")}
    memory.append(ctx.memory_dir, "recipes", f"{ctx.node_id}: {out.get('plan','')[:220]}")
    return ImproveRecipeResult(
        lora_path=out["adapter"], recipe=recipe,
        train={"config": out.get("config", ""), "steps": out.get("steps", "")},
        summary=out.get("plan", ""),
    )
