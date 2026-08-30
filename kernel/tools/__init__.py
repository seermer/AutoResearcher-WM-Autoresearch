"""Core tool set. Deliberately minimal: files, online search, shell.

Everything else lives in the editable agent layer under roles/tools_ext/ and is
populated by the meta agents over time.
"""
from . import context, files, plan, shell, web

CORE_TOOLS = [*files.TOOLS, *web.TOOLS, *shell.TOOLS, *plan.TOOLS]

__all__ = ["CORE_TOOLS", "context", "files", "plan", "web", "shell"]
