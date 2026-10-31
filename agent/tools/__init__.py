"""Tool registry: schemas for the API call, dispatch of function calls.

Adding a tool later means adding one file in this package and one line in
_TOOL_MODULES below. Tools never import a database driver; they only call
the injected backend and record through the context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.record import AnalysisRecord

if TYPE_CHECKING:
    from agent.backend import DatabaseBackend


@dataclass
class ToolContext:
    """What tools may touch: the database backend and this turn's record.

    The record starts as None and is created lazily by the first successful
    set_checklist call, so a turn with no tool calls stores nothing.
    """

    backend: DatabaseBackend
    question: str = ""
    runs_dir: Path = field(default_factory=lambda: Path("workspace") / "runs")
    record: AnalysisRecord | None = None


def _tool_modules() -> tuple:
    from agent.tools import get_schema, list_tables, run_sql, set_checklist, update_plan

    return (list_tables, get_schema, set_checklist, update_plan, run_sql)


_TOOL_MODULES = _tool_modules()

TOOL_SCHEMAS: list[dict[str, Any]] = [m.SCHEMA for m in _TOOL_MODULES]
_HANDLERS = {m.NAME: m.run for m in _TOOL_MODULES}


def dispatch(ctx: ToolContext, name: str, args: dict[str, Any]) -> str:
    """Run the named tool and return its JSON result string."""
    try:
        handler = _HANDLERS[name]
    except KeyError:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        return handler(ctx, args)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
