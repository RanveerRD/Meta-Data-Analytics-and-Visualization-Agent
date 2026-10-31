"""Tool registry: schemas for the API call, dispatch of function calls.

Adding a tool later means adding one file in this package and one line in
_TOOL_MODULES below. Tools never import a database driver; they only call
the injected backend.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent.tools import get_schema, list_tables, run_sql

if TYPE_CHECKING:
    from agent.backend import DatabaseBackend

_TOOL_MODULES = (list_tables, get_schema, run_sql)

TOOL_SCHEMAS: list[dict[str, Any]] = [m.SCHEMA for m in _TOOL_MODULES]
_HANDLERS = {m.NAME: m.run for m in _TOOL_MODULES}


def dispatch(backend: DatabaseBackend, name: str, args: dict[str, Any]) -> str:
    """Run the named tool and return its JSON result string."""
    try:
        handler = _HANDLERS[name]
    except KeyError:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        return handler(backend, args)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
