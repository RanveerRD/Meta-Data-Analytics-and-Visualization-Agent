"""Describe one table: columns, types, and sample values."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.backend import DatabaseBackend

NAME = "get_schema"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Describe a table: column names, declared types, and 3 sample "
        "values per column. The table name must come from list_tables."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "table": {
                "type": "string",
                "description": "Exact table name from list_tables.",
            }
        },
        "required": ["table"],
        "additionalProperties": False,
    },
    "strict": True,
}


def run(backend: DatabaseBackend, args: dict[str, Any]) -> str:
    return json.dumps(backend.get_schema(args["table"]))
