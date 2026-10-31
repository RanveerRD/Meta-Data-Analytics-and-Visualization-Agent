"""Run exactly one read SQL statement, capped at 200 rows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.backend import DatabaseBackend

NAME = "run_sql"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Run exactly one SQL SELECT statement. At most 200 rows are "
        "returned; the result says whether it was truncated. "
        "Statements that write, change the schema, or take longer than "
        "10 seconds are refused."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A single SQL statement, ending without a second statement.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}


def run(backend: DatabaseBackend, args: dict[str, Any]) -> str:
    return json.dumps(backend.run_sql(args["query"]))
