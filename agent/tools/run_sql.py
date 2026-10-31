"""Run exactly one read SQL statement, capped at 200 rows.

Refused until a checklist and a covering plan exist. Every successful call
is recorded in the turn's AnalysisRecord as q1, q2, ...
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "run_sql"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Run exactly one SQL SELECT statement. At most 200 rows are "
        "returned; the result says whether it was truncated and names its "
        "query id (q1, q2, ...). Refused until set_checklist and a covering "
        "update_plan exist. Statements that write, change the schema, or "
        "take longer than 10 seconds are refused."
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

_GATING_ERROR = (
    "run_sql is refused: call set_checklist first, then update_plan "
    "covering every checklist item."
)


def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    record = ctx.record
    if record is None or not record.plan_covers_checklist():
        return json.dumps({"error": _GATING_ERROR})
    try:
        result = ctx.backend.run_sql(args["query"])
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    query_id = record.record_query(args["query"], result)
    return json.dumps({**result, "query_id": query_id})
