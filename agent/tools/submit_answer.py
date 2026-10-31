"""Submit the final answer; every number must trace to recorded data."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent.numbers import verify_text
from agent.record import STATUS_ESCALATED

if TYPE_CHECKING:
    from agent.tools import ToolContext

NAME = "submit_answer"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": NAME,
    "description": (
        "Submit the final answer. Rejected unless every checklist item is "
        "proven or named in gaps, and every number in the answer appears in "
        "a recorded query result or the user's question. A rejection lists "
        "the unmatched numbers so the answer can be fixed and resubmitted. "
        "After 3 rejections the answer is accepted but marked escalated."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The full answer text for the user.",
            },
            "assumptions": {
                "type": "array",
                "description": "Assumptions stated briefly.",
                "items": {"type": "string"},
            },
            "gaps": {
                "type": "array",
                "description": "Items the data cannot answer, each naming its checklist item.",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {
                            "type": "string",
                            "description": "Checklist id, or empty for a general gap.",
                        },
                        "note": {
                            "type": "string",
                            "description": "What is missing and what is needed instead.",
                        },
                    },
                    "required": ["item_id", "note"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answer", "assumptions", "gaps"],
        "additionalProperties": False,
    },
    "strict": True,
}

_MISSING_RECORD_ERROR = "submit_answer is refused: call set_checklist first."
_MAX_REJECTIONS = 3


def _validate_gaps(gaps: Any, item_ids: list[str]) -> list[dict[str, str]] | str:
    """Return cleaned gaps, or an error string."""
    if not isinstance(gaps, list):
        return "gaps must be a list of {item_id, note}"
    known = set(item_ids)
    cleaned: list[dict[str, str]] = []
    for i, entry in enumerate(gaps):
        if not isinstance(entry, dict):
            return f"gaps entry {i} must be an object with item_id and note"
        item_id = entry.get("item_id", "")
        if not isinstance(item_id, str):
            return f"gaps entry {i} has a bad item_id"
        if item_id and item_id not in known:
            return f"gaps entry {i} names unknown checklist item {item_id!r}"
        note = entry.get("note")
        if not isinstance(note, str) or not note.strip():
            return f"gaps entry {i} needs a non-empty note"
        cleaned.append({"item_id": item_id, "note": note.strip()})
    return cleaned


def run(ctx: ToolContext, args: dict[str, Any]) -> str:
    record = ctx.record
    if record is None:
        return json.dumps({"error": _MISSING_RECORD_ERROR})
    answer = args.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return json.dumps({"error": "submit_answer needs a non-empty answer"})
    assumptions = args.get("assumptions", [])
    if not isinstance(assumptions, list) or any(
        not isinstance(a, str) for a in assumptions
    ):
        return json.dumps({"error": "assumptions must be a list of strings"})
    gaps = _validate_gaps(args.get("gaps", []), record.item_ids())
    if isinstance(gaps, str):
        return json.dumps({"error": gaps})
    gap_items = {g["item_id"] for g in gaps if g["item_id"]}
    uncovered = [
        item_id
        for item_id in record.item_ids()
        if not record.is_proven(item_id) and item_id not in gap_items
    ]
    if uncovered:
        reason = f"uncovered checklist items: {uncovered}"
        return _reject(record, reason, [], answer)
    cells = [v for q in record.queries for row in q["rows"] for v in row]
    matched, unmatched = verify_text(answer, record.question, cells)
    if unmatched:
        reason = f"unmatched numbers: {unmatched}"
        return _reject(record, reason, unmatched, answer)
    result = record.submit(answer, assumptions, gaps, matched, unmatched)
    return json.dumps(result)


def _reject(
    record, reason: str, unmatched: list[str], answer: str
) -> str:
    """Count the rejection; escalate-accept once the limit is reached."""
    total = record.note_rejection(reason, unmatched)
    if total >= _MAX_REJECTIONS:
        warning = f"UNVERIFIED NUMBERS: {unmatched or reason}"
        result = record.submit(
            answer,
            assumptions=[],
            gaps=[],
            matched=[],
            unmatched=list(unmatched),
            status=STATUS_ESCALATED,
        )
        return json.dumps({**result, "warning": warning})
    return json.dumps({"error": reason, "unmatched": unmatched, "rejections": total})
