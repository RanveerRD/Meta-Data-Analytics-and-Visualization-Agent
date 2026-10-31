"""Per-question analysis record, persisted as JSON.

One record per question at workspace/runs/<run-id>/record.json. The record
is changed only through its own methods; properties return copies so callers
can never edit the stored state directly. Writes are atomic.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUS_OPEN = "open"
STATUS_PROVEN = "proven"

STATUS_ANSWERED = "answered"
STATUS_ESCALATED = "escalated"

MAX_FINDINGS_PER_ITEM = 3
TOLERANCE = 0.005

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_checklist_items(items: Any) -> list[dict[str, str]]:
    """Validate raw set_checklist input; return cleaned items or raise."""
    if not isinstance(items, list) or not items:
        raise ValueError("set_checklist needs a non-empty list of items")
    seen: set[str] = set()
    cleaned: list[dict[str, str]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"item {i} must be an object with id, question, evidence_needed")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not _ID_RE.match(item_id):
            raise ValueError(f"item {i} has a bad id {item_id!r}: use snake_case")
        if item_id in seen:
            raise ValueError(f"duplicate checklist id: {item_id}")
        seen.add(item_id)
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"checklist item {item_id!r} needs a non-empty question")
        evidence = item.get("evidence_needed")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"checklist item {item_id!r} needs non-empty evidence_needed")
        cleaned.append(
            {
                "id": item_id,
                "question": question.strip(),
                "evidence_needed": evidence.strip(),
                "status": STATUS_OPEN,
            }
        )
    return cleaned


def validate_plan(plan: Any, checklist_ids: list[str]) -> list[dict[str, Any]]:
    """Validate raw update_plan input against the checklist; raise if short."""
    if not isinstance(plan, list) or not plan:
        raise ValueError("update_plan needs a non-empty list of {item_id, steps}")
    required = set(checklist_ids)
    covered: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for i, entry in enumerate(plan):
        if not isinstance(entry, dict):
            raise ValueError(f"plan entry {i} must be an object with item_id and steps")
        item_id = entry.get("item_id")
        if item_id not in required:
            raise ValueError(f"plan entry {i} names unknown checklist item {item_id!r}")
        steps = entry.get("steps")
        if (
            not isinstance(steps, list)
            or not steps
            or any(not isinstance(s, str) or not s.strip() for s in steps)
        ):
            raise ValueError(f"plan entry for {item_id!r} needs a non-empty list of steps")
        covered.add(item_id)
        cleaned.append({"item_id": item_id, "steps": [s.strip() for s in steps]})
    missing = sorted(required - covered)
    if missing:
        raise ValueError(f"plan must cover every checklist item, missing: {missing}")
    return cleaned


class AnalysisRecord:
    """One question's checklist, plan, queries, findings, and audit trail."""

    def __init__(
        self,
        run_id: str,
        question: str,
        path: str | Path,
        state: dict[str, Any] | None = None,
    ) -> None:
        self._run_id = run_id
        self._question = question
        self._path = Path(path)
        self._state = state if state is not None else _blank_state(question)
        self._seq = len(self._state["audit"])

    @classmethod
    def create(cls, question: str, runs_dir: str | Path) -> AnalysisRecord:
        """Start a new record; the caller saves it via a mutating method."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{secrets.token_hex(3)}"
        path = Path(runs_dir) / run_id / "record.json"
        record = cls(run_id, question, path)
        record._append_audit("created", {"question": question})
        return record

    @classmethod
    def load(cls, path: str | Path) -> AnalysisRecord:
        """Read a record back from disk (verification and review)."""
        path = Path(path)
        state = json.loads(path.read_text())
        run_id = path.parent.name
        return cls(run_id, state.get("question", ""), path, state)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def checklist(self) -> list[dict[str, str]]:
        return copy.deepcopy(self._state["checklist"])

    @property
    def plan(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._state["plan"])

    @property
    def queries(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._state["queries"])

    @property
    def findings(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._state["findings"])

    @property
    def rejected(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._state.get("rejected", []))

    @property
    def submission(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._state.get("submission"))

    @property
    def rejections(self) -> int:
        return int(self._state.get("rejections", 0))

    @property
    def question(self) -> str:
        return self._question

    @property
    def audit(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._state["audit"])

    def plan_covers_checklist(self) -> bool:
        """True when a plan exists and covers every checklist item."""
        if not self._state["checklist"] or not self._state["plan"]:
            return False
        required = {c["id"] for c in self._state["checklist"]}
        return required == {p["item_id"] for p in self._state["plan"]}

    def set_checklist(self, items: Any) -> AnalysisRecord:
        """Replace the checklist and plan, keep recorded queries, save.

        Findings for items that no longer exist are dropped; the rejected
        list is append-only history and stays. A prior submission belongs to
        the old checklist, so it and the rejection counter reset.
        """
        self._state["checklist"] = validate_checklist_items(items)
        self._state["plan"] = []
        kept = {c["id"] for c in self._state["checklist"]}
        self._state["findings"] = [
            f for f in self._state["findings"] if f["item_id"] in kept
        ]
        self._state["submission"] = None
        self._state["rejections"] = 0
        self._append_audit("set_checklist", {"items": [c["id"] for c in self._state["checklist"]]})
        self.save()
        return self

    def update_plan(self, plan: Any) -> AnalysisRecord:
        """Store a plan covering every checklist item; save."""
        checklist_ids = [c["id"] for c in self._state["checklist"]]
        self._state["plan"] = validate_plan(plan, checklist_ids)
        self._append_audit("update_plan", {"items": [p["item_id"] for p in self._state["plan"]]})
        self.save()
        return self

    def record_query(self, sql: str, result: dict[str, Any]) -> str:
        """Store a successful query result; return its id (q1, q2, ...)."""
        query_id = f"q{len(self._state['queries']) + 1}"
        self._state["queries"].append(
            {
                "id": query_id,
                "sql": sql,
                "columns": list(result.get("columns", [])),
                "row_count": result.get("row_count", 0),
                "truncated": bool(result.get("truncated", False)),
                "rows": copy.deepcopy(result.get("rows", [])),
                "timestamp": _utc_now_iso(),
            }
        )
        self._append_audit("record_query", {"id": query_id})
        self.save()
        return query_id

    def item_ids(self) -> list[str]:
        """Ids on the current checklist."""
        return [c["id"] for c in self._state["checklist"]]

    def is_proven(self, item_id: str) -> bool:
        """True when the item has at least one accepted finding."""
        return any(
            c["id"] == item_id and c["status"] == STATUS_PROVEN
            for c in self._state["checklist"]
        )

    def _check_finding_refusal(self, item_id: Any, query_id: Any, value: Any) -> str | None:
        """Return the refusal reason, or None when the finding may be stored."""
        known_items = {c["id"] for c in self._state["checklist"]}
        if item_id not in known_items:
            return f"unknown checklist item: {item_id!r}"
        known_queries = {q["id"] for q in self._state["queries"]}
        if query_id not in known_queries:
            return f"unknown query id: {query_id!r}"
        if value is None:
            return "finding needs a value"
        if sum(1 for f in self._state["findings"] if f["item_id"] == item_id) >= MAX_FINDINGS_PER_ITEM:
            return f"checklist item {item_id!r} already has {MAX_FINDINGS_PER_ITEM} findings"
        query = next(q for q in self._state["queries"] if q["id"] == query_id)
        if not _value_in_rows(value, [v for row in query["rows"] for v in row]):
            return f"value {value!r} does not appear in {query_id}'s stored result"
        return None

    def record_finding(
        self, item_id: str, claim: str, query_id: str, value: Any
    ) -> dict[str, Any]:
        """Store a verified finding, or refuse with a reason. Always audited."""
        if not isinstance(claim, str) or not claim.strip():
            return self._refuse_finding(item_id, claim, query_id, value, "finding needs a claim")
        reason = self._check_finding_refusal(item_id, query_id, value)
        if reason is not None:
            return self._refuse_finding(item_id, claim, query_id, value, reason)
        self._state["findings"].append(
            {
                "item_id": item_id,
                "claim": claim.strip(),
                "query_id": query_id,
                "value": value,
                "timestamp": _utc_now_iso(),
            }
        )
        for entry in self._state["checklist"]:
            if entry["id"] == item_id:
                entry["status"] = STATUS_PROVEN
        self._append_audit("record_finding", {"item_id": item_id, "query_id": query_id})
        self.save()
        return {"ok": True, "item_id": item_id, "status": STATUS_PROVEN}

    def _refuse_finding(
        self, item_id: Any, claim: Any, query_id: Any, value: Any, reason: str
    ) -> dict[str, Any]:
        self._state.setdefault("rejected", []).append(
            {
                "tool": "record_finding",
                "item_id": item_id,
                "claim": claim,
                "query_id": query_id,
                "value": value,
                "reason": reason,
                "timestamp": _utc_now_iso(),
            }
        )
        self._append_audit("finding_rejected", {"item_id": item_id, "reason": reason})
        self.save()
        return {"error": reason}

    def submit(
        self,
        answer: str,
        assumptions: Any,
        gaps: Any,
        matched: list[str],
        unmatched: list[str],
        status: str = STATUS_ANSWERED,
    ) -> dict[str, Any]:
        """Store the accepted answer. Called only after all gates pass."""
        submission = {
            "answer": answer,
            "assumptions": list(assumptions),
            "gaps": list(gaps),
            "rejections": self.rejections,
            "status": status,
            "matched": list(matched),
            "unmatched": list(unmatched),
            "timestamp": _utc_now_iso(),
        }
        self._state["submission"] = submission
        self._append_audit(
            "submit_answer",
            {"status": status, "unmatched": list(unmatched)},
        )
        self.save()
        return {"ok": True, **submission}

    def note_rejection(self, reason: str, unmatched: list[str]) -> int:
        """Count a submit_answer rejection; return the new total."""
        self._state["rejections"] = self.rejections + 1
        self._state.setdefault("rejected", []).append(
            {
                "tool": "submit_answer",
                "reason": reason,
                "unmatched": list(unmatched),
                "timestamp": _utc_now_iso(),
            }
        )
        self._append_audit("submit_rejected", {"reason": reason, "unmatched": list(unmatched)})
        self.save()
        return self.rejections

    def save(self) -> Path:
        """Persist atomically: temp file, flush, fsync, rename."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        with open(tmp, "w") as handle:
            json.dump(self._to_dict(), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._path)
        return self._path

    def _append_audit(self, action: str, detail: dict[str, Any]) -> None:
        self._seq += 1
        self._state["audit"].append(
            {"seq": self._seq, "action": action, "detail": detail, "timestamp": _utc_now_iso()}
        )

    def _to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self._run_id,
            "question": self._question,
            "checklist": self._state["checklist"],
            "plan": self._state["plan"],
            "queries": self._state["queries"],
            "findings": self._state["findings"],
            "rejected": self._state.get("rejected", []),
            "submission": self._state.get("submission"),
            "rejections": self._state.get("rejections", 0),
            "audit": self._state["audit"],
        }


def _value_in_rows(value: Any, cells: list[Any]) -> bool:
    """True when a stored cell matches the claimed value.

    Numbers compare with tolerance 0.005; anything else compares exact.
    """
    if isinstance(value, bool):
        return any(cell is value for cell in cells)
    if isinstance(value, (int, float)):
        target = float(value)
        for cell in cells:
            if isinstance(cell, bool):
                continue
            if isinstance(cell, (int, float)) and abs(float(cell) - target) <= TOLERANCE:
                return True
        return False
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        try:
            target = float(text)
        except ValueError:
            return any(cell == value for cell in cells if isinstance(cell, str))
        for cell in cells:
            if isinstance(cell, bool):
                continue
            if isinstance(cell, (int, float)) and abs(float(cell) - target) <= TOLERANCE:
                return True
        return False
    return any(cell == value for cell in cells)


def _blank_state(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "checklist": [],
        "plan": [],
        "queries": [],
        "findings": [],
        "rejected": [],
        "submission": None,
        "rejections": 0,
        "audit": [],
    }
