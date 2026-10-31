"""Regression tests for findings, submit_answer, and the submit loop.

Run with the project environment:

    .venv/bin/python -m pytest tests/test_verification.py -v
    .venv/bin/python tests/test_verification.py

No API calls: the loop tests use a stubbed client. Temporary directories
only; the real data/olist.sqlite is never touched.
"""

import json
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.backend import SqliteBackend  # noqa: E402
from agent.loop import answer_question  # noqa: E402
from agent.numbers import extract_numbers, verify_text  # noqa: E402
from agent.tools import TOOL_SCHEMAS, ToolContext, dispatch  # noqa: E402

ITEMS = [
    {"id": "total_revenue", "question": "Total revenue?", "evidence_needed": "sum of revenue"},
    {"id": "top_region", "question": "Top region?", "evidence_needed": "revenue by region"},
]

PLAN = [
    {"item_id": "total_revenue", "steps": ["sum revenue"]},
    {"item_id": "top_region", "steps": ["group by region"]},
]


def _make_backend(tmp: str) -> SqliteBackend:
    path = Path(tmp) / "fixture.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE sales (region TEXT NOT NULL, revenue REAL NOT NULL);")
        conn.executemany(
            "INSERT INTO sales (region, revenue) VALUES (?, ?)",
            [("SP", 498440.43), ("RJ", 12.345)],
        )
        conn.commit()
    finally:
        conn.close()
    return SqliteBackend(path)


def _ctx(backend: SqliteBackend, tmp: str, question: str = "q?") -> ToolContext:
    return ToolContext(backend=backend, question=question, runs_dir=Path(tmp) / "runs")


def _flowed_ctx(backend: SqliteBackend, tmp: str) -> ToolContext:
    """Context with checklist, plan, and one recorded query (q1)."""
    ctx = _ctx(backend, tmp)
    assert json.loads(dispatch(ctx, "set_checklist", {"items": ITEMS}))["ok"] is True
    assert json.loads(dispatch(ctx, "update_plan", {"plan": PLAN}))["ok"] is True
    result = json.loads(
        dispatch(ctx, "run_sql", {"query": "SELECT region, revenue FROM sales ORDER BY region;"})
    )
    assert result["query_id"] == "q1"
    return ctx


def test_finding_refused_unknown_query_id():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            result = json.loads(
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": "total_revenue", "claim": "c", "query_id": "q9", "value": 1},
                )
            )
            assert "unknown query id" in result["error"]
            assert ctx.record.rejected and "unknown query id" in ctx.record.rejected[-1]["reason"]
        finally:
            backend.close()


def test_finding_refused_value_not_in_result():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            result = json.loads(
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": "total_revenue", "claim": "c", "query_id": "q1", "value": 777},
                )
            )
            assert "does not appear" in result["error"]
            assert not ctx.record.is_proven("total_revenue")
        finally:
            backend.close()


def test_finding_accepted_within_tolerance_and_max_three():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            ok = json.loads(
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": "top_region", "claim": "RJ is 12.345", "query_id": "q1", "value": 12.347},
                )
            )
            assert ok.get("ok") is True
            assert ctx.record.is_proven("top_region")
            for i in range(2):
                ok = json.loads(
                    dispatch(
                        ctx,
                        "record_finding",
                        {"item_id": "top_region", "claim": f"more {i}", "query_id": "q1", "value": "RJ"},
                    )
                )
                assert ok.get("ok") is True
            fourth = json.loads(
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": "top_region", "claim": "fourth", "query_id": "q1", "value": "RJ"},
                )
            )
            assert "already has 3 findings" in fourth["error"]
        finally:
            backend.close()


def test_repeat_set_checklist_keeps_queries_drops_stale_findings():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            dispatch(
                ctx,
                "record_finding",
                {"item_id": "top_region", "claim": "RJ", "query_id": "q1", "value": "RJ"},
            )
            assert ctx.record.is_proven("top_region")
            dispatch(ctx, "set_checklist", {"items": [ITEMS[0]]})
            assert [q["id"] for q in ctx.record.queries] == ["q1"]
            assert ctx.record.findings == []
            assert ctx.record.plan == []
            assert [c["id"] for c in ctx.record.checklist] == ["total_revenue"]
            assert not ctx.record.is_proven("total_revenue")
        finally:
            backend.close()


def test_submit_refused_uncovered_item():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            dispatch(
                ctx,
                "record_finding",
                {"item_id": "total_revenue", "claim": "sum", "query_id": "q1", "value": 498440.43},
            )
            result = json.loads(
                dispatch(
                    ctx,
                    "submit_answer",
                    {"answer": "Total is 498440.43.", "assumptions": [], "gaps": []},
                )
            )
            assert "uncovered" in result["error"] and "top_region" in result["error"]
            assert ctx.record.submission is None
        finally:
            backend.close()


def test_submit_refused_unmatched_number_then_accepted():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            for item_id, value in (("total_revenue", 498440.43), ("top_region", "RJ")):
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": item_id, "claim": "c", "query_id": "q1", "value": value},
                )
            refused = json.loads(
                dispatch(
                    ctx,
                    "submit_answer",
                    {"answer": "Total is 498440.43 plus 7 bonus.", "assumptions": [], "gaps": []},
                )
            )
            assert refused["unmatched"] == ["7"]
            fixed = json.loads(
                dispatch(
                    ctx,
                    "submit_answer",
                    {
                        "answer": "Total is 498,440.",
                        "assumptions": ["All statuses included."],
                        "gaps": [],
                    },
                )
            )
            assert fixed["ok"] is True
            assert fixed["status"] == "answered"
            assert fixed["rejections"] == 1
            assert fixed["unmatched"] == []
            assert ctx.record.submission["answer"] == "Total is 498,440."
        finally:
            backend.close()


def test_submit_escalated_after_three_rejections():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            for item_id, value in (("total_revenue", 498440.43), ("top_region", "RJ")):
                dispatch(
                    ctx,
                    "record_finding",
                    {"item_id": item_id, "claim": "c", "query_id": "q1", "value": value},
                )
            last = {}
            submitted_assumptions = ["All statuses included."]
            submitted_gaps = [{"item_id": "", "note": "Date range unclear."}]
            for _ in range(3):
                last = json.loads(
                    dispatch(
                        ctx,
                        "submit_answer",
                        {
                            "answer": "Total is 9.",
                            "assumptions": submitted_assumptions,
                            "gaps": submitted_gaps,
                        },
                    )
                )
            assert last["ok"] is True
            assert last["status"] == "escalated"
            assert last["warning"].startswith("UNVERIFIED NUMBERS:")
            assert "9" in last["warning"]
            assert ctx.record.submission["status"] == "escalated"
            assert ctx.record.rejections == 3
            assert ctx.record.submission["assumptions"] == submitted_assumptions
            assert ctx.record.submission["gaps"] == submitted_gaps
        finally:
            backend.close()


def test_years_and_list_numbering_ignored():
    assert extract_numbers("1. First\n2. Second") == []
    matched, unmatched = verify_text(
        "In 2024:\n1. Total\nThe total is 5.", "How many?", [5]
    )
    assert matched == ["5"]
    assert unmatched == []


def test_iso_dates_and_times_produce_no_numbers():
    assert extract_numbers("from 2017-01-01 to 2017-12-31") == []
    assert extract_numbers("in 2017-01") == []
    assert extract_numbers("at 10:30:00") == []
    assert extract_numbers("at 10:30") == []
    matched, unmatched = verify_text(
        "From 2017-01-01 to 2017-12-31.", "Over what period?", [99441]
    )
    assert matched == []
    assert unmatched == []


def test_hyphen_between_digits_is_a_range_separator():
    assert extract_numbers("2017-2018") == ["2017", "2018"]
    assert extract_numbers("10-20") == ["10", "20"]
    assert extract_numbers("-5.2") == ["-5.2"]
    assert extract_numbers("(-5.2)") == ["-5.2"]
    matched, unmatched = verify_text(
        "Total is 5 from 2017-01-01 to 2017-12-31.", "How many?", [5]
    )
    assert matched == ["5"]
    assert unmatched == []


def test_gap_named_item_passes_without_proof():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _flowed_ctx(backend, tmp)
            dispatch(
                ctx,
                "record_finding",
                {"item_id": "total_revenue", "claim": "sum", "query_id": "q1", "value": 498440.43},
            )
            result = json.loads(
                dispatch(
                    ctx,
                    "submit_answer",
                    {
                        "answer": "Total is 498440.43.",
                        "assumptions": [],
                        "gaps": [{"item_id": "top_region", "note": "Need cost data."}],
                    },
                )
            )
            assert result["ok"] is True
        finally:
            backend.close()


def _message(text: str):
    return types.SimpleNamespace(
        type="message",
        content=[types.SimpleNamespace(type="output_text", text=text)],
    )


def _call(call_id: str, name: str, args: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        type="function_call", call_id=call_id, name=name, arguments=json.dumps(args)
    )


def _response(*items) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        output=list(items),
        usage=types.SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
    )


class _StubClient:
    def __init__(self, script):
        self._script = list(script)
        self.responses = self
        self.calls = 0

    def create(self, **kwargs):
        item = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return item


def _stub_backend(tmp: str) -> SqliteBackend:
    path = Path(tmp) / "stub.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY);")
        conn.executemany("INSERT INTO items (id) VALUES (?)", [(1,), (2,), (3,)])
        conn.commit()
    finally:
        conn.close()
    return SqliteBackend(path)


def test_loop_shows_accepted_submit_text():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _response(_call("c1", "set_checklist", {"items": [
                    {"id": "n", "question": "How many?", "evidence_needed": "count"}]})),
                _response(_call("c2", "update_plan", {"plan": [
                    {"item_id": "n", "steps": ["count"]}]})),
                _response(_call("c3", "run_sql", {"query": "SELECT count(*) FROM items;"})),
                _response(_call("c4", "record_finding", {
                    "item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3})),
                _response(_call("c5", "submit_answer", {
                    "answer": "There are 3 items.",
                    "assumptions": ["All rows counted."],
                    "gaps": []})),
                _response(_message("WRONG TEXT")),
            ]
            result = answer_question(
                _StubClient(script), "m", "instr", backend, [], "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            assert result.text == "There are 3 items."
            assert result.verification == "verification: 1 numbers matched, 0 unmatched"
            assert result.record_path is not None
        finally:
            backend.close()


def test_loop_reminds_when_plain_text_without_submit():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _response(
                    _call("c1", "set_checklist", {"items": [
                        {"id": "n", "question": "How many?", "evidence_needed": "count"}]}),
                    _call("c2", "update_plan", {"plan": [
                        {"item_id": "n", "steps": ["count"]}]})),
                _response(_message("The answer is 3 items.")),
                _response(_call("c3", "run_sql", {"query": "SELECT count(*) FROM items;"})),
                _response(_call("c4", "record_finding", {
                    "item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3})),
                _response(_call("c5", "submit_answer", {
                    "answer": "There are 3 items.", "assumptions": [], "gaps": []})),
                _response(_message("done")),
            ]
            history: list = []
            result = answer_question(
                _StubClient(script), "m", "instr", backend, history, "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            assert result.text == "There are 3 items."
            assert any(
                m.get("content", "").startswith("You replied in plain text")
                for m in history
                if m.get("role") == "user"
            )
        finally:
            backend.close()


def test_loop_plain_text_without_record():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [_response(_message("Yes, all statuses."))]
            result = answer_question(
                _StubClient(script), "m", "instr", backend, [], "yes?",
                runs_dir=Path(tmp) / "runs",
            )
            assert result.text == "Yes, all statuses."
            assert result.record_path is None
            assert result.verification is None
        finally:
            backend.close()




def test_acceptance_on_final_round_returns_answer_not_cap():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _response(
                    _call("c1", "set_checklist", {"items": [
                        {"id": "n", "question": "How many?", "evidence_needed": "count"}]}),
                    _call("c2", "update_plan", {"plan": [
                        {"item_id": "n", "steps": ["count"]}]})),
            ]
            for i in range(1, 14):
                script.append(_response(_call(f"g{i}", "get_schema", {"table": "items"})))
            script += [
                _response(
                    _call("c15a", "run_sql", {"query": "SELECT count(*) FROM items;"}),
                    _call("c15b", "record_finding", {
                        "item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3}),
                ),
                _response(_call("c16", "submit_answer", {
                    "answer": "There are 3 items.",
                    "assumptions": ["All rows counted."],
                    "gaps": []})),
            ]
            assert len(script) == 16
            client = _StubClient(script)
            result = answer_question(
                client, "m", "instr", backend, [], "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            assert result.text == "There are 3 items."
            assert "Reached the" not in result.text
            assert client.calls == 16
            assert result.verification == "verification: 1 numbers matched, 0 unmatched"
        finally:
            backend.close()


def test_no_calls_after_early_acceptance():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _stub_backend(tmp)
        try:
            script = [
                _response(_call("c1", "set_checklist", {"items": [{"id": "n", "question": "How many?", "evidence_needed": "count"}]})),
                _response(_call("c2", "update_plan", {"plan": [{"item_id": "n", "steps": ["count"]}]})),
                _response(_call("c3", "get_schema", {"table": "items"})),
                _response(_call("c4", "run_sql", {"query": "SELECT count(*) FROM items;"})),
                _response(_call("c5", "record_finding", {"item_id": "n", "claim": "3 items", "query_id": "q1", "value": 3})),
                _response(_call("c6", "submit_answer", {"answer": "There are 3 items.", "assumptions": ["All rows counted."], "gaps": []})),
            ]
            assert len(script) == 6
            client = _StubClient(script)
            result = answer_question(
                client, "m", "instr", backend, [], "How many?",
                runs_dir=Path(tmp) / "runs",
            )
            assert result.text == "There are 3 items."
            assert client.calls == 6
        finally:
            backend.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
