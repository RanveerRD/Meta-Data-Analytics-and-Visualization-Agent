"""Regression tests for the agent/tools/ registry.

All tools are exercised through dispatch() with a ToolContext on a
temporary fixture database. The real data/olist.sqlite is never touched.
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.backend import SqliteBackend  # noqa: E402
from agent.tools import TOOL_SCHEMAS, ToolContext, dispatch  # noqa: E402

ITEMS = [
    {"id": "count_items", "question": "How many items?", "evidence_needed": "count of items"},
]

PLAN = [{"item_id": "count_items", "steps": ["count rows in items"]}]


def _make_backend(tmp: str) -> SqliteBackend:
    path = Path(tmp) / "fixture.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL);")
        conn.executemany(
            "INSERT INTO items (name) VALUES (?)", [("a",), ("b",), ("c",)]
        )
        conn.commit()
    finally:
        conn.close()
    return SqliteBackend(path)


def _ctx(backend: SqliteBackend, tmp: str, question: str = "how many?") -> ToolContext:
    return ToolContext(backend=backend, question=question, runs_dir=Path(tmp) / "runs")


def test_registry_lists_all_seven_tools():
    assert sorted(s["name"] for s in TOOL_SCHEMAS) == [
        "get_schema",
        "list_tables",
        "record_finding",
        "run_sql",
        "set_checklist",
        "submit_answer",
        "update_plan",
    ]


def test_list_tables_through_registry():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            assert json.loads(dispatch(_ctx(backend, tmp), "list_tables", {})) == {
                "tables": ["items"]
            }
        finally:
            backend.close()


def test_get_schema_through_registry():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            schema = json.loads(
                dispatch(_ctx(backend, tmp), "get_schema", {"table": "items"})
            )
        finally:
            backend.close()
    assert schema["table"] == "items"
    assert [c["name"] for c in schema["columns"]] == ["id", "name"]


def test_no_record_without_set_checklist():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            dispatch(ctx, "list_tables", {})
            dispatch(ctx, "get_schema", {"table": "items"})
            assert ctx.record is None
            assert not (Path(tmp) / "runs").exists()
        finally:
            backend.close()


def test_run_sql_refused_before_checklist_and_plan():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            refused = json.loads(
                dispatch(ctx, "run_sql", {"query": "SELECT count(*) FROM items;"})
            )
            assert "set_checklist" in refused["error"]
            ok = json.loads(dispatch(ctx, "set_checklist", {"items": ITEMS}))
            assert ok["ok"] is True
            refused = json.loads(
                dispatch(ctx, "run_sql", {"query": "SELECT count(*) FROM items;"})
            )
            assert "update_plan" in refused["error"]
        finally:
            backend.close()


def test_update_plan_rejected_unless_covering():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            dispatch(ctx, "set_checklist", {"items": ITEMS + [
                {"id": "other", "question": "q?", "evidence_needed": "e"}
            ]})
            partial = json.loads(dispatch(ctx, "update_plan", {"plan": PLAN}))
            assert "missing" in partial["error"]
        finally:
            backend.close()


def test_queries_recorded_with_ids():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            dispatch(ctx, "set_checklist", {"items": ITEMS})
            dispatch(ctx, "update_plan", {"plan": PLAN})
            first = json.loads(
                dispatch(ctx, "run_sql", {"query": "SELECT count(*) FROM items;"})
            )
            second = json.loads(
                dispatch(ctx, "run_sql", {"query": "SELECT count(*) FROM items;"})
            )
            assert (first["query_id"], second["query_id"]) == ("q1", "q2")
            assert first["rows"] == [[3]]
            assert [q["id"] for q in ctx.record.queries] == ["q1", "q2"]
            assert ctx.record.path.is_file()
        finally:
            backend.close()


def test_set_checklist_validation():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            ctx = _ctx(backend, tmp)
            for bad in (
                {"items": []},
                {"items": [{"id": "a", "question": "q"}]},
                {"items": [
                    {"id": "a", "question": "q", "evidence_needed": "e"},
                    {"id": "a", "question": "q", "evidence_needed": "e"},
                ]},
            ):
                result = json.loads(dispatch(ctx, "set_checklist", bad))
                assert "error" in result, f"expected error for {bad!r}"
            assert ctx.record is None
        finally:
            backend.close()


def test_unknown_tool():
    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(tmp)
        try:
            result = json.loads(dispatch(_ctx(backend, tmp), "drop_database", {}))
        finally:
            backend.close()
    assert "error" in result


def test_tools_never_import_a_database_driver():
    tools_dir = Path(__file__).resolve().parent.parent / "agent" / "tools"
    for module in (
        "__init__.py",
        "list_tables.py",
        "get_schema.py",
        "run_sql.py",
        "set_checklist.py",
        "update_plan.py",
        "record_finding.py",
        "submit_answer.py",
    ):
        text = (tools_dir / module).read_text()
        assert "sqlite3" not in text, f"{module} touches a database driver"
        assert "psycopg" not in text, f"{module} touches a database driver"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
