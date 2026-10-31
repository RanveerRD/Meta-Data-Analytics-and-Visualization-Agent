"""Regression tests for the agent/tools/ registry.

All three tools are exercised through dispatch(), on a temporary fixture
database. The real data/olist.sqlite is never touched.
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.backend import SqliteBackend  # noqa: E402
from agent.tools import TOOL_SCHEMAS, dispatch  # noqa: E402


def _make_backend(tmp: str) -> tuple[SqliteBackend, Path]:
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
    return SqliteBackend(path), path


def test_registry_lists_all_three_tools():
    assert sorted(s["name"] for s in TOOL_SCHEMAS) == [
        "get_schema",
        "list_tables",
        "run_sql",
    ]
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert schema["description"]


def test_list_tables_through_registry():
    with tempfile.TemporaryDirectory() as tmp:
        backend, _ = _make_backend(tmp)
        try:
            assert json.loads(dispatch(backend, "list_tables", {})) == {"tables": ["items"]}
        finally:
            backend.close()


def test_get_schema_through_registry():
    with tempfile.TemporaryDirectory() as tmp:
        backend, _ = _make_backend(tmp)
        try:
            schema = json.loads(dispatch(backend, "get_schema", {"table": "items"}))
        finally:
            backend.close()
    assert schema["table"] == "items"
    assert [c["name"] for c in schema["columns"]] == ["id", "name"]


def test_run_sql_through_registry():
    with tempfile.TemporaryDirectory() as tmp:
        backend, _ = _make_backend(tmp)
        try:
            ok = json.loads(
                dispatch(backend, "run_sql", {"query": "SELECT count(*) FROM items;"})
            )
            assert ok["rows"] == [[3]]
            assert ok["truncated"] is False
            denied = json.loads(
                dispatch(backend, "run_sql", {"query": "DELETE FROM items;"})
            )
            assert "error" in denied
        finally:
            backend.close()


def test_unknown_tool():
    with tempfile.TemporaryDirectory() as tmp:
        backend, _ = _make_backend(tmp)
        try:
            result = json.loads(dispatch(backend, "drop_database", {}))
        finally:
            backend.close()
    assert "error" in result


def test_tools_never_import_a_database_driver():
    tools_dir = Path(__file__).resolve().parent.parent / "agent" / "tools"
    for module in ("__init__.py", "list_tables.py", "get_schema.py", "run_sql.py"):
        text = (tools_dir / module).read_text()
        assert "sqlite3" not in text, f"{module} touches a database driver"
        assert "psycopg" not in text, f"{module} touches a database driver"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"all {len(tests)} tests passed")
