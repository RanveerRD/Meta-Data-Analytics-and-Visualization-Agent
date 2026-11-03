"""BI connectors and /publish (no API calls, no network)."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.__main__ import _handle_command, _render_all_outcome  # noqa: E402
from agent.bi import CONNECTORS  # noqa: E402
from agent.bi.datasette import DatasetteConnector  # noqa: E402
from agent.bi.metabase import MetabaseConnector  # noqa: E402
from agent.datasets import REGISTRY  # noqa: E402
from agent.record import AnalysisRecord  # noqa: E402
from agent.tools.make_chart import chart_candidates  # noqa: E402

EXAMPLE_SQL = (
    "   select strftime('%Y-%m', order_purchase_timestamp) as month,"
    " count(*) as orders\r\n   from orders group by 1 order by 1"
)
EXAMPLE_URL = (
    "http://127.0.0.1:8001/olist?sql="
    "+++select+strftime%28%27%25Y-%25m%27%2C+order_purchase_timestamp%29"
    "+as+month%2C+count%28*%29+as+orders%0D%0A"
    "+++from+orders+group+by+1+order+by+1"
    "#g.mark=line&g.x_column=month&g.x_type=ordinal"
    "&g.y_column=orders&g.y_type=quantitative"
)


def test_datasette_matches_working_example():
    result = DatasetteConnector("olist").publish(
        EXAMPLE_SQL, "orders by month", "line", "month", "orders")
    assert result.url == EXAMPLE_URL


def test_datasette_bar_mark_and_round_trip():
    sql = "select a, b from t"
    result = DatasetteConnector("db").publish(sql, "t", "bar", "a", "b")
    assert "#g.mark=bar&" in result.url
    assert parse_qs(urlparse(result.url).query)["sql"] == [sql]


def test_datasette_column_encoding():
    result = DatasetteConnector("db").publish(
        "select 1", "t", "bar", "my col", "my_col")
    assert "g.x_column=my%20col" in result.url
    assert "g.y_column=my_col" in result.url


def test_datasette_overlong_sql_refused():
    with pytest.raises(ValueError, match="too long"):
        DatasetteConnector("db").publish("select " + "x" * 8000, "t", "bar", "a", "b")


def test_datasette_url_override_and_scheme(monkeypatch):
    monkeypatch.setenv("DATASETTE_URL", "https://bi.example.com:8002/")
    result = DatasetteConnector("db").publish("select 1", "t", "bar", "a", "b")
    assert result.url.startswith("https://bi.example.com:8002/db?sql=")
    monkeypatch.setenv("DATASETTE_URL", "ftp://x/y")
    with pytest.raises(ValueError, match="http"):
        DatasetteConnector("db").publish("select 1", "t", "bar", "a", "b")


def test_metabase_stub():
    with pytest.raises(RuntimeError, match="not configured"):
        MetabaseConnector("adventureworks").publish("select 1", "t", "bar", "a", "b")


def test_connector_registry():
    assert CONNECTORS["datasette"] is DatasetteConnector
    assert CONNECTORS["metabase"] is MetabaseConnector


def _record(tmp_path: Path) -> AnalysisRecord:
    record = AnalysisRecord.create("bi?", tmp_path / "runs")
    record.set_checklist(
        [{"id": "rev", "question": "Revenue?", "evidence_needed": "sum"}]
    )
    record.plan_action([{"item_id": "rev", "steps": ["sum"]}])
    qid = record.record_query(
        "SELECT region, revenue FROM t",
        {"columns": ["region", "revenue"],
         "rows": [["SP", 10.0], ["RJ", 20.0]],
         "row_count": 2, "truncated": False},
    )
    record.record_finding("rev", "big", qid, 10.0)
    return record


def _pending(record: AnalysisRecord, sqlite_path: str = "/tmp/my db.sqlite"):
    return {"record_abs": record.path, "sqlite_path": sqlite_path,
            "candidates": chart_candidates(record)}


def test_sql_is_recorded_bytes(monkeypatch):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        record = _record(Path(tmp))
        (candidate,) = chart_candidates(record)
        monkeypatch.delenv("DATASETTE_URL", raising=False)
        result = DatasetteConnector("olist").publish(
            candidate["sql"], "t", candidate["type"],
            candidate["x"], candidate["y"])
        assert parse_qs(urlparse(result.url).query)["sql"] == [
            "SELECT region, revenue FROM t"]


def test_publish_end_to_end_olist(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("DATASETTE_URL", raising=False)
    record = _record(tmp_path)
    (candidate,) = chart_candidates(record)
    expected = DatasetteConnector("olist").publish(
        candidate["sql"], "revenue by region", candidate["type"],
        candidate["x"], candidate["y"])
    assert _handle_command("/publish 1", dataset="olist",
                           pending_charts=_pending(record)) is True
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == expected.url
    assert lines[1] == (
        "Start Datasette first: datasette serve --immutable "
        + shlex.quote("/tmp/my db.sqlite")
    )
    assert lines[2] == expected.note
    fresh = AnalysisRecord.load(record.path)
    assert {"connector": "datasette", "query_id": "q1", "url": expected.url} in [
        {k: e["detail"].get(k) for k in ("connector", "query_id", "url")}
        for e in fresh.audit if e["action"] == "publish"
    ]


def test_publish_out_of_range_and_usage(tmp_path, capsys):
    record = _record(tmp_path)
    pending = _pending(record)
    assert _handle_command("/publish 9", dataset="olist",
                           pending_charts=pending) is True
    assert "choose 1 to 1" in capsys.readouterr().out
    assert _handle_command("/publish abc", dataset="olist",
                           pending_charts=pending) is True
    assert "usage: /publish [n]" in capsys.readouterr().out


def test_publish_empty_and_unconfigured(tmp_path, capsys, monkeypatch):
    assert _handle_command("/publish", dataset="olist",
                           pending_charts=None) is True
    assert "No chart available" in capsys.readouterr().out
    record = _record(tmp_path)
    monkeypatch.setitem(
        REGISTRY, "plain",
        {"description": "d", "label": "P", "prompt": "p> ",
         "make_backend": lambda: None},
    )
    assert _handle_command("/publish", dataset="plain",
                           pending_charts=_pending(record)) is True
    assert "No BI connector configured for plain" in capsys.readouterr().out


def test_all_mode_prints_nothing(capsys):
    outcome = {"routing": ["olist"], "workers": [], "failed": [],
               "merged": "", "verification": {"status": "no-answers", "unmatched": []},
               "total_tokens": 0, "record_paths": [], "orchestration_path": "x"}
    _render_all_outcome(outcome)
    out = capsys.readouterr().out
    assert "Chart available" not in out
    assert "publish" not in out.lower()


def test_help_mentions_publish(capsys):
    assert _handle_command("/help", dataset="olist") is True
    assert "/publish [n]" in capsys.readouterr().out
