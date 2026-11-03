"""Tests for the make_chart tool (no API calls, temp dirs only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.loop import render_answer  # noqa: E402
from agent.record import AnalysisRecord  # noqa: E402
from agent.tools import TOOL_SCHEMAS, ToolContext, dispatch, schemas_for  # noqa: E402

ROWS = [["SP", 498440.43], ["RJ", 12.345], ["MG", 100.0]]


def _chart_ctx(
    tmp: Path,
    rows: list[list[Any]] | None = None,
    truncated: bool = False,
    with_finding: bool = True,
    columns: tuple[str, str] = ("region", "revenue"),
) -> tuple[ToolContext, str]:
    runs = tmp / "runs"
    record = AnalysisRecord.create("chart?", runs)
    record.set_checklist(
        [{"id": "rev", "question": "Revenue?", "evidence_needed": "sum"}]
    )
    record.plan_action([{"item_id": "rev", "steps": ["sum"]}])
    data = rows if rows is not None else [list(r) for r in ROWS]
    qid = record.record_query(
        "SELECT region, revenue FROM t",
        {"columns": list(columns), "rows": data,
         "row_count": len(data), "truncated": truncated},
    )
    if with_finding:
        record.record_finding("rev", "revenue is big", qid, data[0][1])
    return ToolContext(backend=object(), question="chart?", runs_dir=runs, record=record), qid


def _chart(ctx: ToolContext, qid: str, **over: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "query_id": qid,
        "chart_type": "bar",
        "x_column": "region",
        "y_column": "revenue",
        "title": "Revenue by region",
    }
    args.update(over)
    return json.loads(dispatch(ctx, "make_chart", args))


def test_unknown_query_id_refused(tmp_path):
    ctx, _ = _chart_ctx(tmp_path)
    assert "unknown query" in _chart(ctx, "q9")["error"]


def test_failed_query_id_refused(tmp_path):
    ctx, _ = _chart_ctx(tmp_path)
    ctx.record.record_failed_query("SELECT bad", "no such column", "error")
    result = _chart(ctx, "f1")
    assert "no citable result" in result["error"]


def test_finding_required_first(tmp_path):
    ctx, qid = _chart_ctx(tmp_path, with_finding=False)
    result = _chart(ctx, qid)
    assert result["error"] == (
        f"query {qid!r} has no recorded finding; "
        "record a finding from this query first."
    )


def test_bad_column_refused(tmp_path):
    ctx, qid = _chart_ctx(tmp_path)
    assert "not a column" in _chart(ctx, qid, x_column="nope")["error"]


def test_bad_chart_type_refused(tmp_path):
    ctx, qid = _chart_ctx(tmp_path)
    assert "bar" in _chart(ctx, qid, chart_type="pie")["error"]


def test_non_numeric_y_refused(tmp_path):
    rows = [["a", "high"], ["b", "low"], ["c", "mid"]]
    ctx, qid = _chart_ctx(tmp_path, rows=rows, columns=("region", "label"))
    assert "non-numeric" in _chart(ctx, qid, y_column="label")["error"]


def test_too_many_points_refused(tmp_path):
    rows = [[f"r{i}", float(i)] for i in range(51)]
    ctx, qid = _chart_ctx(tmp_path, rows=rows)
    result = _chart(ctx, qid)
    assert "2 to 50" in result["error"]


def test_too_few_points_refused(tmp_path):
    ctx, qid = _chart_ctx(tmp_path, rows=[["only", 1.0]])
    assert "2 to 50" in _chart(ctx, qid)["error"]


def test_truncated_always_refused(tmp_path):
    ctx, qid = _chart_ctx(tmp_path, truncated=True)
    assert "cut at 200 rows" in _chart(ctx, qid)["error"]


def test_title_number_not_in_rows_refused(tmp_path):
    rows = [[f"r{i}", float(i * 10)] for i in range(5)]
    ctx, qid = _chart_ctx(tmp_path, rows=rows)
    result = _chart(ctx, qid, title="Top 7 regions")
    assert "not in the query rows" in result["error"]


def test_title_count_allowed(tmp_path):
    rows = [[f"r{i}", float(i * 10)] for i in range(5)]
    ctx, qid = _chart_ctx(tmp_path, rows=rows)
    assert _chart(ctx, qid, title="Top 5 regions")["ok"] is True


def test_title_year_passes_without_row_match(tmp_path):
    ctx, qid = _chart_ctx(tmp_path)
    result = _chart(ctx, qid, title="Revenue 2017")
    assert result["ok"] is True


def test_success_writes_png_and_records(tmp_path):
    ctx, qid = _chart_ctx(tmp_path)
    result = _chart(ctx, qid)
    assert result["ok"] is True
    assert result["points"] == 3
    path = Path(result["path"])
    assert path == tmp_path / "charts" / f"{ctx.record.run_id}-1.png"
    assert path.read_bytes()[:4] == b"\x89PNG"
    assert "bar chart of revenue by region, 3 points" in result["description"]
    charts = ctx.record.charts
    assert len(charts) == 1
    assert charts[0]["query_id"] == qid
    assert charts[0]["path"] == str(path)
    assert any(e["action"] == "record_chart" for e in ctx.record.audit)


def test_second_chart_increments_filename(tmp_path):
    ctx, qid = _chart_ctx(tmp_path)
    first = _chart(ctx, qid)
    second = _chart(ctx, qid, chart_type="line", title="Revenue line")
    assert second["ok"] is True
    assert Path(second["path"]).name == f"{ctx.record.run_id}-2.png"
    assert first["path"] != second["path"]


def test_line_chart_ok_and_none_rows_skipped(tmp_path):
    rows = [["a", 1.0], ["b", None], ["c", 3.0]]
    ctx, qid = _chart_ctx(tmp_path, rows=rows)
    result = _chart(ctx, qid, chart_type="line", title="Values")
    assert result["ok"] is True
    assert result["points"] == 2
    assert "skipped" in result["description"]


def test_tool_in_registry_and_schema_lint(tmp_path):
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "make_chart" in names
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == "make_chart")
    assert schema["strict"] is True
    assert schema["parameters"]["additionalProperties"] is False
    assert set(schema["parameters"]["required"]) == set(
        schema["parameters"]["properties"]
    )
    assert "path" not in schema["parameters"]["properties"]
    ctx, _ = _chart_ctx(tmp_path)
    assert any(s["name"] == "make_chart" for s in schemas_for(ctx))
    ctx.allow_charts = False
    assert all(s["name"] != "make_chart" for s in schemas_for(ctx))
    refused = json.loads(dispatch(ctx, "make_chart", {}))
    assert "cross-dataset" in refused["error"]


def test_render_chart_line_never_verified():
    submission = {
        "answer": "Revenue is 498,440.43.",
        "decisions": [],
        "gaps": [],
        "matched": ["498,440.43"],
        "unmatched": [],
    }
    before = json.dumps(submission, sort_keys=True)
    text = render_answer(
        submission, [], ["workspace/charts/20240102-abcdef-1.png"]
    )
    assert "Chart: workspace/charts/20240102-abcdef-1.png" in text
    assert json.dumps(submission, sort_keys=True) == before
    assert submission["matched"] == ["498,440.43"]
    assert submission["unmatched"] == []
