"""Tests for the orchestrator/worker pattern (no API calls, temp dirs only)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from agent.__main__ import parse_args
from agent.datasets import REGISTRY
from agent.loop import TurnResult
from agent import orchestrator
from agent.orchestrator import merge_answers, plan_datasets


class _FakeUsage:
    def __init__(self, total: int = 5):
        self.total_tokens = total


class _FakeCall:
    def __init__(self, name: str, arguments: str):
        self.type = "function_call"
        self.name = name
        self.arguments = arguments


class _FakeResponse:
    def __init__(self, output_text: str = "", output: list | None = None, total: int = 5):
        self.output_text = output_text
        self.output = output or []
        self.usage = _FakeUsage(total)


class _FakeClient:
    """Scripted stand-in for the OpenAI client."""

    def __init__(self, script: list[_FakeResponse]):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    @property
    def responses(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._script.pop(0)


def _plan_response(names: list[str]) -> _FakeResponse:
    return _FakeResponse(output_text=json.dumps({"datasets": names}))


def _merge_response(text: str | None, separate: bool = False) -> _FakeResponse:
    args = json.dumps({"merged_text": text, "use_separate": separate})
    return _FakeResponse(output=[_FakeCall("record_merge", args)], total=11)


def _mapping(names: tuple[str, ...] = ("b", "a", "c")) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "description": f"{name} description",
            "backend": object(),
            "instructions": f"INSTRUCTIONS-{name}",
        }
        for name in names
    }


def _worker_answer(text: str, tokens: int = 3) -> TurnResult:
    return TurnResult(
        text=text + "\n\nDecisions:\n\nGaps:\n- none found",
        total_tokens=tokens,
        record_path="workspace/runs/fake/record.json",
        submission={
            "answer": text,
            "decisions": [],
            "gaps": [],
            "matched": [],
            "unmatched": [],
        },
    )


def _run(question, mapping, client, worker_texts, ask_user_calls=None):
    """Run the orchestrator with patched boundaries; returns (outcome, seen)."""
    seen: list[dict[str, Any]] = []

    def _fake_answer(client_, model_, instructions_, backend_, history_, question_, **kwargs):
        seen.append(
            {
                "question": question_,
                "instructions": instructions_,
                "dataset": kwargs.get("dataset"),
                "history": history_,
            }
        )
        if ask_user_calls is not None:
            reply = kwargs["ask_user"]("Which scope?", ["x", "y"])
            ask_user_calls.append(reply)
        return worker_texts[kwargs.get("dataset")]

    with (
        mock.patch.object(orchestrator, "build_client", return_value=client),
        mock.patch.object(orchestrator, "answer_question", side_effect=_fake_answer),
        mock.patch.object(
            orchestrator.AnalysisRecord,
            "load",
            return_value=SimpleNamespace(clarifications=[]),
        ),
    ):
        outcome = orchestrator.run_cross_dataset_question(question, mapping)
    return outcome, seen


@pytest.fixture
def _tmp_run_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(orchestrator, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("META_MODEL", "fake-model")
    return tmp_path


# --- planner validation ---


def test_plan_valid_returns_registry_order():
    client = _FakeClient([_plan_response(["c", "b"])])
    selected, tokens = plan_datasets("q", _mapping(), client, "m")
    assert selected == ["b", "c"]
    assert tokens == 5


def test_plan_unknown_name_retried_then_errors():
    client = _FakeClient([_plan_response(["nope"]), _plan_response(["nope"])])
    with pytest.raises(RuntimeError):
        plan_datasets("q", _mapping(), client, "m")
    assert len(client.calls) == 2


def test_plan_unknown_then_valid_recovers():
    client = _FakeClient([_plan_response(["nope"]), _plan_response(["a"])])
    selected, _ = plan_datasets("q", _mapping(), client, "m")
    assert selected == ["a"]


def test_plan_duplicate_rejected():
    client = _FakeClient([_plan_response(["a", "a"]), _plan_response(["a", "a"])])
    with pytest.raises(RuntimeError):
        plan_datasets("q", _mapping(), client, "m")


def test_plan_empty_rejected():
    client = _FakeClient([_plan_response([]), _plan_response([])])
    with pytest.raises(RuntimeError):
        plan_datasets("q", _mapping(), client, "m")


def test_plan_bad_json_retried():
    client = _FakeClient([_FakeResponse(output_text="not json"), _plan_response(["b"])])
    selected, _ = plan_datasets("q", _mapping(), client, "m")
    assert selected == ["b"]


# --- workers run in registry order with verbatim question + scope note ---


def test_workers_run_in_mapping_order_with_scope(_tmp_run_dirs):
    texts = {
        "b": _worker_answer("Bee total is 10."),
        "a": _worker_answer("Ay total is 20."),
        "c": _worker_answer("Cee total is 30."),
    }
    client = _FakeClient(
        [
            _plan_response(["c", "b", "a"]),
            _merge_response("Bee total is 10. Ay total is 20. Cee total is 30."),
        ]
    )
    outcome, seen = _run("How many?", _mapping(), client, texts)
    assert outcome["routing"] == ["b", "a", "c"]
    assert [s["dataset"] for s in seen] == ["b", "a", "c"]
    for s in seen:
        assert s["question"] == "How many?"
        assert s["history"] == []
        assert s["instructions"].startswith(f"INSTRUCTIONS-{s['dataset']}")
        assert f"Scope: use only the {s['dataset']} data:" in s["instructions"]
    assert outcome["merged"] == "Bee total is 10. Ay total is 20. Cee total is 30."
    assert outcome["verification"] == {"status": "verified", "unmatched": []}


def test_third_fake_dataset_proves_nothing_hard_coded(_tmp_run_dirs):
    texts = {
        "b": _worker_answer("Bee total is 10."),
        "a": _worker_answer("Ay total is 20."),
        "c": _worker_answer("Cee total is 30."),
    }
    client = _FakeClient(
        [_plan_response(["c"]), _merge_response("Cee total is 30.")]
    )
    outcome, seen = _run("How many?", _mapping(), client, texts)
    assert outcome["routing"] == ["c"]
    assert [s["dataset"] for s in seen] == ["c"]
    assert outcome["merged"] == "Cee total is 30."


def test_clarification_prompt_carries_dataset_prefix(_tmp_run_dirs, capsys):
    texts = {"b": _worker_answer("Bee total is 10.")}
    client = _FakeClient([_plan_response(["b"]), _merge_response("Bee total is 10.")])
    replies: list[str] = []
    with mock.patch("builtins.input", return_value="1"):
        _run("How many?", _mapping(("b",)), client, texts, ask_user_calls=replies)
    assert replies == ["x"]
    assert "[b]" in capsys.readouterr().out


# --- merge verification ---


def test_merge_unmatched_retried_then_fallback(_tmp_run_dirs):
    texts = {"b": _worker_answer("Bee total is 10.")}
    client = _FakeClient(
        [
            _plan_response(["b"]),
            _merge_response("Bee total is 999."),
            _merge_response("Bee total is 999."),
            _merge_response("Bee total is 999."),
        ]
    )
    outcome, _ = _run("How many?", _mapping(("b",)), client, texts)
    assert outcome["merged"] == ""
    assert outcome["verification"]["status"] == "fallback"
    assert outcome["verification"]["unmatched"] == ["999"]
    assert len(client.calls) == 4  # 1 plan + 3 merge attempts


def test_merge_use_separate_shows_answers_separately(_tmp_run_dirs):
    texts = {"b": _worker_answer("Bee total is 10.")}
    client = _FakeClient([_plan_response(["b"]), _merge_response(None, separate=True)])
    outcome, _ = _run("How many?", _mapping(("b",)), client, texts)
    assert outcome["merged"] == ""
    assert outcome["verification"]["status"] == "separate"


def test_failed_worker_excluded_from_merge(_tmp_run_dirs):
    texts = {
        "b": _worker_answer("Bee total is 10."),
        "a": TurnResult(text="error: boom", total_tokens=1),
    }
    client = _FakeClient([_plan_response(["b", "a"]), _merge_response("Bee total is 10.")])
    merge_inputs: list[str] = []
    real_merge = orchestrator.merge_answers

    def _spy(question, answers, client_, model_):
        merge_inputs.extend(a["text"] for a in answers)
        return real_merge(question, answers, client_, model_)

    with mock.patch.object(orchestrator, "merge_answers", side_effect=_spy):
        outcome, _ = _run("How many?", _mapping(("b", "a")), client, texts)
    assert outcome["failed"] == ["a"]
    assert merge_inputs == ["Bee total is 10."]
    assert outcome["merged"] == "Bee total is 10."


def test_orchestration_record_written(_tmp_run_dirs):
    texts = {
        "b": _worker_answer("Bee total is 10."),
        "a": _worker_answer("Ay total is 20."),
    }
    client = _FakeClient(
        [_plan_response(["b", "a"]), _merge_response("Bee total is 10. Ay total is 20.")]
    )
    outcome, _ = _run("How many?", _mapping(("b", "a")), client, texts)
    record_file = _tmp_run_dirs / outcome["orchestration_path"]
    payload = json.loads(record_file.read_text())
    assert payload["question"] == "How many?"
    assert payload["routing"] == ["b", "a"]
    assert [w["dataset"] for w in payload["workers"]] == ["b", "a"]
    assert all(w["status"] == "answered" for w in payload["workers"])
    assert payload["merged_answer"] == "Bee total is 10. Ay total is 20."
    assert payload["merge_verification"]["status"] == "verified"
    assert payload["total_tokens"] == 5 + 3 + 3 + 11
    assert outcome["orchestration_path"].endswith("-orchestration.json")


def test_orchestrator_imports_no_backend_or_driver():
    src = Path("agent/orchestrator.py").read_text()
    for banned in ("sqlite3", "psycopg", "Backend", "pg_backend"):
        assert banned not in src, banned


def test_turn_result_submission_defaults_to_none():
    assert TurnResult(text="hi").submission is None


def test_parse_args_all_and_registry_default():
    assert parse_args([]).dataset == next(iter(REGISTRY))
    assert parse_args(["--dataset", "all"]).dataset == "all"
    assert parse_args(["--dataset", "olist"]).dataset == "olist"
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "nope"])