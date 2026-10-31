"""Agentic loop: one question in, grounded answer out.

Uses the OpenAI Python SDK against a Responses-compatible API with function
tools. Conversation history is kept across questions in the same session.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.backend import DatabaseBackend
from agent.tools import TOOL_SCHEMAS, ToolContext, dispatch

MODEL_DEFAULT = "muse-spark-1.3-contributor"
BASE_URL = "https://api.meta.ai/v1"
MAX_TURNS = 12
REMINDER_TURNS = 3
CAP_MESSAGE = "Reached the 12-turn limit without a final answer."
SUBMIT_REMINDER = (
    "You replied in plain text without submitting. Record a finding for "
    "each key number with record_finding, then call submit_answer. "
    "Do not answer in plain text."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "workspace" / "runs"


def _read_env_file_value(name: str) -> str | None:
    """Read one KEY=value line from .env without sourcing the whole file."""
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return None
    prefix = name + "="
    try:
        for line in env_file.read_text().splitlines():
            if line.startswith(prefix):
                return line.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        return None
    return None


def build_client() -> tuple[OpenAI, str]:
    """Build the API client. Raises a clear error if the key is missing."""
    api_key = os.environ.get("META_API_KEY") or _read_env_file_value("META_API_KEY")
    if not api_key:
        raise RuntimeError(
            "META_API_KEY is missing: add it to .env as META_API_KEY=<key> "
            "or export it in the environment. The key itself is never printed."
        )
    model = (
        os.environ.get("META_MODEL")
        or _read_env_file_value("META_MODEL")
        or MODEL_DEFAULT
    )
    return OpenAI(base_url=BASE_URL, api_key=api_key), model


@dataclass
class TurnResult:
    """Everything one console turn needs to print."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_known: bool = True
    record_path: str | None = None
    verification: str | None = None


@dataclass
class _Usage:
    """Running token totals for one turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    known: bool = True


def _final_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", []):
            if getattr(block, "type", None) == "output_text":
                parts.append(getattr(block, "text", ""))
    return "".join(parts)


def _short_args(args: dict[str, Any], limit: int = 120) -> str:
    text = json.dumps(args)
    return text if len(text) <= limit else text[:limit] + "..."


def _summarize(name: str, result: str) -> str:
    """One-line summary of a tool's JSON result for console display."""
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return str(result)[:100]
    if not isinstance(data, dict):
        return str(result)[:100]
    if "error" in data:
        return f"error: {str(data['error'])[:120]}"
    if "rows" in data:
        count = data.get("row_count", len(data["rows"]))
        suffix = " (truncated)" if data.get("truncated") else ""
        query_id = f" [{data['query_id']}]" if "query_id" in data else ""
        return f"{name}: {count} rows{suffix}{query_id}"
    if "tables" in data:
        return f"{name}: {len(data['tables'])} tables"
    if "columns" in data:
        return f"{name}: {len(data['columns'])} columns"
    if data.get("ok"):
        status = f" ({data['status']})" if "status" in data else ""
        return f"{name}: ok{status}"
    return str(result)[:100]


def _record_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _round(
    client: OpenAI,
    model: str,
    instructions: str,
    ctx: ToolContext,
    history: list[dict[str, Any]],
    usage: _Usage,
) -> tuple[str, bool, bool]:
    """One model call plus tool execution. Returns (text, had_calls, api_error)."""
    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            tools=TOOL_SCHEMAS,
            input=history,
        )
    except Exception as exc:
        print(f"api error: {type(exc).__name__}: {exc}")
        return "", False, True
    response_usage = getattr(response, "usage", None)
    if response_usage is None:
        usage.known = False
    else:
        usage.input_tokens += getattr(response_usage, "input_tokens", 0) or 0
        usage.output_tokens += getattr(response_usage, "output_tokens", 0) or 0
        usage.total_tokens += getattr(response_usage, "total_tokens", 0) or 0
    calls = [i for i in response.output if getattr(i, "type", None) == "function_call"]
    for item in calls:
        history.append(
            {
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            }
        )
        try:
            args = json.loads(item.arguments or "{}")
        except json.JSONDecodeError as exc:
            args = {}
            result = json.dumps({"error": f"JSONDecodeError: {exc}"})
        else:
            print(f"  tool: {item.name}({_short_args(args)})")
            result = dispatch(ctx, item.name, args)
        print(f"    -> {_summarize(item.name, result)}")
        try:
            warning = json.loads(result).get("warning", "")
        except (json.JSONDecodeError, TypeError, AttributeError):
            warning = ""
        if isinstance(warning, str) and warning.startswith("UNVERIFIED"):
            print(f"  !! {warning}")
        history.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": result,
            }
        )
    return _final_text(response), bool(calls), False


def answer_question(
    client: OpenAI,
    model: str,
    instructions: str,
    backend: DatabaseBackend,
    history: list[dict[str, Any]],
    question: str,
    runs_dir: Path = RUNS_DIR,
) -> TurnResult:
    """Ask one question, keeping shared history. At most MAX_TURNS model calls."""
    ctx = ToolContext(backend=backend, question=question, runs_dir=runs_dir)
    history.append({"role": "user", "content": question})
    snapshot = len(history) - 1
    usage = _Usage()
    last_text = ""
    capped = False
    api_failed = False
    for _ in range(MAX_TURNS):
        last_text, had_calls, api_error = _round(
            client, model, instructions, ctx, history, usage
        )
        if api_error:
            del history[snapshot:]
            api_failed = True
            break
        if not had_calls:
            break
    else:
        capped = True
    submission = ctx.record.submission if ctx.record is not None else None
    if (
        not capped
        and not api_failed
        and ctx.record is not None
        and submission is None
        and last_text
    ):
        history.append({"role": "user", "content": SUBMIT_REMINDER})
        for _ in range(REMINDER_TURNS):
            last_text, had_calls, api_error = _round(
                client, model, instructions, ctx, history, usage
            )
            if api_error:
                del history[snapshot:]
                api_failed = True
                break
            submission = ctx.record.submission if ctx.record is not None else None
            if submission is not None or not had_calls:
                break
    submission = ctx.record.submission if ctx.record is not None else None
    record_path = _record_display_path(ctx.record.path) if ctx.record else None
    if capped:
        return TurnResult(
            text=CAP_MESSAGE,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            usage_known=usage.known,
            record_path=record_path,
            verification="verification: no submitted answer"
            if ctx.record
            else None,
        )
    verification: str | None = None
    if ctx.record is not None:
        if submission is not None:
            text = submission["answer"]
            verification = (
                f"verification: {len(submission['matched'])} numbers matched, "
                f"{len(submission['unmatched'])} unmatched"
            )
        else:
            text = last_text
            verification = "verification: no submitted answer"
        if text:
            history.append({"role": "assistant", "content": text})
    else:
        text = last_text
        if text:
            history.append({"role": "assistant", "content": text})
    return TurnResult(
        text=text,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        usage_known=usage.known,
        record_path=record_path,
        verification=verification,
    )
