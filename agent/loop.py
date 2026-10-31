"""Agentic loop: one question in, grounded answer out.

Uses the OpenAI Python SDK against a Responses-compatible API with function
tools. Conversation history is kept across questions in the same session.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.backend import DatabaseBackend
from agent.tools import TOOL_SCHEMAS, ToolContext, dispatch

MODEL_DEFAULT = "muse-spark-1.3-contributor"
BASE_URL = "https://api.meta.ai/v1"
MAX_TURNS = 12
CAP_MESSAGE = "Reached the 12-turn limit without a final answer."

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
        return f"{name}: ok"
    return str(result)[:100]


def _record_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
    in_tokens = out_tokens = total_tokens = 0
    usage_known = True
    last_text = ""
    capped = False
    for _ in range(MAX_TURNS):
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                tools=TOOL_SCHEMAS,
                input=history,
            )
        except Exception as exc:
            print(f"api error: {type(exc).__name__}: {exc}")
            del history[snapshot:]
            break
        usage = getattr(response, "usage", None)
        if usage is None:
            usage_known = False
        else:
            in_tokens += getattr(usage, "input_tokens", 0) or 0
            out_tokens += getattr(usage, "output_tokens", 0) or 0
            total_tokens += getattr(usage, "total_tokens", 0) or 0
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
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result,
                }
            )
        last_text = _final_text(response)
        if not calls:
            break
    else:
        capped = True
    if capped:
        return TurnResult(
            text=CAP_MESSAGE,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=total_tokens,
            usage_known=usage_known,
            record_path=_record_display_path(ctx.record.path) if ctx.record else None,
        )
    if last_text:
        history.append({"role": "assistant", "content": last_text})
    return TurnResult(
        text=last_text,
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        total_tokens=total_tokens,
        usage_known=usage_known,
        record_path=_record_display_path(ctx.record.path) if ctx.record else None,
    )
