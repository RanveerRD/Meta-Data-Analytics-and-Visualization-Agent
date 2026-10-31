"""Agentic loop: one question in, grounded answer out.

Uses the OpenAI Python SDK against a Responses-compatible API with function
tools. Conversation history is kept across questions in the same session.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.backend import DatabaseBackend
from agent.tools import TOOL_SCHEMAS, dispatch

MODEL_DEFAULT = "muse-spark-1.3-contributor"
BASE_URL = "https://api.meta.ai/v1"
MAX_TURNS = 12

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _final_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", []):
            if getattr(block, "type", None) == "output_text":
                parts.append(getattr(block, "text", ""))
    return "".join(parts)


def answer_question(
    client: OpenAI,
    model: str,
    instructions: str,
    backend: DatabaseBackend,
    history: list[dict[str, Any]],
    question: str,
) -> str:
    """Ask one question, keeping shared history. At most MAX_TURNS model calls."""
    history.append({"role": "user", "content": question})
    last_text = ""
    for _ in range(MAX_TURNS):
        response = client.responses.create(
            model=model,
            instructions=instructions,
            tools=TOOL_SCHEMAS,
            input=history,
        )
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
                result = dispatch(backend, item.name, args)
            except Exception as exc:
                result = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
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
    if last_text:
        history.append({"role": "assistant", "content": last_text})
    return last_text or "(no answer returned)"
