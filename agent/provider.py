"""Model provider seam: Meta (Responses API) or Baseten (Chat Completions).

The loop and orchestrator above this module see only RoundResult; every
provider difference (endpoint, key, message format, tool schema shape,
usage fields) lives here. The Meta path is the original behaviour.

Baseten notes: Chat Completions only (never the Responses endpoint, no raw
HTTP — the OpenAI SDK points at Baseten's base URL with Bearer auth).
`reasoning_content` is deliberately ignored everywhere: it is never read
into the result, history, records, logs or errors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

META_BASE_URL = "https://api.meta.ai/v1"
META_MODEL_DEFAULT = "muse-spark-1.3-contributor"
BASETEN_BASE_URL = "https://inference.baseten.co/v1"
BASETEN_MODEL_DEFAULT = "meta-models/muse-spark-open-1026"

PROVIDERS = ("meta", "baseten")

# Baseten list-price estimate (USD per 1M tokens). Easy to correct; never
# presented as anything but an estimate, and never used for Meta.
BASENTEN_PRICE_AS_OF = "4 Nov 2026"
BASENTEN_PRICE_INPUT_PER_M = 1.25
BASENTEN_PRICE_CACHED_INPUT_PER_M = 0.15
BASENTEN_PRICE_OUTPUT_PER_M = 4.25


def estimate_cost_usd(
    prompt_tokens: int, cached_tokens: int, completion_tokens: int
) -> float:
    """Baseten list-price estimate. Output tokens already include reasoning."""
    uncached = max(prompt_tokens - cached_tokens, 0)
    return (
        uncached * BASENTEN_PRICE_INPUT_PER_M
        + cached_tokens * BASENTEN_PRICE_CACHED_INPUT_PER_M
        + completion_tokens * BASENTEN_PRICE_OUTPUT_PER_M
    ) / 1_000_000


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


def get_provider_name() -> str:
    """Selected provider: MODEL_PROVIDER (env, then .env), default meta."""
    name = (
        os.environ.get("MODEL_PROVIDER")
        or _read_env_file_value("MODEL_PROVIDER")
        or "meta"
    ).strip().lower()
    if name not in PROVIDERS:
        raise RuntimeError(
            f"unknown MODEL_PROVIDER {name!r}; use one of: {', '.join(PROVIDERS)}"
        )
    return name


def build_client() -> tuple[OpenAI, str, str]:
    """Build the API client for the selected provider.

    Returns (client, model, provider). Raises a clear error if the key is
    missing. Keys and DSNs are never printed, logged or put in errors.
    """
    provider = get_provider_name()
    if provider == "baseten":
        api_key = os.environ.get("BASETEN_API_KEY") or _read_env_file_value(
            "BASETEN_API_KEY"
        )
        if not api_key:
            raise RuntimeError(
                "BASETEN_API_KEY is missing: add it to .env as "
                "BASETEN_API_KEY=<key> or export it in the environment. "
                "The key itself is never printed."
            )
        model = (
            os.environ.get("BASETEN_MODEL")
            or _read_env_file_value("BASETEN_MODEL")
            or BASETEN_MODEL_DEFAULT
        )
        return OpenAI(base_url=BASETEN_BASE_URL, api_key=api_key), model, provider
    api_key = os.environ.get("META_API_KEY") or _read_env_file_value("META_API_KEY")
    if not api_key:
        raise RuntimeError(
            "META_API_KEY is missing: add it to .env as META_API_KEY=<key> "
            "or export it in the environment. The key itself is never printed."
        )
    model = (
        os.environ.get("META_MODEL")
        or _read_env_file_value("META_MODEL")
        or META_MODEL_DEFAULT
    )
    return OpenAI(base_url=META_BASE_URL, api_key=api_key), model, provider


@dataclass
class RoundResult:
    """One normalized model round: final text plus tool calls."""

    text: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    usage_known: bool = True
    strict_fallback: bool = False


def chat_round(
    client: Any,
    model: str,
    provider: str,
    *,
    instructions: str,
    tools: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> RoundResult:
    """One model call plus normalization. Raises on API errors."""
    if provider == "baseten":
        return _baseten_round(client, model, instructions, tools, history)
    return _meta_round(client, model, instructions, tools, history)


def _meta_round(
    client: Any,
    model: str,
    instructions: str,
    tools: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> RoundResult:
    """Original Responses-API behaviour, unchanged."""
    response = client.responses.create(
        model=model,
        instructions=instructions,
        tools=tools,
        input=history,
    )
    response_usage = getattr(response, "usage", None)
    calls = [
        {
            "id": getattr(item, "call_id", "") or "",
            "name": getattr(item, "name", "") or "",
            "arguments": getattr(item, "arguments", "") or "{}",
        }
        for item in getattr(response, "output", [])
        if getattr(item, "type", None) == "function_call"
    ]
    text = "".join(
        block.text
        for item in getattr(response, "output", [])
        if getattr(item, "type", None) == "message"
        for block in getattr(item, "content", [])
        if getattr(block, "type", None) == "output_text"
    ) or getattr(response, "output_text", "") or ""
    if response_usage is None:
        return RoundResult(text=text, calls=calls, usage_known=False)
    return RoundResult(
        text=text,
        calls=calls,
        input_tokens=getattr(response_usage, "input_tokens", 0) or 0,
        output_tokens=getattr(response_usage, "output_tokens", 0) or 0,
        total_tokens=getattr(response_usage, "total_tokens", 0) or 0,
    )


def _chat_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Responses tool schemas translated to the Chat Completions shape."""
    translated = []
    for schema in schemas:
        function: dict[str, Any] = {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema.get(
                "parameters", {"type": "object", "properties": {}}
            ),
        }
        if schema.get("strict"):
            function["strict"] = True
        translated.append({"type": "function", "function": function})
    return translated


def _chat_messages(
    instructions: str, history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Shared history translated to Chat Completions messages."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": instructions}
    ]
    for item in history:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": item.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "") or "{}",
                            },
                        }
                    ],
                }
            )
        elif kind == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": str(item.get("output", "")),
                }
            )
        elif "role" in item:
            messages.append(
                {"role": item["role"], "content": item.get("content", "") or ""}
            )
        # Anything else is never sent (and never logged).
    return messages


def _is_strict_rejection(exc: BaseException) -> bool:
    """A 400 that complains about strict/structured output."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "strict" not in text and "structured" not in text:
        return False
    return (
        "400" in text
        or "bad request" in text
        or getattr(exc, "status_code", None) == 400
    )


def _normalize_chat_response(response: Any) -> RoundResult:
    """Chat Completions response normalized. reasoning_content is dropped."""
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    text = ""
    calls: list[dict[str, Any]] = []
    if message is not None:
        text = getattr(message, "content", None) or ""
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            arguments = (
                getattr(function, "arguments", "") if function is not None else ""
            )
            calls.append(
                {
                    "id": getattr(call, "id", ""),
                    "name": getattr(function, "name", "")
                    if function is not None
                    else "",
                    "arguments": arguments or "{}",
                }
            )
    # NOTE: message.reasoning_content is deliberately never read here: it
    # stays out of the result, history, records, logs and errors.
    usage = getattr(response, "usage", None)
    if usage is None:
        return RoundResult(text=text, calls=calls, usage_known=False)
    details = getattr(usage, "prompt_tokens_details", None)
    return RoundResult(
        text=text,
        calls=calls,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        cached_input_tokens=getattr(details, "cached_tokens", 0) or 0,
    )


def _baseten_round(
    client: Any,
    model: str,
    instructions: str,
    tools: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> RoundResult:
    """Chat Completions round with a one-time strict fallback."""
    chat_tools = _chat_tools(tools)
    messages = _chat_messages(instructions, history)
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if chat_tools:
        kwargs["tools"] = chat_tools
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        if _is_strict_rejection(exc) and any(
            tool["function"].get("strict") for tool in chat_tools
        ):
            kwargs["tools"] = [
                {
                    "type": tool["type"],
                    "function": {
                        key: value
                        for key, value in tool["function"].items()
                        if key != "strict"
                    },
                }
                for tool in chat_tools
            ]
            response = client.chat.completions.create(**kwargs)
            result = _normalize_chat_response(response)
            result.strict_fallback = True
            return result
        raise
    return _normalize_chat_response(response)
