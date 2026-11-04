"""python -m agent.whatsapp: loopback webhook server for WhatsApp questions."""

from __future__ import annotations

import logging
import sys
from typing import Any

from agent.__main__ import _base_instructions, _instructions_for_question
from agent.datasets import REGISTRY
from agent.loop import REPO_ROOT, answer_question, build_client  # noqa: F401
from agent.whatsapp.client import WhatsAppClient
from agent.whatsapp.config import ConfigError, load_config
from agent.whatsapp.server import WhatsAppServer, format_turn_result

HOST = "127.0.0.1"
PORT = 8080
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    """INFO to stderr with timestamp and level."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format=_LOG_FORMAT, force=True)


def startup_line(config) -> str:
    """One startup line; kept exact by the startup test."""
    count = len(config.allowed_numbers)
    return (
        f"WhatsApp webhook ready on {HOST}:{PORT}, "
        f"dataset={config.dataset}, {count} allowed number(s)"
    )


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    configure_logging()
    try:
        client, model = build_client()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    try:
        _base_instructions("SQLite")  # probe: fail fast if instructions unreadable
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    base_by_dialect: dict[str, str] = {}
    wa_client = WhatsAppClient(
        config.api_version, config.phone_number_id, config.token
    )

    def make_answer(session: dict) -> Any:
        """Build one session's answer function around its worker-thread backend."""
        session_backend = session["backend"]
        base_instructions = base_by_dialect.get(session_backend.dialect)
        if base_instructions is None:
            base_instructions = _base_instructions(session_backend.dialect)
            base_by_dialect[session_backend.dialect] = base_instructions
        session_instructions = _instructions_for_question(
            base_instructions, session["dataset"], session_backend.dialect_notes
        )

        def answer(history: list, question: str, ask_user) -> str | None:
            result = answer_question(
                client,
                model,
                session_instructions,
                session_backend,
                history,
                question,
                ask_user=ask_user,
                dataset=session["dataset"],
            )
            return format_turn_result(result)

        return answer

    print(startup_line(config))
    WhatsAppServer(
        config,
        wa_client,
        make_answer,
        backend_factory=lambda name: REGISTRY[name]["make_backend"](),
        datasets=REGISTRY,
        default_dataset=None if config.dataset == "ask" else config.dataset,
    ).serve(HOST, PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
