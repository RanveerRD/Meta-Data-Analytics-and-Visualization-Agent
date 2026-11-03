"""python -m agent.whatsapp: loopback webhook server for WhatsApp questions."""

from __future__ import annotations

from agent.__main__ import _base_instructions, _instructions_for_question
from agent.datasets import REGISTRY
from agent.loop import REPO_ROOT, answer_question, build_client  # noqa: F401
from agent.whatsapp.client import WhatsAppClient
from agent.whatsapp.config import ConfigError, load_config
from agent.whatsapp.server import WhatsAppServer, format_turn_result


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    try:
        client, model = build_client()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    try:
        backend = REGISTRY[config.dataset]["make_backend"]()
        base_instructions = _base_instructions(backend.dialect)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    instructions = _instructions_for_question(
        base_instructions, config.dataset, backend.dialect_notes
    )
    wa_client = WhatsAppClient(
        config.api_version, config.phone_number_id, config.token
    )

    def answer(history: list, question: str, ask_user) -> str | None:
        result = answer_question(
            client,
            model,
            instructions,
            backend,
            history,
            question,
            ask_user=ask_user,
            dataset=config.dataset,
        )
        return format_turn_result(result)

    WhatsAppServer(config, wa_client, answer).serve()
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
