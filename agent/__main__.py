"""Console entry point: .venv/bin/python -m agent."""

from __future__ import annotations

from agent.backend import SqliteBackend
from agent.loop import REPO_ROOT, answer_question, build_client


def main() -> int:
    try:
        client, model = build_client()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    backend = SqliteBackend(REPO_ROOT / "data" / "olist.sqlite")
    try:
        instructions = (REPO_ROOT / "agent" / "instructions.md").read_text()
    except OSError as exc:
        print(f"error: cannot read instructions: {exc}")
        return 2
    instructions = instructions.replace("{dialect}", backend.dialect)
    history: list = []
    print("Ask about the Olist database. Empty line exits.")
    while True:
        try:
            question = input("olist> ")
        except EOFError:
            break
        if not question.strip():
            break
        result = answer_question(client, model, instructions, backend, history, question)
        if result.text:
            print(result.text)
        if result.usage_known:
            print(
                f"tokens: {result.input_tokens} in / "
                f"{result.output_tokens} out / {result.total_tokens} total"
            )
        else:
            print("tokens: n/a")
        if result.record_path:
            print(f"Record: {result.record_path}")
        if result.verification:
            print(result.verification)
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
