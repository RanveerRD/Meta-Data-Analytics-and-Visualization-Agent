"""Console entry point: .venv/bin/python -m agent."""

from __future__ import annotations

from agent.backend import SqliteBackend
from agent.loop import REPO_ROOT, answer_question, build_client
from agent.preliminary_analysis import (
    DEFAULT_DATASET,
    DefinitionStore,
    approve_definition,
)


def _base_instructions(dialect: str) -> str:
    try:
        text = (REPO_ROOT / "agent" / "instructions.md").read_text()
    except OSError as exc:
        raise RuntimeError(f"cannot read instructions: {exc}") from exc
    return text.replace("{dialect}", dialect)


def _instructions_for_question(base: str) -> str:
    """Base instructions plus this question's approved definitions."""
    section = DefinitionStore.load(DEFAULT_DATASET).build_prompt_section()
    if not section:
        return base
    return base + "\n\n" + section


def _handle_command(line: str) -> bool:
    """Run a / command. Returns True when the line was a command."""
    if not line.startswith("/"):
        return False
    if line == "/notes":
        approved = DefinitionStore.load(DEFAULT_DATASET).approved_entries()
        if not approved:
            print("no approved notes yet")
        for entry in approved:
            print(f"- {entry['key']} ({entry['kind']}): {entry['text']}")
        return True
    if line.startswith("/define "):
        rest = line[len("/define "):]
        key, sep, text = rest.partition("=")
        try:
            if not sep:
                raise ValueError("use /define <key> = <text>")
            entry = approve_definition(key.strip(), text.strip())
        except ValueError as exc:
            print(f"error: {exc}")
        else:
            print(f"approved {entry['key']}: {entry['text']}")
        return True
    print("unknown command (try /define <key> = <text> or /notes)")
    return True


def _format_evidence(value: object) -> str:
    """Format one evidence value; numbers get thousands separators."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,}"
    return str(value)


def _console_ask_user(
    question: str, options: list[dict], input_fn=input
) -> dict[str, str]:
    """Numbered choice prompt for ask_clarification.

    Returns {"choice": ...} for a valid option number or 0 plus typed text,
    else {"user_question": ...}. Bare digits out of range and empty replies
    reprompt locally without touching the model or the call budget.
    """
    labels = [o["label"] for o in options]
    print(f"clarification: {question}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option['label']} [{_format_evidence(option['evidence'])}]")
        print(f"     {option['explanation']}")
    print("  0. something else (type your own)")
    while True:
        reply = input_fn("choice (number): ").strip()
        if reply == "0":
            typed = input_fn("type your answer: ").strip()
            if typed:
                return {"choice": typed}
            continue
        try:
            index = int(reply)
        except ValueError:
            index = -1
        if 1 <= index <= len(labels):
            return {"choice": labels[index - 1]}
        if reply == "" or reply.isdigit():
            print(f"pick 1 to {len(labels)}, or 0 to type your own")
            continue
        return {"user_question": reply}


def _offer_definition_save(
    clarification: dict | None, ask=input, approve=approve_definition
) -> bool:
    """Offer to save a clarification choice as a default definition.

    Only runs when the user's choice carries a definition_key, and only
    console code can write the approval — never a model tool. Returns True
    when a definition was saved.
    """
    if not clarification or not clarification.get("definition_key"):
        return False
    key = clarification["definition_key"]
    try:
        reply = ask(
            f"Save this as the default definition for {key}? (y/n) "
        ).strip().lower()
    except EOFError:
        return False
    if reply not in ("y", "yes"):
        return False
    evidence = sorted({o["query_id"] for o in clarification.get("options", [])})
    try:
        entry = approve(key, clarification["reply"], evidence)
    except ValueError as exc:
        print(f"error: {exc}")
        return False
    else:
        print(f"approved {entry['key']}: {entry['text']}")
        return True


def _review_proposed_notes() -> None:
    """Ask the user to approve or reject each proposed note."""
    store = DefinitionStore.load(DEFAULT_DATASET)
    for entry in store.entries:
        if entry["status"] != "proposed":
            continue
        print(f"proposed note [{entry['id']}] ({entry['kind']}) {entry['key']}:")
        print(f"  {entry['text']}")
        if entry["evidence"]:
            print(f"  evidence: {', '.join(entry['evidence'])}")
        try:
            reply = input("Save this note? (y/n) ").strip().lower()
        except EOFError:
            break
        if reply in ("y", "yes"):
            store.decide(entry["id"], True)
            print(f"approved {entry['key']}")
        else:
            store.decide(entry["id"], False)
            print(f"rejected {entry['key']}")


def main() -> int:
    try:
        client, model = build_client()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    backend = SqliteBackend(REPO_ROOT / "data" / "olist.sqlite")
    try:
        base_instructions = _base_instructions(backend.dialect)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    history: list = []
    print("Ask about the Olist database. Empty line exits.")
    while True:
        try:
            question = input("olist> ")
        except EOFError:
            break
        if not question.strip():
            break
        if _handle_command(question.strip()):
            continue
        instructions = _instructions_for_question(base_instructions)
        result = answer_question(
            client, model, instructions, backend, history, question,
            ask_user=_console_ask_user,
        )
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
        _offer_definition_save(result.clarification)
        _review_proposed_notes()
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
