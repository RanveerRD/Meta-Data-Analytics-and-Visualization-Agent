"""Console entry point: .venv/bin/python -m agent."""

from __future__ import annotations

import argparse
from functools import partial

from agent.backend import SqliteBackend
from agent.loop import REPO_ROOT, answer_question, build_client
from agent.pg_backend import PostgresBackend
from agent.preliminary_analysis import (
    DEFAULT_DATASET,
    MEMORY_DIR,
    DefinitionStore,
    approve_definition,
)

DATASETS = ("olist", "adventureworks")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse console flags; --dataset picks the database and memory store."""
    parser = argparse.ArgumentParser(description="Ask questions about a dataset.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default="olist",
        help="which dataset to query (default: olist)",
    )
    return parser.parse_args(argv)


def _base_instructions(dialect: str) -> str:
    try:
        text = (REPO_ROOT / "agent" / "instructions.md").read_text()
    except OSError as exc:
        raise RuntimeError(f"cannot read instructions: {exc}") from exc
    return text.replace("{dialect}", dialect)


def _instructions_for_question(
    base: str, dataset: str = DEFAULT_DATASET, dialect_notes: str = ""
) -> str:
    """Base instructions plus dialect notes and this question's definitions."""
    instructions = base
    if dialect_notes:
        instructions += "\n\nDialect notes:\n" + dialect_notes
    section = DefinitionStore.load(dataset).build_prompt_section()
    if not section:
        return instructions
    return instructions + "\n\n" + section


def _handle_command(
    line: str, memory_dir=MEMORY_DIR, dataset: str = DEFAULT_DATASET
) -> bool:
    """Run a / command. Returns True when the line was a command."""
    if not line.startswith("/"):
        return False
    if line == "/notes":
        approved = DefinitionStore.load(dataset, memory_dir).approved_entries()
        if not approved:
            print("no approved notes yet")
        for entry in approved:
            print(f"- {entry['key']} ({entry['kind']}): {entry['text']}")
        return True
    if line.startswith("/forget "):
        key = line[len("/forget "):].strip()
        try:
            store = DefinitionStore.load(dataset, memory_dir)
            entry = store.retire(key)
        except ValueError as exc:
            print(f"error: {exc}")
        else:
            print(f"retired {entry['key']}: {entry['text']}")
        return True
    print("unknown command (try /notes or /forget <key>)")
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
        marker = " (last used)" if option.get("last_used") else ""
        print(f"  {i}. {option['label']} [{_format_evidence(option['evidence'])}]{marker}")
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
        if reply == "":
            for option in options:
                if option.get("last_used"):
                    return {"choice": option["label"]}
            print(f"pick 1 to {len(labels)}, or 0 to type your own")
            continue
        if reply.isdigit():
            print(f"pick 1 to {len(labels)}, or 0 to type your own")
            continue
        return {"user_question": reply}


def _definition_text(clarification: dict) -> str:
    """Text to save for a clarification choice.

    An option pick saves as "<label>: <explanation>"; a typed answer, which
    matches no option label, saves as typed.
    """
    reply = clarification["reply"]
    for option in clarification.get("options", []):
        if option.get("label") == reply:
            return f"{reply}: {option.get('explanation', '')}".strip()
    return reply


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
    for option in clarification.get("options", []):
        if option.get("last_used") and option.get("label") == clarification["reply"]:
            return False
    try:
        reply = ask(
            f"Save as the new default for {key}? (y/n) "
        ).strip().lower()
    except EOFError:
        return False
    if reply not in ("y", "yes"):
        return False
    evidence = sorted({o["query_id"] for o in clarification.get("options", [])})
    try:
        entry = approve(key, _definition_text(clarification), evidence)
    except ValueError as exc:
        print(f"error: {exc}")
        return False
    else:
        print(f"approved {entry['key']}: {entry['text']}")
        return True


def _review_proposed_notes(dataset: str = DEFAULT_DATASET) -> None:
    """Ask the user to approve or reject each proposed note."""
    store = DefinitionStore.load(dataset)
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


def _backend_for_dataset(dataset: str):
    """Connect to the chosen dataset; fail fast with a clear error."""
    if dataset == "adventureworks":
        return PostgresBackend()
    return SqliteBackend(REPO_ROOT / "data" / "olist.sqlite")


_BANNERS = {
    "olist": ("Ask about the Olist database. Empty line exits.", "olist> "),
    "adventureworks": (
        "Ask about the AdventureWorks database. Empty line exits.",
        "adventureworks> ",
    ),
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset
    try:
        client, model = build_client()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    try:
        backend = _backend_for_dataset(dataset)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    try:
        base_instructions = _base_instructions(backend.dialect)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    banner, prompt_label = _BANNERS[dataset]
    save = partial(approve_definition, dataset=dataset)
    history: list = []
    print(banner)
    while True:
        try:
            question = input(prompt_label)
        except EOFError:
            break
        if not question.strip():
            break
        if _handle_command(question.strip(), dataset=dataset):
            continue
        instructions = _instructions_for_question(
            base_instructions, dataset, backend.dialect_notes
        )
        result = answer_question(
            client, model, instructions, backend, history, question,
            ask_user=_console_ask_user,
            dataset=dataset,
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
        _offer_definition_save(result.clarification, approve=save)
        _review_proposed_notes(dataset)
    backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
