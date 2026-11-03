"""Console entry point: .venv/bin/python -m agent."""

from __future__ import annotations

import argparse
from functools import partial
from typing import Any

from agent.datasets import REGISTRY
from agent.loop import REPO_ROOT, answer_question, build_client
from agent.record import AnalysisRecord
from agent.preliminary_analysis import (
    DEFAULT_DATASET,
    MEMORY_DIR,
    DefinitionStore,
    approve_definition,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse console flags; --dataset picks the database and memory store."""
    parser = argparse.ArgumentParser(description="Ask questions about a dataset.")
    parser.add_argument(
        "--dataset",
        choices=[*REGISTRY, "all"],
        default=next(iter(REGISTRY)),
        help="which dataset to query (default: first registry entry)",
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


def _print_datasets(current: str) -> None:
    """List every registry dataset; no model call."""
    for name, entry in REGISTRY.items():
        marker = " (current)" if name == current else ""
        print(f"{name} ({entry['label']}): {entry['description']}{marker}")
    if current == "all":
        print("asking across all datasets (current)")
    print("Run with --dataset <name> or --dataset all")


def _run_chart_command(line: str, pending: dict | None) -> bool:
    """Draw a console-offered candidate with no model call. Always handled."""
    from agent.tools.make_chart import _is_date_series, render_stored_chart

    if not pending or not pending["candidates"]:
        print("No chart available for the last answer.")
        return True
    parts = line.split()
    number: Any = 1
    override: str | None = None
    for token in parts[1:]:
        if token in ("bar", "line"):
            override = token
        else:
            try:
                number = int(token)
            except ValueError:
                print("usage: /chart [n] [bar|line]")
                return True
    candidates = pending["candidates"]
    if not 1 <= number <= len(candidates):
        print(f"choose 1 to {len(candidates)}")
        return True
    candidate = candidates[number - 1]
    chart_type = override or candidate["type"]
    if chart_type not in ("bar", "line"):
        print("usage: /chart [n] [bar|line]")
        return True
    record = AnalysisRecord.load(pending["record_abs"])
    if chart_type == "line":
        query = next(
            (q for q in record.queries if q["id"] == candidate["query_id"]), None
        )
        columns = list(query.get("columns", [])) if query else []
        xi = columns.index(candidate["x"]) if candidate["x"] in columns else -1
        yi = columns.index(candidate["y"]) if candidate["y"] in columns else -1
        labels = (
            [str(r[xi]) for r in query["rows"] if r[yi] is not None]
            if query and xi >= 0 and yi >= 0
            else []
        )
        if len(labels) < 3 or not _is_date_series(labels):
            print(
                "line needs date-like x values (YYYY-MM-DD, YYYY-MM or YYYY) "
                "with at least 3 rows; use bar instead."
            )
            return True
    result = render_stored_chart(record, candidate, chart_type)
    if "error" in result:
        print(f"error: {result['error']}")
    else:
        print(f"chart saved: {result['path']}")
    return True


def _handle_command(
    line: str,
    memory_dir=MEMORY_DIR,
    dataset: str = DEFAULT_DATASET,
    pending_charts: dict | None = None,
) -> bool:
    """Run a / command. Returns True when the line was a command."""
    if not line.startswith("/"):
        return False
    if line == "/datasets":
        _print_datasets(dataset)
        return True
    if line == "/help":
        print("/notes - list approved notes")
        print("/forget <key> - retire the approved note for a key")
        print("/datasets - list available datasets")
        print("/chart [n] [bar|line] - draw an offered chart for the last answer")
        print("/help - show this list")
        return True
    if line == "/chart" or line.startswith("/chart "):
        return _run_chart_command(line, pending_charts)
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
    print("unknown command (try /help)")
    return True


def _print_new_charts(text: str, charts: list[str]) -> None:
    """Announce charts not already listed under the answer (e.g. /chart)."""
    for path in charts:
        if path not in text:
            print(f"chart saved: {path}")


def _offer_charts(candidates: list[dict], drawn: list[str]) -> None:
    """Print one offer line per candidate. No model call, no tokens."""
    drawn_ids = set(drawn)
    for i, candidate in enumerate(candidates, 1):
        suffix = " (already drawn)" if candidate["query_id"] in drawn_ids else ""
        print(
            f"Chart available: {i}. {candidate['type']} chart of "
            f"{candidate['y']} by {candidate['x']} "
            f"({candidate['points']} points). Type /chart {i}{suffix}"
        )


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
    return REGISTRY[dataset]["make_backend"]()


def _banner_and_prompt(dataset: str) -> tuple[str, str]:
    """Banner and prompt label derived from the registry entry."""
    entry = REGISTRY[dataset]
    return (
        f"Ask about the {entry['label']} database. Empty line exits.",
        entry["prompt"],
    )


def _all_banner() -> str:
    labels = [entry["label"] for entry in REGISTRY.values()]
    return f"Ask across {' and '.join(labels)}. Empty line exits."


def _instructions_for_dataset(name: str, backend) -> str:
    """Full per-question instructions for one dataset worker."""
    instructions = _base_instructions(backend.dialect)
    if backend.dialect_notes:
        instructions += "\n\nDialect notes:\n" + backend.dialect_notes
    section = DefinitionStore.load(name).build_prompt_section()
    if section:
        instructions += "\n\n" + section
    return instructions


def _render_all_outcome(outcome: dict) -> None:
    """Print the merged reply, then each worker's rendered answer."""
    print(f"routing: {', '.join(outcome['routing'])}")
    if outcome["merged"]:
        print("\nCombined:\n" + outcome["merged"])
    elif outcome["verification"]["status"] == "fallback":
        print(
            "\nUNVERIFIED NUMBERS: "
            + ", ".join(outcome["verification"]["unmatched"])
        )
    else:
        print("\nNote: answers shown separately; no combined reply recorded.")
    for worker in outcome["workers"]:
        print(f"\n[{worker['dataset']}]")
        if worker["answer"] is None:
            if worker.get("partial_summary"):
                print(worker["partial_summary"])
            else:
                print(worker.get("note", "No accepted submission."))
            continue
        print(worker["answer"])
    if outcome["failed"]:
        print(
            "\nNote: no accepted answer from: "
            + ", ".join(outcome["failed"])
            + "; their numbers are excluded."
        )
    print(f"\ntokens: {outcome['total_tokens']} total")
    for path in outcome["record_paths"]:
        print(f"Record: {path}")
    print(f"Orchestration: {outcome['orchestration_path']}")


def _run_all_mode(client, model) -> int:
    """Ask one question across every dataset via the orchestrator."""
    from agent.orchestrator import run_cross_dataset_question

    datasets: dict = {}
    try:
        for name, entry in REGISTRY.items():
            backend = entry["make_backend"]()
            datasets[name] = {
                "description": entry["description"],
                "backend": backend,
                "instructions": _instructions_for_dataset(name, backend),
            }
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    print(_all_banner())
    try:
        while True:
            try:
                question = input("all> ")
            except EOFError:
                break
            if not question.strip():
                break
            if question.strip() == "/datasets":
                _print_datasets("all")
                continue
            if question.strip() == "/help":
                _handle_command("/help", dataset="all")
                continue
            if question.strip().startswith("/"):
                names = " or ".join(REGISTRY)
                print(f"Use --dataset {names} for saved definitions.")
                continue
            try:
                outcome = run_cross_dataset_question(
                    question, datasets, client, model, _console_ask_user
                )
            except RuntimeError as exc:
                print(f"error: {exc}")
                continue
            _render_all_outcome(outcome)
    finally:
        for info in datasets.values():
            info["backend"].close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset
    try:
        client, model = build_client()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2
    if dataset == "all":
        return _run_all_mode(client, model)
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
    banner, prompt_label = _banner_and_prompt(dataset)
    save = partial(approve_definition, dataset=dataset)
    history: list = []
    pending_charts: dict | None = None
    print(banner)
    while True:
        try:
            question = input(prompt_label)
        except EOFError:
            break
        if not question.strip():
            break
        if _handle_command(
            question.strip(), dataset=dataset, pending_charts=pending_charts
        ):
            continue
        pending_charts = None
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
        _print_new_charts(result.text, result.charts)
        if result.chart_candidates and result.record_path:
            pending_charts = {
                "record_abs": REPO_ROOT / result.record_path,
                "candidates": result.chart_candidates,
            }
            _offer_charts(result.chart_candidates, result.chart_drawn)
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
