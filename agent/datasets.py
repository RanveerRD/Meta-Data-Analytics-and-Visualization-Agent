"""Dataset registry: the single place where available databases are listed.

Adding a database later means adding one entry here and nothing else:
name -> {description, label, prompt, make_backend}.

Factories import their backend lazily so importing this module never
requires a database driver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent


def _olist_backend() -> Any:
    from agent.backend import SqliteBackend

    return SqliteBackend(REPO_ROOT / "data" / "olist.sqlite")


def _adventureworks_backend() -> Any:
    from agent.pg_backend import PostgresBackend

    return PostgresBackend()


REGISTRY: dict[str, dict[str, Any]] = {
    "olist": {
        "description": (
            "Brazilian e-commerce marketplace orders, items, products, "
            "payments, and reviews."
        ),
        "label": "Olist",
        "prompt": "olist> ",
        "make_backend": _olist_backend,
        "bi": {"connector": "datasette", "database": "olist"},
    },
    "adventureworks": {
        "description": (
            "Bicycle company sales, products, purchasing, and HR."
        ),
        "label": "AdventureWorks",
        "prompt": "adventureworks> ",
        "make_backend": _adventureworks_backend,
        "bi": {"connector": "metabase", "database": "adventureworks"},
    },
}
