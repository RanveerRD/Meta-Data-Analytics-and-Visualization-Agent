"""Metabase connector stub: raises until it is configured."""

from __future__ import annotations

from agent.bi import PublishResult


class MetabaseConnector:
    """Placeholder; publish always refuses with a clear message."""

    def __init__(self, database: str) -> None:
        self.database = database

    def publish(
        self, sql: str, title: str, chart_type: str, x: str, y: str
    ) -> PublishResult:
        raise RuntimeError("Metabase connector is not configured yet")
