"""Webhook notification system — async, fire-and-forget."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class Notifier:
    def __init__(
        self, webhook_url: str = "", enabled: bool = True, events: list[str] | None = None, webhook_timeout: int = 5
    ):
        self._url = webhook_url
        self._enabled = enabled and bool(webhook_url)
        self._timeout = webhook_timeout
        _default_events = [
            "trade",
            "rejection",
            "circuit_breaker",
            "reconcile_mismatch",
            "daily_summary",
            "portfolio_stale",
        ]
        self._events = set(events or _default_events)

    async def notify(self, event: str, data: dict[str, Any]) -> None:
        if not self._enabled or event not in self._events:
            return
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                await client.post(self._url, json={"event": event, **data})
        except Exception as e:
            logger.warning("Notification failed: %s", e)
