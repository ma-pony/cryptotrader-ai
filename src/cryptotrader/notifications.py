"""Webhook delivery for the scheduler's daily summary."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

import httpx

logger = logging.getLogger(__name__)


@runtime_checkable
class NotifierBackend(Protocol):
    async def send(self, event: str, data: dict[str, Any]) -> None: ...


class WebhookBackend:
    def __init__(self, url: str, timeout: int = 5) -> None:
        self._url = url
        self._timeout = timeout

    async def send(self, event: str, data: dict[str, Any]) -> None:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            await client.post(self._url, json={"event": event, **data})


class Notifier:
    def __init__(
        self,
        webhook_url: str = "",
        enabled: bool = True,
        events: Sequence[str] | None = None,
        webhook_timeout: int = 5,
    ):
        self._events = set(("daily_summary",) if events is None else events)
        self._backends: list[NotifierBackend] = []
        if webhook_url:
            self._backends.append(WebhookBackend(webhook_url, webhook_timeout))
        self._enabled = enabled and bool(self._backends)

    async def notify(self, event: str, data: dict[str, Any]) -> None:
        if not self._enabled or event not in self._events:
            return
        if not self._backends:
            return
        for backend in self._backends:
            try:
                await backend.send(event, data)
            except Exception as e:
                logger.warning("Notification failed (%s): %s", type(backend).__name__, e)
