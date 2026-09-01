"""Webhook delivery for the scheduler's daily summary."""

from __future__ import annotations

from typing import Any

import httpx


class WebhookBackend:
    def __init__(self, url: str, timeout: int = 5, *, transport=None) -> None:
        self._url = url
        self._timeout = timeout
        self._transport = transport

    async def send(self, event: str, data: dict[str, Any]) -> None:
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            response = await client.post(self._url, json={"event": event, **data})
            response.raise_for_status()
