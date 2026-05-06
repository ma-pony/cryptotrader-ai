"""Multi-backend notification system — async, fire-and-forget."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    from cryptotrader.config import TelegramConfig

logger = logging.getLogger(__name__)

_DEFAULT_EVENTS = [
    "trade",
    "rejection",
    "circuit_breaker",
    "reconcile_mismatch",
    "daily_summary",
    "portfolio_stale",
    "price_trigger",
]


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


class TelegramBackend:
    def __init__(self, config: TelegramConfig) -> None:
        self._token = config.bot_token
        self._chat_id = config.chat_id
        self._base_url = f"https://api.telegram.org/bot{self._token}"
        self._polling_task: asyncio.Task[None] | None = None
        self._status_callback: Any = None

    async def send(self, event: str, data: dict[str, Any]) -> None:
        text = self._format_message(event, data)
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        f"{self._base_url}/sendMessage",
                        json={"chat_id": self._chat_id, "text": text, "parse_mode": "Markdown"},
                    )
                    resp.raise_for_status()
                    return
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

    def start_polling(self, status_callback: Any = None) -> None:
        self._status_callback = status_callback
        self._polling_task = asyncio.create_task(self._poll_updates())

    def stop_polling(self) -> None:
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()

    async def _poll_updates(self) -> None:
        offset = 0
        while True:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(
                        f"{self._base_url}/getUpdates",
                        params={"offset": offset, "timeout": 25},
                    )
                    if resp.status_code != 200:
                        await asyncio.sleep(5)
                        continue
                    updates = resp.json().get("result", [])
                    for update in updates:
                        offset = update["update_id"] + 1
                        await self._handle_update(update)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.info("Telegram polling error", exc_info=True)
                await asyncio.sleep(5)

    async def _handle_update(self, update: dict[str, Any]) -> None:
        msg = update.get("message", {})
        text = msg.get("text", "")
        chat_id = msg.get("chat", {}).get("id")
        if text == "/status" and chat_id:
            status_text = "Scheduler status: running"
            if self._status_callback:
                try:
                    status_text = self._status_callback()
                except Exception:
                    status_text = "Failed to get status"
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(
                        f"{self._base_url}/sendMessage",
                        json={"chat_id": chat_id, "text": status_text},
                    )
            except Exception:
                logger.info("Failed to send /status reply", exc_info=True)

    @staticmethod
    def _format_message(event: str, data: dict[str, Any]) -> str:
        if event == "price_trigger":
            pair = data.get("pair", "")
            reason = data.get("trigger_reason", "")
            return f"*Price Trigger*\n{pair}: {reason}"
        if event == "trade":
            return f"*Trade* {data.get('pair', '')}: {data.get('action', '')}"
        if event == "daily_summary":
            return f"*Daily Summary*\nPnL: {data.get('daily_pnl', 'N/A')}"
        return f"*{event}*\n{json.dumps(data, default=str)[:500]}"


class Notifier:
    def __init__(
        self,
        webhook_url: str = "",
        enabled: bool = True,
        events: list[str] | None = None,
        webhook_timeout: int = 5,
        telegram_config: TelegramConfig | None = None,
    ):
        self._events = set(events or _DEFAULT_EVENTS)
        self._backends: list[NotifierBackend] = []
        if webhook_url:
            self._backends.append(WebhookBackend(webhook_url, webhook_timeout))
        if telegram_config and telegram_config.enabled and telegram_config.bot_token:
            self._telegram = TelegramBackend(telegram_config)
            self._backends.append(self._telegram)
        else:
            self._telegram = None
        self._enabled = enabled and bool(self._backends)

    @property
    def telegram(self) -> TelegramBackend | None:
        return self._telegram

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
