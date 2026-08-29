"""Notification event policy uses runtime metadata and explicit secret delivery."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from cryptotrader.runtime_config.models import TelegramConfig


@pytest.mark.asyncio
async def test_runtime_notifier_filters_events_and_does_not_create_telegram_without_explicit_secret():
    from cryptotrader.notifications import Notifier

    notifier = Notifier(telegram_config=TelegramConfig(enabled=True, chat_id="ops"), events=["trade"])
    assert notifier.telegram is None

    notifier = Notifier(
        telegram_config=TelegramConfig(enabled=True, chat_id="ops"),
        telegram_bot_token="runtime-secret",  # pragma: allowlist secret
        events=["trade"],
    )
    send = AsyncMock()
    notifier._backends[0].send = send
    await notifier.notify("rejection", {"pair": "BTC/USDT"})
    await notifier.notify("trade", {"pair": "BTC/USDT"})

    send.assert_awaited_once_with("trade", {"pair": "BTC/USDT"})
