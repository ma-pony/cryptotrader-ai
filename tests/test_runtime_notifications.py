"""Runtime summary delivery honors explicit event selection."""

from unittest.mock import AsyncMock

import pytest

from cryptotrader.notifications import Notifier


@pytest.mark.asyncio
async def test_runtime_notifier_filters_unemitted_events():
    notifier = Notifier(webhook_url="https://example.test/hook", events=["daily_summary"])
    send = AsyncMock()
    notifier._backends[0].send = send
    await notifier.notify("trade", {"pair": "BTC/USDT"})
    await notifier.notify("daily_summary", {"config_revision": 3})
    send.assert_awaited_once_with("daily_summary", {"config_revision": 3})
