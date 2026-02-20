"""Webhook notification tests."""

import pytest
from cryptotrader.notifications import Notifier


def test_notifier_disabled():
    n = Notifier(webhook_url="", enabled=True)
    assert not n._enabled


def test_notifier_event_filter():
    n = Notifier(webhook_url="http://example.com", events=["trade"])
    assert "trade" in n._events
    assert "rejection" not in n._events


@pytest.mark.asyncio
async def test_notify_skips_disabled():
    n = Notifier(webhook_url="", enabled=False)
    await n.notify("trade", {"pair": "BTC/USDT"})  # should not raise
