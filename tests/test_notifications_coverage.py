"""Tests for notifications.py — Notifier, WebhookBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptotrader.notifications import (
    Notifier,
    WebhookBackend,
)


class TestWebhookBackend:
    @pytest.mark.asyncio
    async def test_send(self):
        backend = WebhookBackend("https://example.com/hook")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        from unittest.mock import patch

        with patch("httpx.AsyncClient", return_value=mock_client):
            await backend.send("trade", {"pair": "BTC/USDT"})
        mock_client.post.assert_called_once()


class TestNotifier:
    @pytest.mark.asyncio
    async def test_disabled_notifier(self):
        n = Notifier(enabled=False)
        await n.notify("daily_summary", {"pair": "BTC/USDT"})

    @pytest.mark.asyncio
    async def test_no_backends(self):
        n = Notifier()
        assert not n._enabled
        await n.notify("daily_summary", {})

    @pytest.mark.asyncio
    async def test_event_not_in_list(self):
        n = Notifier(webhook_url="https://example.com/hook")
        await n.notify("unknown_event_xyz", {})

    @pytest.mark.asyncio
    async def test_with_webhook(self):
        n = Notifier(webhook_url="https://example.com/hook")
        assert n._enabled
        assert len(n._backends) == 1
        n._backends[0] = MagicMock()
        n._backends[0].send = AsyncMock()
        await n.notify("daily_summary", {"pair": "BTC"})
        n._backends[0].send.assert_called_once()

    @pytest.mark.asyncio
    async def test_backend_error_handled(self):
        n = Notifier(webhook_url="https://example.com/hook")
        n._backends[0] = MagicMock()
        n._backends[0].send = AsyncMock(side_effect=Exception("network"))
        await n.notify("daily_summary", {"pair": "BTC"})
