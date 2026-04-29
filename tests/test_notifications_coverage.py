"""Tests for notifications.py — Notifier, WebhookBackend, TelegramBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptotrader.notifications import (
    Notifier,
    TelegramBackend,
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


class TestTelegramBackend:
    def _make_backend(self):
        cfg = MagicMock()
        cfg.bot_token = "test-token"
        cfg.chat_id = "12345"
        return TelegramBackend(cfg)

    def test_format_price_trigger(self):
        msg = TelegramBackend._format_message("price_trigger", {"pair": "BTC/USDT", "trigger_reason": "crossed 50k"})
        assert "Price Trigger" in msg
        assert "BTC/USDT" in msg

    def test_format_trade(self):
        msg = TelegramBackend._format_message("trade", {"pair": "ETH/USDT", "action": "long"})
        assert "Trade" in msg
        assert "long" in msg

    def test_format_daily_summary(self):
        msg = TelegramBackend._format_message("daily_summary", {"daily_pnl": "+2.5%"})
        assert "Daily Summary" in msg
        assert "+2.5%" in msg

    def test_format_generic(self):
        msg = TelegramBackend._format_message("custom_event", {"key": "value"})
        assert "custom_event" in msg

    def test_start_stop_polling(self):
        backend = self._make_backend()
        # Can't actually start polling without event loop, but test stop on None task
        backend.stop_polling()
        assert backend._polling_task is None


class TestNotifier:
    @pytest.mark.asyncio
    async def test_disabled_notifier(self):
        n = Notifier(enabled=False)
        await n.notify("trade", {"pair": "BTC/USDT"})

    @pytest.mark.asyncio
    async def test_no_backends(self):
        n = Notifier()
        assert not n._enabled
        await n.notify("trade", {})

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
        await n.notify("trade", {"pair": "BTC"})
        n._backends[0].send.assert_called_once()

    @pytest.mark.asyncio
    async def test_backend_error_handled(self):
        n = Notifier(webhook_url="https://example.com/hook")
        n._backends[0] = MagicMock()
        n._backends[0].send = AsyncMock(side_effect=Exception("network"))
        await n.notify("trade", {"pair": "BTC"})

    def test_telegram_property(self):
        n = Notifier()
        assert n.telegram is None

    @pytest.mark.asyncio
    async def test_with_telegram_config(self):
        cfg = MagicMock()
        cfg.enabled = True
        cfg.bot_token = "test-token"
        cfg.chat_id = "123"
        n = Notifier(telegram_config=cfg)
        assert n.telegram is not None
        assert len(n._backends) == 1
