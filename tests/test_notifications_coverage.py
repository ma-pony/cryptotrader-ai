"""Tests for notifications.py — Notifier, WebhookBackend."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from cryptotrader.notifications import (
    WebhookBackend,
)


class TestWebhookBackend:
    @pytest.mark.asyncio
    async def test_send(self):
        backend = WebhookBackend("https://example.com/hook")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            return_value=httpx.Response(204, request=httpx.Request("POST", "https://example.com/hook"))
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        from unittest.mock import patch

        with patch("httpx.AsyncClient", return_value=mock_client):
            await backend.send("trade", {"pair": "BTC/USDT"})
        mock_client.post.assert_called_once()
