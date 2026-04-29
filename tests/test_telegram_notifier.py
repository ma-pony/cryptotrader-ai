"""Tests for TelegramBackend and multi-backend Notifier (T016).

Mocks httpx.AsyncClient to avoid real HTTP calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cryptotrader.config import TelegramConfig
from cryptotrader.notifications import Notifier, TelegramBackend, WebhookBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TEST_BOT_TOKEN = "test-token"


def _telegram_cfg(
    bot_token: str = _TEST_BOT_TOKEN,
    chat_id: str = "12345",
    enabled: bool = True,
) -> TelegramConfig:
    return TelegramConfig(bot_token=bot_token, chat_id=chat_id, enabled=enabled)


def _mock_http_client(status_code: int = 200, raise_exc: Exception | None = None):
    """Return a context-manager mock for httpx.AsyncClient."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"result": []})

    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    if raise_exc:
        client.post = AsyncMock(side_effect=raise_exc)
    else:
        client.post = AsyncMock(return_value=resp)

    client.get = AsyncMock(return_value=resp)
    return client


# ---------------------------------------------------------------------------
# TelegramBackend.send
# ---------------------------------------------------------------------------


class TestTelegramBackendSend:
    async def test_send_succeeds_on_first_attempt(self) -> None:
        backend = TelegramBackend(_telegram_cfg())
        mock_client = _mock_http_client()
        with patch("httpx.AsyncClient", return_value=mock_client):
            await backend.send("price_trigger", {"pair": "BTC/USDT", "trigger_reason": "dropped"})
        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert "sendMessage" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["chat_id"] == "12345"
        assert "parse_mode" in body

    async def test_send_retries_on_transient_failure(self) -> None:
        backend = TelegramBackend(_telegram_cfg())
        fail_resp = MagicMock()
        fail_resp.raise_for_status = MagicMock(side_effect=Exception("503"))
        ok_resp = MagicMock()
        ok_resp.raise_for_status = MagicMock()

        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        # First 2 fail, 3rd succeeds
        client.post = AsyncMock(side_effect=[fail_resp, fail_resp, ok_resp])

        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            await backend.send("trade", {"pair": "BTC/USDT", "action": "BUY"})

        assert client.post.await_count == 3

    async def test_send_raises_after_three_failures(self) -> None:
        backend = TelegramBackend(_telegram_cfg())
        bad_resp = MagicMock()
        bad_resp.status_code = 500
        bad_resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=bad_resp)
        )

        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.post = AsyncMock(return_value=bad_resp)

        with (
            patch("httpx.AsyncClient", return_value=client),
            patch("asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await backend.send("trade", {"pair": "ETH/USDT", "action": "SELL"})

        assert client.post.await_count == 3


# ---------------------------------------------------------------------------
# TelegramBackend._format_message
# ---------------------------------------------------------------------------


class TestTelegramFormatMessage:
    def test_price_trigger_format(self) -> None:
        msg = TelegramBackend._format_message("price_trigger", {"pair": "BTC/USDT", "trigger_reason": "fell below 50k"})
        assert "*Price Trigger*" in msg
        assert "BTC/USDT" in msg
        assert "fell below 50k" in msg

    def test_trade_format(self) -> None:
        msg = TelegramBackend._format_message("trade", {"pair": "ETH/USDT", "action": "BUY"})
        assert "*Trade*" in msg
        assert "ETH/USDT" in msg
        assert "BUY" in msg

    def test_daily_summary_format(self) -> None:
        msg = TelegramBackend._format_message("daily_summary", {"daily_pnl": "+2.5%"})
        assert "*Daily Summary*" in msg
        assert "+2.5%" in msg

    def test_unknown_event_falls_back_to_json(self) -> None:
        data = {"foo": "bar"}
        msg = TelegramBackend._format_message("unknown_event", data)
        assert "*unknown_event*" in msg
        assert "bar" in msg

    def test_daily_summary_missing_pnl(self) -> None:
        msg = TelegramBackend._format_message("daily_summary", {})
        assert "N/A" in msg


# ---------------------------------------------------------------------------
# TelegramBackend._handle_update
# ---------------------------------------------------------------------------


class TestTelegramHandleUpdate:
    async def test_status_command_calls_callback(self) -> None:
        backend = TelegramBackend(_telegram_cfg())
        backend._status_callback = MagicMock(return_value="All systems go")

        update = {
            "update_id": 1,
            "message": {"text": "/status", "chat": {"id": 99}},
        }
        mock_client = _mock_http_client()
        with patch("httpx.AsyncClient", return_value=mock_client):
            await backend._handle_update(update)

        mock_client.post.assert_awaited_once()
        body = mock_client.post.call_args[1]["json"]
        assert body["text"] == "All systems go"
        assert body["chat_id"] == 99

    async def test_status_command_without_callback(self) -> None:
        backend = TelegramBackend(_telegram_cfg())
        # No _status_callback set
        update = {
            "update_id": 2,
            "message": {"text": "/status", "chat": {"id": 42}},
        }
        mock_client = _mock_http_client()
        with patch("httpx.AsyncClient", return_value=mock_client):
            await backend._handle_update(update)

        # Should send default message
        mock_client.post.assert_awaited_once()
        body = mock_client.post.call_args[1]["json"]
        assert "running" in body["text"]

    async def test_non_status_command_ignored(self) -> None:
        backend = TelegramBackend(_telegram_cfg())
        update = {
            "update_id": 3,
            "message": {"text": "/help", "chat": {"id": 42}},
        }
        mock_client = _mock_http_client()
        with patch("httpx.AsyncClient", return_value=mock_client):
            await backend._handle_update(update)

        mock_client.post.assert_not_called()

    async def test_no_message_key_ignored(self) -> None:
        backend = TelegramBackend(_telegram_cfg())
        update = {"update_id": 4}
        mock_client = _mock_http_client()
        with patch("httpx.AsyncClient", return_value=mock_client):
            await backend._handle_update(update)
        mock_client.post.assert_not_called()


# ---------------------------------------------------------------------------
# Notifier with multiple backends
# ---------------------------------------------------------------------------


class TestNotifierMultiBackend:
    async def test_notify_calls_both_webhook_and_telegram(self) -> None:
        webhook_send = AsyncMock()
        telegram_send = AsyncMock()

        notifier = Notifier(
            webhook_url="http://example.com/hook",
            enabled=True,
            telegram_config=_telegram_cfg(),
        )
        # Patch the actual backend send methods
        notifier._backends[0].send = webhook_send
        notifier._backends[1].send = telegram_send

        await notifier.notify("trade", {"pair": "BTC/USDT", "action": "BUY"})

        webhook_send.assert_awaited_once_with("trade", {"pair": "BTC/USDT", "action": "BUY"})
        telegram_send.assert_awaited_once_with("trade", {"pair": "BTC/USDT", "action": "BUY"})

    async def test_notifier_webhook_only_backward_compatible(self) -> None:
        notifier = Notifier(webhook_url="http://example.com/hook", enabled=True)
        assert len(notifier._backends) == 1
        assert isinstance(notifier._backends[0], WebhookBackend)
        assert notifier.telegram is None

    async def test_failed_backend_does_not_block_others(self) -> None:
        backend1 = AsyncMock()
        backend1.send = AsyncMock(side_effect=RuntimeError("backend1 error"))
        backend2 = AsyncMock()
        backend2.send = AsyncMock()

        notifier = Notifier(enabled=True)
        notifier._backends = [backend1, backend2]
        notifier._enabled = True

        # Should not raise even though backend1 fails
        await notifier.notify("trade", {"pair": "BTC/USDT"})
        backend2.send.assert_awaited_once()

    async def test_disabled_notifier_skips_all_backends(self) -> None:
        backend = AsyncMock()
        backend.send = AsyncMock()

        notifier = Notifier(enabled=False)
        notifier._backends = [backend]

        await notifier.notify("trade", {"pair": "BTC/USDT"})
        backend.send.assert_not_called()

    async def test_event_not_in_events_list_skipped(self) -> None:
        backend = AsyncMock()
        backend.send = AsyncMock()

        notifier = Notifier(enabled=True, events=["trade"])
        notifier._backends = [backend]

        await notifier.notify("price_trigger", {"pair": "BTC/USDT"})
        backend.send.assert_not_called()

    async def test_telegram_not_added_when_disabled(self) -> None:
        cfg = _telegram_cfg(enabled=False)
        notifier = Notifier(webhook_url="http://example.com", telegram_config=cfg)
        assert notifier.telegram is None
        assert len(notifier._backends) == 1

    async def test_telegram_not_added_when_token_empty(self) -> None:
        cfg = TelegramConfig(bot_token="", chat_id="12345", enabled=True)
        notifier = Notifier(webhook_url="http://example.com", telegram_config=cfg)
        assert notifier.telegram is None


# ---------------------------------------------------------------------------
# WebhookBackend.send
# ---------------------------------------------------------------------------


class TestWebhookBackendSend:
    async def test_send_posts_correct_payload(self) -> None:
        backend = WebhookBackend("http://hooks.example.com/test")
        mock_client = _mock_http_client()
        with patch("httpx.AsyncClient", return_value=mock_client):
            await backend.send("price_trigger", {"pair": "BTC/USDT", "trigger_reason": "test"})

        mock_client.post.assert_awaited_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://hooks.example.com/test"
        body = call_args[1]["json"]
        assert body["event"] == "price_trigger"
        assert body["pair"] == "BTC/USDT"

    async def test_send_uses_configured_timeout(self) -> None:
        backend = WebhookBackend("http://hooks.example.com/test", timeout=15)
        mock_client = _mock_http_client()
        with patch("httpx.AsyncClient", return_value=mock_client) as cls_mock:
            await backend.send("trade", {})
        cls_mock.assert_called_once_with(timeout=15)

    async def test_send_includes_all_data_keys(self) -> None:
        backend = WebhookBackend("http://example.com")
        mock_client = _mock_http_client()
        data = {"pair": "ETH/USDT", "action": "SELL", "price": 3000.0}
        with patch("httpx.AsyncClient", return_value=mock_client):
            await backend.send("trade", data)

        body = mock_client.post.call_args[1]["json"]
        for key in ("event", "pair", "action", "price"):
            assert key in body

    async def test_notifier_no_backends_does_nothing(self) -> None:
        notifier = Notifier(enabled=True)
        # No backends — notify() should return without error
        await notifier.notify("trade", {"pair": "BTC/USDT"})


# ---------------------------------------------------------------------------
# Notifier.telegram property
# ---------------------------------------------------------------------------


class TestNotifierTelegramProperty:
    def test_telegram_property_returns_backend(self) -> None:
        cfg = _telegram_cfg()
        notifier = Notifier(telegram_config=cfg)
        assert notifier.telegram is not None
        assert isinstance(notifier.telegram, TelegramBackend)

    def test_telegram_property_returns_none_without_config(self) -> None:
        notifier = Notifier()
        assert notifier.telegram is None

    def test_format_message_large_data_truncated(self) -> None:
        big_data = {"key": "x" * 1000}
        msg = TelegramBackend._format_message("custom_event", big_data)
        # The message is truncated at 500 chars inside json.dumps
        assert len(msg) <= 600  # some slack for the header
        # Check that the json output inside was indeed applied
        parsed_inner = msg.split("\n", 1)[1] if "\n" in msg else msg
        # The truncation happens on json.dumps(...) output
        assert len(parsed_inner) <= 510

    def test_format_message_encodes_non_serializable(self) -> None:
        from datetime import datetime

        from cryptotrader._compat import UTC

        data = {"ts": datetime(2026, 1, 1, tzinfo=UTC)}
        # Should not raise due to default=str
        msg = TelegramBackend._format_message("event", data)
        assert "2026" in msg


# ---------------------------------------------------------------------------
# json module used correctly in _format_message
# ---------------------------------------------------------------------------


class TestFormatMessageJsonFallback:
    def test_fallback_uses_default_str(self) -> None:
        from datetime import datetime

        from cryptotrader._compat import UTC

        data = {"created_at": datetime(2026, 4, 17, 12, 0, tzinfo=UTC)}
        msg = TelegramBackend._format_message("some_event", data)
        assert "2026-04-17" in msg

    def test_fallback_truncates_at_500_chars(self) -> None:
        data = {"payload": "A" * 600}
        msg = TelegramBackend._format_message("event", data)
        # json.dumps produces {"payload": "AAA...AAA"} which is >500, slice applied
        json_part = msg.split("\n", 1)[1]
        assert len(json_part) <= 500
