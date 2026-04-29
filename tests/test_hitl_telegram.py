"""Tests for HITL Telegram Bot — send, callback, update."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_telegram():
    """Patch the telegram module so tests don't need python-telegram-bot installed."""
    mock_tg = MagicMock()

    mock_bot_instance = AsyncMock()
    mock_bot_instance.send_message = AsyncMock()
    mock_bot_instance.edit_message_text = AsyncMock()
    mock_bot_instance.__aenter__ = AsyncMock(return_value=mock_bot_instance)
    mock_bot_instance.__aexit__ = AsyncMock(return_value=False)

    mock_tg.Bot.return_value = mock_bot_instance
    mock_tg.InlineKeyboardMarkup = MagicMock()
    mock_tg.InlineKeyboardButton = MagicMock()

    with patch.dict("sys.modules", {"telegram": mock_tg, "telegram.ext": MagicMock()}):
        yield mock_tg, mock_bot_instance


@pytest.fixture
def tg_config():
    from cryptotrader.config import HitlTelegramConfig

    return HitlTelegramConfig(enabled=True, bot_token="test-token", chat_id="123")


@pytest.mark.asyncio
async def test_send_approval_notification(mock_telegram, tg_config):
    _mock_tg, mock_bot = mock_telegram

    mock_msg = MagicMock()
    mock_msg.message_id = 42
    mock_bot.send_message.return_value = mock_msg

    from cryptotrader.hitl.telegram import send_approval_notification

    result = await send_approval_notification(
        approval_id="test-001",
        pair="BTC/USDT",
        trigger_reason="position_scale",
        verdict={"action": "long", "position_scale": 0.8, "confidence": 0.7, "reasoning": "test"},
        tg_config=tg_config,
    )

    assert result == 42
    mock_bot.send_message.assert_awaited_once()
    call_kwargs = mock_bot.send_message.call_args
    assert call_kwargs.kwargs["chat_id"] == "123"
    assert "BTC/USDT" in call_kwargs.kwargs["text"]


@pytest.mark.asyncio
async def test_send_failure(mock_telegram, tg_config):
    _mock_tg, mock_bot = mock_telegram
    mock_bot.send_message.side_effect = RuntimeError("network error")

    from cryptotrader.hitl.telegram import send_approval_notification

    result = await send_approval_notification(
        approval_id="test-002",
        pair="ETH/USDT",
        trigger_reason="divergence",
        verdict={"action": "short"},
        tg_config=tg_config,
    )
    assert result is None


@pytest.mark.asyncio
async def test_update_message_approve(mock_telegram, tg_config):
    _mock_tg, mock_bot = mock_telegram

    from cryptotrader.hitl.telegram import update_telegram_message

    await update_telegram_message(tg_config, message_id=42, decision="approve", decision_by="web")

    mock_bot.edit_message_text.assert_awaited_once()
    call_kwargs = mock_bot.edit_message_text.call_args
    assert "已批准" in call_kwargs.kwargs["text"]


@pytest.mark.asyncio
async def test_update_message_expired(mock_telegram, tg_config):
    _mock_tg, mock_bot = mock_telegram

    from cryptotrader.hitl.telegram import update_telegram_message

    await update_telegram_message(tg_config, message_id=42, decision="expired", decision_by="timeout")

    call_kwargs = mock_bot.edit_message_text.call_args
    assert "已超时" in call_kwargs.kwargs["text"]


@pytest.mark.asyncio
async def test_callback_approve(mock_telegram):
    _mock_tg, _mock_bot = mock_telegram

    with (
        patch("cryptotrader.hitl.store.ApprovalStore") as mock_store,
        patch("cryptotrader.hitl.notifier.notify_hitl_decision", new_callable=AsyncMock),
    ):
        mock_store.decide = AsyncMock(return_value=True)
        mock_store.get = AsyncMock(return_value={"pair": "BTC/USDT", "trigger_reason": "position_scale"})

        from cryptotrader.hitl.telegram import TelegramApprovalBot

        bot = TelegramApprovalBot(bot_token="tok", chat_id="123", db_url="sqlite:///test.db")

        mock_query = AsyncMock()
        mock_query.data = "hitl:test-001:approve"
        mock_update = MagicMock()
        mock_update.callback_query = mock_query

        await bot._handle_callback(mock_update, None)

        mock_query.answer.assert_awaited_with("Decision recorded")
        mock_query.edit_message_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_callback_reject(mock_telegram):
    _mock_tg, _mock_bot = mock_telegram

    with (
        patch("cryptotrader.hitl.store.ApprovalStore") as mock_store,
        patch("cryptotrader.hitl.notifier.notify_hitl_decision", new_callable=AsyncMock),
    ):
        mock_store.decide = AsyncMock(return_value=True)
        mock_store.get = AsyncMock(return_value={"pair": "ETH/USDT", "trigger_reason": "divergence"})

        from cryptotrader.hitl.telegram import TelegramApprovalBot

        bot = TelegramApprovalBot(bot_token="tok", chat_id="123", db_url="sqlite:///test.db")

        mock_query = AsyncMock()
        mock_query.data = "hitl:test-002:reject"
        mock_update = MagicMock()
        mock_update.callback_query = mock_query

        await bot._handle_callback(mock_update, None)

        mock_query.answer.assert_awaited_with("Decision recorded")
        text = mock_query.edit_message_text.call_args.kwargs["text"]
        assert "已拒绝" in text


@pytest.mark.asyncio
async def test_callback_already_decided(mock_telegram):
    _mock_tg, _mock_bot = mock_telegram

    with patch("cryptotrader.hitl.store.ApprovalStore") as mock_store:
        mock_store.decide = AsyncMock(return_value=False)

        from cryptotrader.hitl.telegram import TelegramApprovalBot

        bot = TelegramApprovalBot(bot_token="tok", chat_id="123", db_url="sqlite:///test.db")

        mock_query = AsyncMock()
        mock_query.data = "hitl:test-003:approve"
        mock_update = MagicMock()
        mock_update.callback_query = mock_query

        await bot._handle_callback(mock_update, None)

        mock_query.answer.assert_awaited_with("Already decided or expired")
        mock_query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_invalid_data(mock_telegram):
    _mock_tg, _mock_bot = mock_telegram

    from cryptotrader.hitl.telegram import TelegramApprovalBot

    bot = TelegramApprovalBot(bot_token="tok", chat_id="123", db_url="sqlite:///test.db")

    mock_query = AsyncMock()
    mock_query.data = "invalid:data"
    mock_update = MagicMock()
    mock_update.callback_query = mock_query

    await bot._handle_callback(mock_update, None)

    mock_query.answer.assert_awaited_with("Invalid callback")
