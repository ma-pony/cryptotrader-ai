"""Tests for learning/verbal.py and hitl/notifier.py."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader._compat import UTC
from cryptotrader.learning.verbal import format_experience_text, get_experience

# ── get_experience ──


class TestGetExperience:
    @pytest.mark.asyncio
    async def test_regime_based_search(self):
        store = MagicMock()
        cases = [MagicMock()]
        with patch("cryptotrader.learning.verbal.search_by_regime", new_callable=AsyncMock, return_value=cases):
            result = await get_experience(
                store,
                {"funding_rate": 0.01},
                regime_tags=["bullish"],
                thresholds=MagicMock(),
            )
        assert result == cases

    @pytest.mark.asyncio
    async def test_regime_search_empty_fallback(self):
        store = MagicMock()
        fallback_cases = [MagicMock()]
        with (
            patch("cryptotrader.learning.verbal.search_by_regime", new_callable=AsyncMock, return_value=[]),
            patch("cryptotrader.learning.verbal.search_similar", new_callable=AsyncMock, return_value=fallback_cases),
        ):
            result = await get_experience(
                store,
                {"funding_rate": 0.01, "volatility": 0.03},
                regime_tags=["bullish"],
                thresholds=MagicMock(),
            )
        assert result == fallback_cases

    @pytest.mark.asyncio
    async def test_no_regime_tags(self):
        store = MagicMock()
        with patch("cryptotrader.learning.verbal.search_similar", new_callable=AsyncMock, return_value=[]):
            result = await get_experience(store, {"funding_rate": 0.01})
        assert result == []


# ── format_experience_text ──


class TestFormatExperienceText:
    @pytest.mark.asyncio
    async def test_empty(self):
        assert await format_experience_text([]) == ""

    @pytest.mark.asyncio
    async def test_with_cases(self):
        dc = MagicMock()
        dc.pair = "BTC/USDT"
        dc.timestamp = datetime(2025, 1, 15, tzinfo=UTC)
        dc.pnl = 0.05
        dc.verdict.action = "long"
        dc.retrospective = "Good entry"
        result = await format_experience_text([dc])
        assert "BTC/USDT" in result
        assert "pnl=0.05" in result
        assert "Good entry" in result

    @pytest.mark.asyncio
    async def test_no_pnl(self):
        dc = MagicMock()
        dc.pair = "ETH/USDT"
        dc.timestamp = datetime(2025, 1, 15, tzinfo=UTC)
        dc.pnl = None
        dc.verdict.action = "short"
        dc.retrospective = ""
        result = await format_experience_text([dc])
        assert "no outcome yet" in result

    @pytest.mark.asyncio
    async def test_no_verdict(self):
        dc = MagicMock()
        dc.pair = "SOL/USDT"
        dc.timestamp = datetime(2025, 1, 15, tzinfo=UTC)
        dc.pnl = -0.02
        dc.verdict = None
        dc.retrospective = None
        result = await format_experience_text([dc])
        assert "hold" in result


# ── hitl/notifier.py ──


class TestHitlNotifier:
    @pytest.mark.asyncio
    async def test_notify_request_telegram_disabled(self):
        from cryptotrader.hitl.notifier import notify_hitl_request

        cfg = MagicMock()
        cfg.telegram.enabled = False
        cfg.telegram.bot_token = ""
        await notify_hitl_request("id1", "BTC/USDT", "divergence", {}, {}, cfg)

    @pytest.mark.asyncio
    async def test_notify_request_telegram_error(self):
        from cryptotrader.hitl.notifier import notify_hitl_request

        cfg = MagicMock()
        cfg.telegram.enabled = True
        cfg.telegram.bot_token = "test-token"
        with (
            patch("cryptotrader.hitl.notifier.send_approval_notification", side_effect=Exception("fail"), create=True),
            patch(
                "cryptotrader.hitl.telegram.send_approval_notification",
                new_callable=AsyncMock,
                side_effect=Exception("fail"),
            ),
        ):
            await notify_hitl_request("id1", "BTC/USDT", "divergence", {}, {}, cfg)

    @pytest.mark.asyncio
    async def test_notify_decision(self):
        from cryptotrader.hitl.notifier import notify_hitl_decision

        await notify_hitl_decision("id1", "BTC/USDT", "divergence", "approved", "auto", 5.0)
