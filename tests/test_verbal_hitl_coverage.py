"""Tests for hitl/notifier.py.

2026-05-13: learning/verbal.py removed entirely (along with its tests),
because verbal-reinforcement historical-case injection re-introduced the
kind of prior anchoring round-3 minimal skills had just deleted. The
hitl notifier tests are independent and remain.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
