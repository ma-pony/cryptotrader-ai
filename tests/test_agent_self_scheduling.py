"""Tests for agent self-scheduling via verdict follow-up (T037).

Validates schedule_depth limit, TTL capping, and rule creation through
_process_schedule_follow_up in nodes/verdict.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from cryptotrader._compat import UTC


def _make_state(
    pair: str = "BTC/USDT",
    schedule_depth: int = 0,
    database_url: str = "sqlite+aiosqlite:///test.db",
) -> dict:
    return {
        "messages": [],
        "data": {},
        "metadata": {
            "pair": pair,
            "engine": "paper",
            "exchange_id": "binance",
            "schedule_depth": schedule_depth,
            "database_url": database_url,
        },
        "debate_round": 0,
        "max_debate_rounds": 2,
        "divergence_scores": [],
    }


def _follow_up(
    *,
    trigger_type: str = "price_threshold",
    parameters: dict | None = None,
    ttl_hours: int = 24,
    cooldown_minutes: int = 60,
    name: str = "agent-follow-up-BTC/USDT",
) -> dict:
    return {
        "trigger_type": trigger_type,
        "parameters": parameters or {"direction": "below", "price": 50_000},
        "ttl_hours": ttl_hours,
        "cooldown_minutes": cooldown_minutes,
        "name": name,
    }


# ---------------------------------------------------------------------------
# Depth limit
# ---------------------------------------------------------------------------


class TestDepthLimit:
    async def test_depth_below_limit_creates_rule(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state(schedule_depth=0)
        verdict_data = {"schedule_follow_up": _follow_up()}

        mock_store = AsyncMock()
        mock_store.create_rule = AsyncMock()

        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session"),
            patch("cryptotrader.triggers.store.TriggerRuleStore", return_value=mock_store),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

        mock_store.create_rule.assert_awaited_once()
        call_data = mock_store.create_rule.call_args[0][0]
        assert call_data["schedule_depth"] == 1

    async def test_depth_at_limit_skips(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state(schedule_depth=3)
        verdict_data = {"schedule_follow_up": _follow_up()}

        mock_store = AsyncMock()
        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session"),
            patch("cryptotrader.triggers.store.TriggerRuleStore", return_value=mock_store),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

        mock_store.create_rule.assert_not_called()

    async def test_depth_above_limit_skips(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state(schedule_depth=5)
        verdict_data = {"schedule_follow_up": _follow_up()}

        mock_store = AsyncMock()
        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session"),
            patch("cryptotrader.triggers.store.TriggerRuleStore", return_value=mock_store),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

        mock_store.create_rule.assert_not_called()

    async def test_depth_increments_by_one(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state(schedule_depth=2)
        verdict_data = {"schedule_follow_up": _follow_up()}

        mock_store = AsyncMock()
        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session"),
            patch("cryptotrader.triggers.store.TriggerRuleStore", return_value=mock_store),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

        call_data = mock_store.create_rule.call_args[0][0]
        assert call_data["schedule_depth"] == 3


# ---------------------------------------------------------------------------
# TTL capping
# ---------------------------------------------------------------------------


class TestTTLCapping:
    async def test_ttl_capped_at_72_hours(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state(schedule_depth=0)
        verdict_data = {"schedule_follow_up": _follow_up(ttl_hours=200)}

        mock_store = AsyncMock()
        before = datetime.now(UTC)
        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session"),
            patch("cryptotrader.triggers.store.TriggerRuleStore", return_value=mock_store),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

        call_data = mock_store.create_rule.call_args[0][0]
        ttl_expires = call_data["ttl_expires_at"]
        max_expires = before + timedelta(hours=73)
        assert ttl_expires <= max_expires

    async def test_ttl_respects_lower_value(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state(schedule_depth=0)
        verdict_data = {"schedule_follow_up": _follow_up(ttl_hours=6)}

        mock_store = AsyncMock()
        before = datetime.now(UTC)
        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session"),
            patch("cryptotrader.triggers.store.TriggerRuleStore", return_value=mock_store),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

        call_data = mock_store.create_rule.call_args[0][0]
        ttl_expires = call_data["ttl_expires_at"]
        max_expires = before + timedelta(hours=7)
        assert ttl_expires <= max_expires


# ---------------------------------------------------------------------------
# No follow-up
# ---------------------------------------------------------------------------


class TestNoFollowUp:
    async def test_none_follow_up_is_noop(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state()
        await _process_schedule_follow_up(state, {"action": "buy"})

    async def test_empty_dict_follow_up_is_noop(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state()
        await _process_schedule_follow_up(state, {"schedule_follow_up": {}})

    async def test_non_dict_follow_up_is_noop(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state()
        await _process_schedule_follow_up(state, {"schedule_follow_up": "not a dict"})


# ---------------------------------------------------------------------------
# Rule field propagation
# ---------------------------------------------------------------------------


class TestRuleFields:
    async def test_created_by_is_agent(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state()
        verdict_data = {"schedule_follow_up": _follow_up()}

        mock_store = AsyncMock()
        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session"),
            patch("cryptotrader.triggers.store.TriggerRuleStore", return_value=mock_store),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

        call_data = mock_store.create_rule.call_args[0][0]
        assert call_data["created_by"] == "agent"

    async def test_pair_from_state_metadata(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state(pair="ETH/USDT")
        verdict_data = {"schedule_follow_up": _follow_up()}

        mock_store = AsyncMock()
        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session"),
            patch("cryptotrader.triggers.store.TriggerRuleStore", return_value=mock_store),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

        call_data = mock_store.create_rule.call_args[0][0]
        assert call_data["pair"] == "ETH/USDT"

    async def test_trigger_type_from_follow_up(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state()
        verdict_data = {"schedule_follow_up": _follow_up(trigger_type="pct_change")}

        mock_store = AsyncMock()
        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session"),
            patch("cryptotrader.triggers.store.TriggerRuleStore", return_value=mock_store),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

        call_data = mock_store.create_rule.call_args[0][0]
        assert call_data["trigger_type"] == "pct_change"

    async def test_cooldown_from_follow_up(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state()
        verdict_data = {"schedule_follow_up": _follow_up(cooldown_minutes=120)}

        mock_store = AsyncMock()
        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session"),
            patch("cryptotrader.triggers.store.TriggerRuleStore", return_value=mock_store),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

        call_data = mock_store.create_rule.call_args[0][0]
        assert call_data["cooldown_minutes"] == 120


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_db_error_does_not_raise(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state()
        verdict_data = {"schedule_follow_up": _follow_up()}

        with (
            patch("cryptotrader.config.load_config") as mock_cfg,
            patch("cryptotrader.db.get_async_session", side_effect=RuntimeError("db down")),
        ):
            mock_cfg.return_value.infrastructure.database_url = "sqlite+aiosqlite:///test.db"
            await _process_schedule_follow_up(state, verdict_data)

    async def test_empty_database_url_skips(self) -> None:
        from cryptotrader.nodes.verdict import _process_schedule_follow_up

        state = _make_state()
        verdict_data = {"schedule_follow_up": _follow_up()}

        with patch("cryptotrader.config.load_config") as mock_cfg:
            mock_cfg.return_value.infrastructure.database_url = ""
            await _process_schedule_follow_up(state, verdict_data)
