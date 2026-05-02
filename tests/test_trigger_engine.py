"""Tests for PriceTriggerEngine (T009).

Mocks WebSocket, Redis state, and HTTP calls. Tests lifecycle, rule evaluation,
dispatch with cooldown, and utility methods.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.config import TriggersConfig
from cryptotrader.triggers.engine import PriceTriggerEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rule(
    *,
    rule_id: str = "rule-1",
    name: str = "Test Rule",
    trigger_type: str = "price_threshold",
    pair: str = "BTC/USDT",
    parameters: dict | None = None,
    cooldown_minutes: int = 30,
    enabled: bool = True,
    schedule_depth: int = 0,
) -> MagicMock:
    rule = MagicMock()
    rule.id = rule_id
    rule.name = name
    rule.trigger_type = trigger_type
    rule.pair = pair
    rule.parameters = parameters or {"direction": "below", "price": 50_000.0}
    rule.cooldown_minutes = cooldown_minutes
    rule.enabled = enabled
    rule.schedule_depth = schedule_depth
    return rule


def _make_config(**kwargs) -> TriggersConfig:
    cfg = TriggersConfig()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_engine(
    rules: list | None = None,
    run_cb: Any = None,
    config: TriggersConfig | None = None,
) -> tuple[PriceTriggerEngine, MagicMock, MagicMock]:
    store = AsyncMock()
    store.list_rules = AsyncMock(return_value=rules or [])
    store.record_event = AsyncMock(return_value=MagicMock(id="event-1"))

    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()

    cb = run_cb or AsyncMock()
    cfg = config or _make_config()

    engine = PriceTriggerEngine(store, redis, cb, cfg)
    return engine, store, redis


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_sets_running_flag(self) -> None:
        engine, _store, _ = _make_engine()
        with patch("cryptotrader.triggers.engine.asyncio.create_task", return_value=MagicMock()):
            await engine.start()
        assert engine._running is True

    async def test_start_calls_reload_rules(self) -> None:
        engine, store, _ = _make_engine()
        with patch("cryptotrader.triggers.engine.asyncio.create_task", return_value=MagicMock()):
            await engine.start()
        store.list_rules.assert_awaited_once_with(enabled_only=True)

    async def test_start_twice_is_idempotent(self) -> None:
        engine, store, _ = _make_engine()
        with patch("cryptotrader.triggers.engine.asyncio.create_task", return_value=MagicMock()):
            await engine.start()
            await engine.start()
        assert store.list_rules.await_count == 1

    async def test_stop_clears_running_flag(self) -> None:
        engine, _, _ = _make_engine()
        engine._running = True
        task = asyncio.create_task(asyncio.sleep(0))
        engine._ws_task = task
        await engine.stop()
        assert engine._running is False

    async def test_stop_cancels_ws_task(self) -> None:
        engine, _, _ = _make_engine()
        engine._running = True

        async def _long_sleep():
            await asyncio.sleep(100)

        task = asyncio.create_task(_long_sleep())
        engine._ws_task = task
        await asyncio.sleep(0)
        await engine.stop()
        assert task.cancelled()

    async def test_stop_when_not_running_is_safe(self) -> None:
        engine, _, _ = _make_engine()
        engine._running = False
        await engine.stop()  # Should not raise


# ---------------------------------------------------------------------------
# reload_rules
# ---------------------------------------------------------------------------


class TestReloadRules:
    async def test_reload_populates_rules(self) -> None:
        rules = [_make_rule(pair="BTC/USDT"), _make_rule(rule_id="rule-2", pair="ETH/USDT")]
        engine, _store, _ = _make_engine(rules=rules)
        await engine.reload_rules()
        assert len(engine._rules) == 2

    async def test_reload_calls_store_list_rules(self) -> None:
        engine, store, _ = _make_engine()
        await engine.reload_rules()
        store.list_rules.assert_awaited_once_with(enabled_only=True)

    async def test_reload_replaces_previous_rules(self) -> None:
        engine, store, _ = _make_engine(rules=[_make_rule()])
        await engine.reload_rules()
        store.list_rules = AsyncMock(return_value=[])
        await engine.reload_rules()
        assert engine._rules == []


# ---------------------------------------------------------------------------
# _evaluate_rule — dispatch to correct condition function
# ---------------------------------------------------------------------------


class TestEvaluateRule:
    def test_price_threshold_dispatches_correctly(self) -> None:
        engine, _, _ = _make_engine()
        rule = _make_rule(trigger_type="price_threshold", parameters={"direction": "below", "price": 50_000.0})
        # price below threshold => True (signature: rule, current_price, now)
        assert engine._evaluate_rule(rule, 49_000.0, 0.0) is True
        assert engine._evaluate_rule(rule, 51_000.0, 0.0) is False

    def test_pct_change_dispatches_correctly(self) -> None:
        # Reference now comes from the rolling price buffer instead of the
        # previous tick, so the test must seed the buffer with the reference
        # price BEFORE the window cutoff.
        engine, _, _ = _make_engine()
        rule = _make_rule(
            trigger_type="pct_change",
            parameters={"window_minutes": 15, "threshold_pct": 5.0},
        )
        now = 10_000_000.0
        engine._append_price_buffer("BTC/USDT", now - 16 * 60, 100.0)
        # 10% drop relative to 15-min-ago price => True
        assert engine._evaluate_rule(rule, 90.0, now) is True
        # 1% change => False
        assert engine._evaluate_rule(rule, 101.0, now) is False

    def test_candle_pattern_dispatches_correctly(self) -> None:
        engine, _, _ = _make_engine()
        rule = _make_rule(
            trigger_type="candle_pattern",
            parameters={"interval": "1h", "consecutive_count": 2, "direction": "bearish"},
        )
        engine._klines[("BTC/USDT", "1h")] = [
            {"open": 100.0, "close": 99.0},
            {"open": 99.0, "close": 98.0},
        ]
        assert engine._evaluate_rule(rule, 98.0, 0.0) is True

    def test_funding_rate_dispatches_correctly(self) -> None:
        engine, _, _ = _make_engine()
        rule = _make_rule(trigger_type="funding_rate", parameters={"threshold_pct": 0.1})
        engine._funding_rates["BTC/USDT"] = 0.003  # 0.3% > 0.1%
        assert engine._evaluate_rule(rule, 50_000.0, 0.0) is True

    def test_unknown_trigger_type_returns_false(self) -> None:
        engine, _, _ = _make_engine()
        rule = _make_rule(trigger_type="unknown_type")
        assert engine._evaluate_rule(rule, 100.0, 100.0) is False


# ---------------------------------------------------------------------------
# _dispatch — event recording and callback invocation
# ---------------------------------------------------------------------------


class TestDispatch:
    async def test_dispatch_records_event_and_calls_callback(self) -> None:
        callback = AsyncMock()
        engine, store, redis = _make_engine(run_cb=callback)
        redis.get = AsyncMock(return_value=None)  # not in cooldown
        rule = _make_rule()
        snapshot = {"pair": "BTC/USDT", "price": 49_000.0, "ts": 0.0}

        await engine._dispatch(rule, snapshot)

        store.record_event.assert_awaited_once()
        call_kwargs = store.record_event.call_args[0][0]
        assert call_kwargs["cooldown_skipped"] is False
        callback.assert_awaited_once_with("BTC/USDT", {"trigger_event_id": "event-1", "schedule_depth": 0})

    async def test_dispatch_in_cooldown_records_skipped_event(self) -> None:
        callback = AsyncMock()
        engine, store, redis = _make_engine(run_cb=callback)
        redis.get = AsyncMock(return_value="1")  # cooldown active
        rule = _make_rule()
        snapshot = {"pair": "BTC/USDT", "price": 49_000.0, "ts": 0.0}

        await engine._dispatch(rule, snapshot)

        store.record_event.assert_awaited_once()
        call_kwargs = store.record_event.call_args[0][0]
        assert call_kwargs["cooldown_skipped"] is True
        callback.assert_not_called()

    async def test_dispatch_sets_cooldown_key(self) -> None:
        engine, _store, redis = _make_engine()
        redis.get = AsyncMock(return_value=None)
        rule = _make_rule()
        snapshot = {"pair": "BTC/USDT", "price": 49_000.0, "ts": 0.0}

        await engine._dispatch(rule, snapshot)

        redis.set.assert_awaited_once_with("trigger:cooldown:rule-1", "1", ex=30 * 60)

    async def test_dispatch_callback_failure_does_not_raise(self) -> None:
        async def _failing_cb(pair, data):
            raise RuntimeError("callback failed")

        engine, _store, redis = _make_engine(run_cb=_failing_cb)
        redis.get = AsyncMock(return_value=None)
        rule = _make_rule()
        snapshot = {"pair": "BTC/USDT", "price": 49_000.0, "ts": 0.0}

        # Should not raise
        await engine._dispatch(rule, snapshot)


# ---------------------------------------------------------------------------
# _check_cooldown / _set_cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    async def test_check_cooldown_returns_true_when_key_exists(self) -> None:
        engine, _, redis = _make_engine()
        redis.get = AsyncMock(return_value="1")
        assert await engine._check_cooldown("some_key") is True

    async def test_check_cooldown_returns_false_when_key_absent(self) -> None:
        engine, _, redis = _make_engine()
        redis.get = AsyncMock(return_value=None)
        assert await engine._check_cooldown("some_key") is False

    async def test_set_cooldown_calls_redis_set(self) -> None:
        engine, _, redis = _make_engine()
        await engine._set_cooldown("cooldown:rule-99", 15)
        redis.set.assert_awaited_once_with("cooldown:rule-99", "1", ex=15 * 60)


# ---------------------------------------------------------------------------
# poll_funding_rates
# ---------------------------------------------------------------------------


class TestPollFundingRates:
    async def test_poll_updates_funding_rates(self) -> None:
        engine, _, _ = _make_engine(rules=[_make_rule(pair="BTC/USDT")])
        engine._rules = [_make_rule(pair="BTC/USDT")]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value=[
                {"symbol": "BTCUSDT", "lastFundingRate": "0.0003"},
                {"symbol": "ETHUSDT", "lastFundingRate": "0.0001"},
            ]
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            rates = await engine.poll_funding_rates()

        assert "BTC/USDT" in rates
        assert abs(rates["BTC/USDT"] - 0.0003) < 1e-9

    async def test_poll_handles_http_error_gracefully(self) -> None:
        engine, _, _ = _make_engine(rules=[_make_rule(pair="BTC/USDT")])
        engine._rules = [_make_rule(pair="BTC/USDT")]

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=OSError("network error"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            rates = await engine.poll_funding_rates()

        # Should return empty dict and not raise
        assert isinstance(rates, dict)


# ---------------------------------------------------------------------------
# _symbol_to_pair (static)
# ---------------------------------------------------------------------------


class TestSymbolToPair:
    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("BTCUSDT", "BTC/USDT"),
            ("ETHUSDT", "ETH/USDT"),
            ("SOLUSDT", "SOL/USDT"),
            ("BTCBUSD", "BTC/BUSD"),
            ("ETHBUSD", "ETH/BUSD"),
            ("XRPUSDT", "XRP/USDT"),
            # Unknown suffix returned as-is (uppercased)
            ("BTCEUR", "BTCEUR"),
        ],
    )
    def test_symbol_to_pair(self, symbol: str, expected: str) -> None:
        assert PriceTriggerEngine._symbol_to_pair(symbol) == expected


# ---------------------------------------------------------------------------
# _build_trigger_reason
# ---------------------------------------------------------------------------


class TestBuildTriggerReason:
    def test_price_threshold_below(self) -> None:
        rule = _make_rule(
            trigger_type="price_threshold",
            parameters={"direction": "below", "price": 50_000},
        )
        snapshot = {"pair": "BTC/USDT", "price": 49_000.0}
        reason = PriceTriggerEngine._build_trigger_reason(rule, snapshot)
        assert "BTC/USDT" in reason
        assert "49,000.00" in reason

    def test_price_threshold_above(self) -> None:
        rule = _make_rule(
            trigger_type="price_threshold",
            parameters={"direction": "above", "price": 60_000},
        )
        snapshot = {"pair": "BTC/USDT", "price": 61_000.0}
        reason = PriceTriggerEngine._build_trigger_reason(rule, snapshot)
        assert "BTC/USDT" in reason

    def test_pct_change_reason(self) -> None:
        rule = _make_rule(
            trigger_type="pct_change",
            parameters={"threshold_pct": 5.0, "window_minutes": 60},
        )
        snapshot = {"pair": "BTC/USDT", "price": 45_000.0}
        reason = PriceTriggerEngine._build_trigger_reason(rule, snapshot)
        assert "60" in reason
        assert "5.0%" in reason

    def test_candle_pattern_reason(self) -> None:
        rule = _make_rule(
            trigger_type="candle_pattern",
            parameters={"candle_count": 3, "direction": "bearish", "timeframe": "1h"},
        )
        snapshot = {"pair": "BTC/USDT", "price": 48_000.0}
        reason = PriceTriggerEngine._build_trigger_reason(rule, snapshot)
        assert "3" in reason
        assert "1h" in reason

    def test_funding_rate_reason(self) -> None:
        rule = _make_rule(
            trigger_type="funding_rate",
            parameters={"threshold_pct": 0.1},
        )
        snapshot = {"pair": "BTC/USDT", "price": 50_000.0}
        reason = PriceTriggerEngine._build_trigger_reason(rule, snapshot)
        assert "funding rate" in reason
        assert "0.1%" in reason

    def test_fallback_reason(self) -> None:
        rule = _make_rule(trigger_type="custom_type")
        snapshot = {"pair": "BTC/USDT", "price": 50_000.0}
        reason = PriceTriggerEngine._build_trigger_reason(rule, snapshot)
        assert "BTC/USDT" in reason
        assert "Test Rule" in reason
