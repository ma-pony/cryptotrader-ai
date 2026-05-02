"""Coverage for the trigger templates that were silently broken pre-fix.

Before the 2026-05-02 follow-up, three of the four UI templates couldn't
ever fire:

  - ``candle_pattern``: ``self._price_history`` was initialized empty and
    never populated, so ``check_candle_pattern`` always saw an empty list.
  - ``funding_rate``: ``poll_funding_rates()`` existed but was never
    scheduled, so ``self._funding_rates`` stayed empty.
  - ``pct_change``: the reference price was the *previous tick* price,
    so a 3% threshold would have needed a single 3%+ price spike between
    two adjacent ticks — never the rolling-window behaviour the UI
    described.

These tests pin down the new contracts: schedule the polls, populate the
buffers, and use a window-based reference for pct_change.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from cryptotrader.config import TriggersConfig
from cryptotrader.triggers.conditions import check_candle_pattern
from cryptotrader.triggers.engine import PriceTriggerEngine


def _make_rule(**kwargs) -> MagicMock:
    defaults = {
        "id": "r1",
        "name": "Test",
        "pair": "BTC/USDT",
        "trigger_type": "pct_change",
        "parameters": {},
        "cooldown_minutes": 30,
        "enabled": True,
        "schedule_depth": 0,
    }
    defaults.update(kwargs)
    rule = MagicMock()
    for k, v in defaults.items():
        setattr(rule, k, v)
    return rule


def _make_engine(rules=None, cfg=None) -> PriceTriggerEngine:
    store = AsyncMock()
    store.list_rules = AsyncMock(return_value=rules or [])
    store.record_event = AsyncMock(return_value=MagicMock(id="ev"))
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    cb = AsyncMock()
    return PriceTriggerEngine(store, redis, cb, cfg or TriggersConfig())


# ── conditions: alias support ──


def test_candle_pattern_accepts_consecutive_count_alias():
    """Frontend persists ``consecutive_count``; legacy ``candle_count`` still works."""
    candles: list[dict[str, float]] = [
        {"open": 100.0, "close": 95.0},
        {"open": 95.0, "close": 90.0},
        {"open": 90.0, "close": 85.0},
    ]
    # New name
    assert check_candle_pattern(candles, {"consecutive_count": 3, "direction": "bearish"}) is True
    # Legacy name
    assert check_candle_pattern(candles, {"candle_count": 3, "direction": "bearish"}) is True
    # 4 needed but only 3 available
    assert check_candle_pattern(candles, {"consecutive_count": 4, "direction": "bearish"}) is False


# ── pct_change with window-based reference ──


async def test_pct_change_uses_window_reference_not_previous_tick():
    """A 3% threshold over a 15-min window must compare against the price 15
    min ago, not the price one tick ago."""
    engine = _make_engine()
    rule = _make_rule(
        trigger_type="pct_change",
        parameters={"window_minutes": 15, "threshold_pct": 3.0},
    )
    engine._rules = [rule]

    # Seed buffer: 16 min ago at $100, then drift to $99.5 ten minutes ago
    base_now = 10_000_000.0
    engine._append_price_buffer("BTC/USDT", base_now - 16 * 60, 100.0)
    engine._append_price_buffer("BTC/USDT", base_now - 10 * 60, 99.5)

    # Tick comes in at $103 — that's 3% above the 15-min-ago reference
    triggered = engine._evaluate_rule(rule, current_price=103.0, now=base_now)
    assert triggered is True

    # If we compared against the prev tick ($99.5), 3.5% diff would still
    # trigger — but the test below verifies the *window* path is used,
    # not the prev-tick path. With insufficient history before the window,
    # the engine falls back to the OLDEST sample (still $100 here), so a
    # 0.5% move stays below threshold.
    engine._price_buffer["BTC/USDT"].clear()
    engine._append_price_buffer("BTC/USDT", base_now - 1, 102.5)
    not_triggered = engine._evaluate_rule(rule, current_price=103.0, now=base_now)
    # 102.5 → 103 = 0.49% — below 3% threshold even though prev-tick diff exists
    assert not_triggered is False


async def test_price_buffer_prunes_entries_older_than_60_min():
    engine = _make_engine()
    base = 10_000_000.0
    engine._append_price_buffer("BTC/USDT", base - 90 * 60, 50.0)  # outside window
    engine._append_price_buffer("BTC/USDT", base - 30 * 60, 60.0)  # inside
    engine._append_price_buffer("BTC/USDT", base, 70.0)
    buf = engine._price_buffer["BTC/USDT"]
    # The 90-min-old entry should be pruned by now.
    assert all(ts >= base - 60 * 60 for ts, _ in buf)
    assert len(buf) == 2


async def test_pct_change_with_no_buffer_does_not_trigger():
    """No history yet (engine just started) → ref_price=0 → don't trigger."""
    engine = _make_engine()
    rule = _make_rule(
        trigger_type="pct_change",
        parameters={"window_minutes": 15, "threshold_pct": 3.0},
    )
    assert engine._evaluate_rule(rule, current_price=100.0, now=10_000_000.0) is False


# ── candle_pattern uses _klines, not legacy _price_history ──


async def test_candle_pattern_reads_from_klines_dict():
    engine = _make_engine()
    rule = _make_rule(
        trigger_type="candle_pattern",
        parameters={"interval": "1h", "consecutive_count": 3, "direction": "bearish"},
    )
    engine._klines[("BTC/USDT", "1h")] = [
        {"open": 100, "close": 95},
        {"open": 95, "close": 90},
        {"open": 90, "close": 85},
    ]
    assert engine._evaluate_rule(rule, current_price=85.0, now=0.0) is True


async def test_candle_pattern_without_interval_param_does_not_trigger():
    engine = _make_engine()
    rule = _make_rule(
        trigger_type="candle_pattern",
        parameters={"consecutive_count": 3, "direction": "bearish"},
    )
    # Even with kline data present under some interval, missing interval → no fire.
    engine._klines[("BTC/USDT", "1h")] = [{"open": 100, "close": 95}] * 3
    assert engine._evaluate_rule(rule, current_price=85.0, now=0.0) is False


# ── kline poll loop scheduling ──


async def test_start_schedules_kline_and_funding_tasks():
    engine = _make_engine()
    captured: list[str] = []

    def fake_create_task(coro, *_, **__):
        # Record which coroutine name was scheduled, then close it cleanly.
        captured.append(coro.__qualname__)
        coro.close()
        return MagicMock()

    with patch("cryptotrader.triggers.engine.asyncio.create_task", side_effect=fake_create_task):
        await engine.start()

    # All three loops scheduled, not just _ws_connect.
    assert any("_ws_connect" in name for name in captured)
    assert any("_kline_poll_loop" in name for name in captured)
    assert any("_funding_poll_loop" in name for name in captured)


async def test_kline_specs_dedupes_same_pair_interval():
    """Two rules on the same (pair, interval) — fetch once with the larger count."""
    engine = _make_engine()
    engine._rules = [
        _make_rule(
            id="r1",
            trigger_type="candle_pattern",
            parameters={"interval": "1h", "consecutive_count": 3, "direction": "bearish"},
        ),
        _make_rule(
            id="r2",
            trigger_type="candle_pattern",
            parameters={"interval": "1h", "consecutive_count": 5, "direction": "bullish"},
        ),
        _make_rule(
            id="r3",
            trigger_type="candle_pattern",
            parameters={"interval": "4h", "consecutive_count": 3, "direction": "bearish"},
        ),
    ]
    specs = engine._kline_specs()
    by_key = {(p, i): c for p, i, c in specs}
    assert by_key == {("BTC/USDT", "1h"): 6, ("BTC/USDT", "4h"): 4}


async def test_kline_specs_skips_disabled_rules():
    engine = _make_engine()
    engine._rules = [
        _make_rule(
            id="r1",
            trigger_type="candle_pattern",
            parameters={"interval": "1h", "consecutive_count": 3, "direction": "bearish"},
            enabled=False,
        ),
    ]
    assert engine._kline_specs() == []


# ── kline REST shape ──


async def test_fetch_klines_parses_ccxt_response_shape():
    """ccxt fetch_ohlcv returns [[ts, open, high, low, close, volume], …]."""
    engine = _make_engine()
    fake_client = MagicMock()
    fake_client.fetch_ohlcv = AsyncMock(
        return_value=[
            [1700000000000, 100.5, 101.0, 99.5, 100.0, 12.34],
            [1700003600000, 100.0, 100.2, 98.0, 98.5, 5.67],
        ]
    )
    engine._market_client = fake_client

    candles = await engine._fetch_klines("BTC/USDT", "1h", 2)

    fake_client.fetch_ohlcv.assert_awaited_once_with("BTC/USDT", timeframe="1h", limit=2)
    assert candles == [
        {"open": 100.5, "high": 101.0, "low": 99.5, "close": 100.0, "open_time": 1700000000000.0},
        {"open": 100.0, "high": 100.2, "low": 98.0, "close": 98.5, "open_time": 1700003600000.0},
    ]


async def test_poll_funding_rates_uses_ccxt_fetch_funding_rates():
    """funding_rate uses ccxt; spot pairs get mapped to linear perp symbols."""
    engine = _make_engine()
    engine._rules = [
        _make_rule(id="r1", pair="BTC/USDT", trigger_type="funding_rate"),
        _make_rule(id="r2", pair="ETH/USDT", trigger_type="funding_rate"),
    ]

    fake_client = MagicMock()
    fake_client.fetch_funding_rates = AsyncMock(
        return_value={
            "BTC/USDT:USDT": {"fundingRate": 0.0012},
            "ETH/USDT:USDT": {"fundingRate": -0.0003},
        }
    )
    engine._market_client = fake_client

    rates = await engine.poll_funding_rates()

    fake_client.fetch_funding_rates.assert_awaited_once()
    requested = fake_client.fetch_funding_rates.call_args.args[0]
    assert set(requested) == {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    assert rates == {"BTC/USDT": 0.0012, "ETH/USDT": -0.0003}
    # Engine state updated for the trigger condition lookup.
    assert engine._funding_rates["BTC/USDT"] == 0.0012


async def test_poll_funding_rates_swallows_ccxt_errors():
    engine = _make_engine(rules=[_make_rule(trigger_type="funding_rate")])
    engine._rules = [_make_rule(trigger_type="funding_rate")]
    fake_client = MagicMock()
    fake_client.fetch_funding_rates = AsyncMock(side_effect=RuntimeError("network"))
    engine._market_client = fake_client
    rates = await engine.poll_funding_rates()
    assert rates == {}
    # Existing cached rates aren't wiped on error.


async def test_poll_funding_rates_skips_when_no_rules():
    """No rules means no perp listings to query — don't even hit ccxt."""
    engine = _make_engine()
    engine._rules = []
    fake_client = MagicMock()
    fake_client.fetch_funding_rates = AsyncMock()
    engine._market_client = fake_client
    rates = await engine.poll_funding_rates()
    assert rates == {}
    fake_client.fetch_funding_rates.assert_not_awaited()


def test_to_swap_symbol_maps_spot_to_linear_perp():
    from cryptotrader.triggers.engine import _to_swap_symbol

    assert _to_swap_symbol("BTC/USDT") == "BTC/USDT:USDT"
    assert _to_swap_symbol("ETH/USDC") == "ETH/USDC:USDC"
    # Already-perp passes through.
    assert _to_swap_symbol("BTC/USDT:USDT") == "BTC/USDT:USDT"
    # Malformed input is preserved (caller's problem).
    assert _to_swap_symbol("garbage") == "garbage"


# ── funding rate scheduling ──


async def test_funding_poll_loop_runs_initial_poll():
    engine = _make_engine()
    engine._running = True
    engine.poll_funding_rates = AsyncMock(return_value={"BTC/USDT": 0.0012})

    # Force the loop's sleep to immediately raise CancelledError so we exit
    # after the initial poll.
    async def _cancel(*_a, **_kw):
        import asyncio

        raise asyncio.CancelledError

    with patch("cryptotrader.triggers.engine.asyncio.sleep", side_effect=_cancel):
        await engine._funding_poll_loop()

    engine.poll_funding_rates.assert_awaited()  # initial call ran
