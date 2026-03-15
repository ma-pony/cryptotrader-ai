"""Backtest engine tests with mock data.

Integration tests in this file run the complete backtest pipeline with all
external network calls mocked out so they can execute in an offline environment.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader.backtest.engine import BacktestEngine
from cryptotrader.backtest.result import BacktestResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAIR = "BTC/USDT"
START = "2024-01-01"
END = "2024-01-10"

# Interval step for 4h candles in milliseconds
_4H_MS = 14_400_000

# Number of synthetic candles we generate: needs to exceed the lookback window
# (default 20) plus enough trading bars to exercise signal logic.
_N_CANDLES = 80


def _make_candles(n: int, base_price: float = 50_000.0, trend: float = 10.0) -> list[list]:
    """Return synthetic OHLCV candles with a mild upward trend.

    Each candle is [ts, open, high, low, close, volume].
    Timestamps start from a fixed epoch so backtest date windows align.
    """
    # Start at 2024-01-01 00:00:00 UTC in milliseconds
    start_ts = 1_704_067_200_000
    candles = []
    price = base_price
    for i in range(n):
        ts = start_ts + i * _4H_MS
        o = price
        c = price + trend
        h = c + 20.0
        lo = o - 20.0
        v = 100.0
        candles.append([ts, o, h, lo, c, v])
        price = c
    return candles


def _rising_candles(n: int) -> list[list]:
    """Strictly rising prices to guarantee long signals from _simple_signal."""
    return _make_candles(n, base_price=50_000.0, trend=50.0)


def _falling_candles(n: int) -> list[list]:
    """Strictly falling prices to guarantee short signals from _simple_signal."""
    return _make_candles(n, base_price=100_000.0, trend=-50.0)


# Patch targets used in multiple tests.
# fetch_historical is imported at module level in engine.py, so patch the engine's
# binding directly.  historical_data functions are imported lazily inside
# _fetch_historical_data(), so we patch them at their source module.
_FETCH_HIST = "cryptotrader.backtest.engine.fetch_historical"
_FETCH_FNG = "cryptotrader.backtest.historical_data.fetch_fear_greed"
_FETCH_FUND = "cryptotrader.backtest.historical_data.fetch_funding_rate"
_FETCH_BTC_DOM = "cryptotrader.backtest.historical_data.fetch_btc_dominance"
_FETCH_FRED = "cryptotrader.backtest.historical_data.fetch_fred_series"
_FETCH_FUT_VOL = "cryptotrader.backtest.historical_data.fetch_futures_volume"
# get_range is imported lazily inside _load_extended_data() from data.store
_GET_RANGE = "cryptotrader.data.store.get_range"


def _patch_all_external(candles: list[list]):
    """Return a context manager stack that mocks every external call in engine."""
    return [
        patch(_FETCH_HIST, new=AsyncMock(return_value=candles)),
        patch(_FETCH_FNG, new=AsyncMock(return_value={})),
        patch(_FETCH_FUND, new=AsyncMock(return_value={})),
        patch(_FETCH_BTC_DOM, new=AsyncMock(return_value={})),
        patch(_FETCH_FRED, new=AsyncMock(return_value={})),
        patch(_FETCH_FUT_VOL, new=AsyncMock(return_value={})),
        patch(_GET_RANGE, return_value={}),
    ]


# ---------------------------------------------------------------------------
# Unit tests (existing, preserved)
# ---------------------------------------------------------------------------


def test_backtest_result_summary():
    r = BacktestResult(total_return=0.15, sharpe_ratio=1.2, max_drawdown=-0.05, win_rate=0.6)
    s = r.summary()
    assert s["total_return"] == "15.00%"
    assert s["num_trades"] == 0


def test_backtest_result_empty():
    r = BacktestResult()
    assert r.total_return == 0.0
    assert r.equity_curve == []


def test_simple_signal():
    engine = BacktestEngine("BTC/USDT", "2025-01-01", "2025-01-02")
    # Rising prices -> long
    window = [[i * 1000, 100 + i, 101 + i, 99 + i, 100 + i, 1000] for i in range(60)]
    assert engine._simple_signal(window) == "long"


def test_simple_signal_short():
    engine = BacktestEngine("BTC/USDT", "2025-01-01", "2025-01-02")
    # Falling prices -> short
    window = [[i * 1000, 200 - i, 201 - i, 199 - i, 200 - i, 1000] for i in range(60)]
    assert engine._simple_signal(window) == "short"


def test_simple_signal_insufficient_data():
    engine = BacktestEngine("BTC/USDT", "2025-01-01", "2025-01-02")
    window = [[0, 100, 101, 99, 100, 1000]]
    assert engine._simple_signal(window) == "hold"


# ---------------------------------------------------------------------------
# Integration tests (new — no network, no filesystem)
# ---------------------------------------------------------------------------


class TestBacktestIntegration:
    """Complete backtest pipeline tests with all external I/O mocked.

    All CCXT and HTTP calls are replaced with AsyncMock objects returning
    synthetic data.  No files are written to disk; no network is required.
    """

    @pytest.mark.asyncio
    async def test_run_returns_backtest_result_type(self):
        """run() must return a BacktestResult instance."""
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        # Apply all patches sequentially
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            result = await engine.run()

        assert isinstance(result, BacktestResult)

    @pytest.mark.asyncio
    async def test_result_has_required_fields(self):
        """Result must expose total_return, win_rate, sharpe_ratio, max_drawdown."""
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            result = await engine.run()

        # Verify all required fields exist and are numeric
        assert isinstance(result.total_return, float), "total_return must be float"
        assert isinstance(result.win_rate, float), "win_rate must be float"
        assert isinstance(result.sharpe_ratio, float), "sharpe_ratio must be float"
        assert isinstance(result.max_drawdown, float), "max_drawdown must be float"
        assert isinstance(result.trades, list), "trades must be list"
        assert isinstance(result.equity_curve, list), "equity_curve must be list"

    @pytest.mark.asyncio
    async def test_win_rate_in_valid_range(self):
        """win_rate must be in [0.0, 1.0]."""
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            result = await engine.run()

        assert 0.0 <= result.win_rate <= 1.0

    @pytest.mark.asyncio
    async def test_equity_curve_starts_at_initial_capital(self):
        """First equity_curve value equals the configured initial_capital."""
        initial = 10_000.0
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False, initial_capital=initial)
            result = await engine.run()

        assert result.equity_curve[0] == pytest.approx(initial)

    @pytest.mark.asyncio
    async def test_rising_prices_generate_long_trades(self):
        """Monotonically rising prices should trigger at least one long trade."""
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            result = await engine.run()

        long_sides = {"buy", "ai_close_long", "close"}
        long_trades = [t for t in result.trades if t.get("side") in long_sides]
        assert len(long_trades) > 0, "Expected at least one long-side trade on rising prices"

    @pytest.mark.asyncio
    async def test_empty_candles_returns_empty_result(self):
        """When fetch_historical returns no candles, result should be empty."""
        patches = _patch_all_external([])
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            result = await engine.run()

        assert result.total_return == 0.0
        assert result.trades == []
        assert result.equity_curve == []

    @pytest.mark.asyncio
    async def test_summary_dict_contains_required_keys(self):
        """summary() output must include total_return, win_rate, sharpe_ratio, max_drawdown."""
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            result = await engine.run()

        s = result.summary()
        required_keys = {"total_return", "win_rate", "sharpe_ratio", "max_drawdown", "num_trades"}
        assert required_keys.issubset(s.keys()), f"Missing keys: {required_keys - s.keys()}"

    @pytest.mark.asyncio
    async def test_total_pnl_consistent_with_equity_curve(self):
        """total_return should be consistent with the equity curve endpoints."""
        initial = 10_000.0
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False, initial_capital=initial)
            result = await engine.run()

        if len(result.equity_curve) > 1:
            # total_return = (final_equity - initial_capital) / initial_capital
            # final_equity is reflected by the last equity_curve value
            expected_return_sign = result.equity_curve[-1] >= initial
            positive_return = result.total_return >= 0.0
            assert expected_return_sign == positive_return

    @pytest.mark.asyncio
    async def test_max_drawdown_is_non_positive(self):
        """max_drawdown must be <= 0 (it represents a loss percentage)."""
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            result = await engine.run()

        assert result.max_drawdown <= 0.0

    @pytest.mark.asyncio
    async def test_decisions_list_populated(self):
        """decisions list should contain per-bar entries when candles are present."""
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            result = await engine.run()

        # Should have one decision per trading bar (candles beyond lookback)
        assert len(result.decisions) > 0

    @pytest.mark.asyncio
    async def test_decisions_have_required_fields(self):
        """Each decision record must have ts, price, and final_action fields."""
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            result = await engine.run()

        for dec in result.decisions[:5]:
            assert "ts" in dec
            assert "price" in dec
            assert "final_action" in dec

    @pytest.mark.asyncio
    async def test_ccxt_not_called_when_cache_mocked(self):
        """fetch_historical mock is called; real ccxt.binance must NOT be instantiated."""
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0] as mock_fetch,
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            # Ensure ccxt module is NOT imported (or if already imported, its binance
            # class should not be called).  We verify fetch_historical was called instead.
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            await engine.run()

        mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_network_calls_made(self):
        """Running with all mocks should never open a real network socket."""
        import socket

        original_connect = socket.socket.connect
        connections: list[tuple] = []

        def patched_connect(self, address):
            connections.append(address)
            return original_connect(self, address)

        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patch.object(socket.socket, "connect", patched_connect),
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            await engine.run()

        # Filter out loopback connections (e.g., Redis health checks during import)
        external = [a for a in connections if not str(a[0]).startswith("127.") and a[0] != "::1"]
        assert external == [], f"Unexpected external connections: {external}"

    @pytest.mark.asyncio
    async def test_apply_costs_affects_fill_price(self):
        """Slippage and fee should increase buy fill price and decrease sell fill price."""
        engine = BacktestEngine(PAIR, START, END, use_llm=False, slippage_bps=5.0, fee_bps=10.0)
        buy_fill = engine._apply_costs(100.0, "buy")
        sell_fill = engine._apply_costs(100.0, "sell")
        assert buy_fill > 100.0
        assert sell_fill < 100.0

    def test_compute_result_win_rate_calculation(self):
        """_compute_result correctly computes win_rate from trade PnL."""
        engine = BacktestEngine(PAIR, START, END, use_llm=False, initial_capital=10_000.0)
        trades = [
            {"side": "close", "price": 100.0, "pnl": 50.0, "ts": 0},
            {"side": "close", "price": 100.0, "pnl": -20.0, "ts": 1},
            {"side": "close", "price": 100.0, "pnl": 30.0, "ts": 2},
        ]
        equity_curve = [10_000.0, 10_050.0, 10_030.0, 10_060.0]
        result = engine._compute_result(10_060.0, equity_curve, trades)

        assert result.win_rate == pytest.approx(2 / 3)
        assert result.total_return == pytest.approx(0.006)
        assert result.max_drawdown <= 0.0

    def test_compute_result_empty_trades(self):
        """_compute_result with no trades produces zero win_rate."""
        engine = BacktestEngine(PAIR, START, END, use_llm=False, initial_capital=10_000.0)
        result = engine._compute_result(10_000.0, [10_000.0], [])
        assert result.win_rate == 0.0
        assert result.total_return == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_close_position_at_end_of_backtest(self):
        """An open position at backtest end should be closed in the final trade list."""
        candles = _rising_candles(_N_CANDLES)
        patches = _patch_all_external(candles)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            engine = BacktestEngine(PAIR, START, END, use_llm=False)
            result = await engine.run()

        # If any trades occurred, at least some should have pnl (position was closed)
        pnl_trades = [t for t in result.trades if "pnl" in t]
        if result.trades:
            assert len(pnl_trades) > 0, "Expected at least one closed trade with PnL"
