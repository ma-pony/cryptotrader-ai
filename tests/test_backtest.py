"""Backtest engine tests with mock data."""

from cryptotrader.backtest.result import BacktestResult
from cryptotrader.backtest.engine import BacktestEngine


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
