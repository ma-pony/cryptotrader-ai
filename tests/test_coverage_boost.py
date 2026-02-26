"""Tests targeting coverage gaps in backtest engine, simulator, notifications, and verdict."""

import pytest

from cryptotrader.backtest.engine import BacktestEngine
from cryptotrader.backtest.result import BacktestResult
from cryptotrader.execution.simulator import PaperExchange
from cryptotrader.models import Order
from cryptotrader.notifications import Notifier


# ── BacktestEngine._apply_costs ──


def test_apply_costs_buy():
    engine = BacktestEngine("BTC/USDT", "2025-01-01", "2025-01-02", slippage_bps=10, fee_bps=10)
    fill = engine._apply_costs(10000.0, "buy")
    # buy: price + slip + fee = 10000 + 10 + 10 = 10020
    assert fill == pytest.approx(10020.0)


def test_apply_costs_sell():
    engine = BacktestEngine("BTC/USDT", "2025-01-01", "2025-01-02", slippage_bps=10, fee_bps=10)
    fill = engine._apply_costs(10000.0, "sell")
    # sell: price - slip - fee = 10000 - 10 - 10 = 9980
    assert fill == pytest.approx(9980.0)


# ── BacktestEngine._compute_result ──


def test_compute_result_basic():
    engine = BacktestEngine("BTC/USDT", "2025-01-01", "2025-01-02", initial_capital=10000)
    trades = [{"side": "buy", "price": 100, "amount": 1, "ts": 0},
              {"side": "close", "price": 110, "pnl": 10, "ts": 1}]
    result = engine._compute_result(10010, [10000, 10005, 10010], trades)
    assert result.total_return == pytest.approx(0.001)
    assert result.win_rate == 1.0
    assert result.max_drawdown == 0.0  # monotonically rising
    assert result.sharpe_ratio > 0


def test_compute_result_no_trades():
    engine = BacktestEngine("BTC/USDT", "2025-01-01", "2025-01-02", initial_capital=10000)
    result = engine._compute_result(10000, [10000], [])
    assert result.total_return == 0.0
    assert result.sharpe_ratio == 0.0
    assert result.win_rate == 0.0


def test_compute_result_drawdown():
    engine = BacktestEngine("BTC/USDT", "2025-01-01", "2025-01-02", initial_capital=10000)
    curve = [10000, 10500, 9000, 9500]  # peak 10500, trough 9000
    result = engine._compute_result(9500, curve, [{"pnl": -500, "side": "close", "price": 95, "ts": 0}])
    expected_dd = (9000 - 10500) / 10500
    assert result.max_drawdown == pytest.approx(expected_dd)
    assert result.win_rate == 0.0  # one losing trade


# ── PaperExchange ──


@pytest.mark.asyncio
async def test_simulator_buy_and_sell():
    ex = PaperExchange()
    buy = await ex.place_order(Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000))
    assert buy["status"] == "filled"
    bal = await ex.get_balance()
    assert "BTC" in bal
    sell = await ex.place_order(Order(pair="BTC/USDT", side="sell", amount=0.1, price=50000))
    assert sell["status"] == "filled"


@pytest.mark.asyncio
async def test_simulator_insufficient_usdt():
    ex = PaperExchange()
    result = await ex.place_order(Order(pair="BTC/USDT", side="buy", amount=100, price=50000))
    assert result["status"] == "failed"
    assert "Insufficient USDT" in result["reason"]


@pytest.mark.asyncio
async def test_simulator_insufficient_base():
    ex = PaperExchange()
    result = await ex.place_order(Order(pair="BTC/USDT", side="sell", amount=1.0, price=50000))
    assert result["status"] == "failed"
    assert "Insufficient BTC" in result["reason"]


@pytest.mark.asyncio
async def test_simulator_cancel_order():
    ex = PaperExchange()
    buy = await ex.place_order(Order(pair="BTC/USDT", side="buy", amount=0.01, price=50000))
    cancelled = await ex.cancel_order(buy["id"])
    assert cancelled["status"] == "cancelled"


@pytest.mark.asyncio
async def test_simulator_cancel_unknown():
    ex = PaperExchange()
    with pytest.raises(ValueError, match="not found"):
        await ex.cancel_order("nonexistent")


@pytest.mark.asyncio
async def test_simulator_get_unknown():
    ex = PaperExchange()
    with pytest.raises(ValueError, match="not found"):
        await ex.get_order("nonexistent")


@pytest.mark.asyncio
async def test_simulator_fetch_open_orders():
    ex = PaperExchange()
    assert await ex.fetch_open_orders() == []


@pytest.mark.asyncio
async def test_simulator_close():
    ex = PaperExchange()
    await ex.close()  # should not raise


# ── Notifier ──


@pytest.mark.asyncio
async def test_notifier_disabled_no_url():
    n = Notifier(webhook_url="")
    await n.notify("trade", {"pair": "BTC/USDT"})  # should not raise


@pytest.mark.asyncio
async def test_notifier_disabled_flag():
    n = Notifier(webhook_url="http://example.com", enabled=False)
    await n.notify("trade", {"pair": "BTC/USDT"})  # should not raise


@pytest.mark.asyncio
async def test_notifier_event_filtered():
    n = Notifier(webhook_url="http://example.com", events=["trade"])
    await n.notify("unknown_event", {"pair": "BTC/USDT"})  # filtered out, no request


# ── BacktestResult.to_json ──


def test_backtest_result_to_json(tmp_path):
    r = BacktestResult(total_return=0.1, trades=[{"side": "buy"}], equity_curve=[10000, 11000])
    path = str(tmp_path / "result.json")
    r.to_json(path)
    import json
    with open(path) as f:
        data = json.load(f)
    assert data["summary"]["total_return"] == "10.00%"
    assert len(data["equity_curve"]) == 2


# ── PaperExchange.estimate_slippage ──


def test_simulator_slippage_scales_with_size():
    ex = PaperExchange()
    small = ex.estimate_slippage(Order(pair="BTC/USDT", side="buy", amount=0.01, price=50000))
    large = ex.estimate_slippage(Order(pair="BTC/USDT", side="buy", amount=10.0, price=50000))
    assert large > small
