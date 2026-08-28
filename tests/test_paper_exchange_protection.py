"""Paper protection orders execute against each newly collected closed bar."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData
from tests.factories.signal_fusion import context, position


def _exchange(*, signed_amount: float):
    cfg = MagicMock()
    cfg.backtest.initial_capital = 10_000.0
    cfg.backtest.slippage_base = 0.0
    cfg.backtest.fee_bps = 0.0
    initial_side = "long" if signed_amount > 0.0 else "short"
    with patch("cryptotrader.config.load_config", return_value=cfg):
        from cryptotrader.execution.simulator import PaperExchange

        return PaperExchange(
            initial_balances={"USDT": 10_000.0},
            initial_positions={
                "BTC/USDT:USDT": {
                    "amount": signed_amount,
                    "side": initial_side,
                    "avg_price": 100.0,
                }
            },
        )


def _context(*, side: str, high: float, low: float):
    frame = pd.DataFrame(
        [[100.0, high, low, 100.0, 1.0]],
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([datetime(2026, 1, 1, tzinfo=UTC)]),
    )
    snapshot = DataSnapshot(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        pair="BTC/USDT:USDT",
        market=MarketData("BTC/USDT:USDT", frame, {"last": 100.0}, 0.0, 0.0, 0.0),
        onchain=OnchainData(),
        news=NewsSentiment(),
        macro=MacroData(),
    )
    signed_amount = 2.0 if side == "long" else -2.0
    return context(
        position=position(side, 2.0, 0.5, avg_price=100.0),
        snapshots={"1h": snapshot},
    ), signed_amount


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side", "high", "low", "stop_loss", "take_profit", "reason", "expected_balance"),
    [
        ("long", 105.0, 89.0, 90.0, 120.0, "stop_loss", 9_980.0),
        ("long", 121.0, 95.0, 90.0, 120.0, "take_profit", 10_040.0),
        ("short", 111.0, 95.0, 110.0, 80.0, "stop_loss", 9_980.0),
        ("short", 105.0, 79.0, 110.0, 80.0, "take_profit", 10_040.0),
    ],
)
async def test_paper_protection_triggers_long_and_short_exits(
    side,
    high,
    low,
    stop_loss,
    take_profit,
    reason,
    expected_balance,
):
    signal_context, signed_amount = _context(side=side, high=high, low=low)
    exchange = _exchange(signed_amount=signed_amount)
    assert exchange.supports_protection_orders() is True
    algo_id = await exchange.place_algo_oco(
        "BTC/USDT:USDT",
        side="sell" if side == "long" else "buy",
        amount=2.0,
        sl_trigger_px=stop_loss,
        tp_trigger_px=take_profit,
        pos_side=side,
    )

    triggered = await exchange.process_pending_protection(signal_context)

    assert triggered is True
    assert await exchange.get_positions() == {}
    assert (await exchange.get_balance())["USDT"] == pytest.approx(expected_balance)
    assert exchange._algos[algo_id]["status"] == "triggered"
    assert exchange._algos[algo_id]["trigger_reason"] == reason
    assert await exchange.list_pending_algos(pair="BTC/USDT:USDT") == []


@pytest.mark.asyncio
async def test_paper_ambiguous_bar_uses_stop_first_and_terminal_oco_is_one_shot():
    signal_context, signed_amount = _context(side="long", high=121.0, low=89.0)
    exchange = _exchange(signed_amount=signed_amount)
    algo_id = await exchange.place_algo_oco(
        "BTC/USDT:USDT",
        side="sell",
        amount=2.0,
        sl_trigger_px=90.0,
        tp_trigger_px=120.0,
        pos_side="long",
    )

    first = await exchange.process_pending_protection(signal_context)
    second = await exchange.process_pending_protection(signal_context)

    assert first is True
    assert second is False
    assert exchange._algos[algo_id]["trigger_reason"] == "stop_loss"
    assert len(exchange._orders) == 1
    assert (await exchange.get_balance())["USDT"] == pytest.approx(9_980.0)
