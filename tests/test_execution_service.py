"""ExecutionService 的顺序成交与统一保护单语义。"""

from __future__ import annotations

import pytest

from cryptotrader.decision.models import ExecutionPlan, OrderIntent
from cryptotrader.execution.order import OrderManager
from tests.factories.signal_fusion import context, position


class _Exchange:
    def __init__(self, statuses=("closed",)) -> None:
        self.statuses = iter(statuses)
        self.orders = []
        self.cancelled_algos = []
        self.ocos = []

    async def place_order(self, order):
        self.orders.append(order)
        status = next(self.statuses)
        return {"id": f"order-{len(self.orders)}", "status": status, "reason": "rejected"}

    async def list_pending_algos(self, pair=None):
        return [{"algoId": "old-oco"}]

    async def cancel_algo(self, algo_id, pair):
        self.cancelled_algos.append((algo_id, pair))

    async def place_algo_oco(self, pair, **kwargs):
        self.ocos.append((pair, kwargs))
        return "new-oco"


@pytest.mark.asyncio
async def test_reversal_fills_in_order_then_replaces_protection():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange(("closed", "closed"))
    service = ExecutionService(OrderManager(), exchange)
    plan = ExecutionPlan(
        intents=(
            OrderIntent("BTC/USDT:USDT", "sell", 1.0, True),
            OrderIntent("BTC/USDT:USDT", "sell", 2.0, False),
        ),
        stop_loss=110.0,
        take_profit=80.0,
    )

    result = await service.execute(plan, context(position=position("long", 1.0, 0.5)))

    assert result.succeeded is True
    assert [order.reduce_only for order in exchange.orders] == [True, False]
    assert exchange.cancelled_algos == [("old-oco", "BTC/USDT:USDT")]
    assert exchange.ocos[0][1] == {
        "side": "buy",
        "amount": 2.0,
        "sl_trigger_px": 110.0,
        "tp_trigger_px": 80.0,
        "pos_side": "short",
    }
    assert result.algo_id == "new-oco"


@pytest.mark.asyncio
async def test_failed_first_reversal_leg_never_sends_second_or_changes_protection():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange(("failed", "closed"))
    service = ExecutionService(OrderManager(), exchange)
    plan = ExecutionPlan(
        intents=(
            OrderIntent("BTC/USDT:USDT", "sell", 1.0, True),
            OrderIntent("BTC/USDT:USDT", "sell", 2.0, False),
        ),
        stop_loss=110.0,
        take_profit=80.0,
    )

    result = await service.execute(plan, context(position=position("long", 1.0, 0.5)))

    assert result.succeeded is False
    assert len(exchange.orders) == 1
    assert exchange.cancelled_algos == []
    assert exchange.ocos == []


@pytest.mark.asyncio
async def test_flat_result_cancels_protection_without_creating_oco():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange(("closed",))
    service = ExecutionService(OrderManager(), exchange)
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "sell", 1.0, True),),
        stop_loss=None,
        take_profit=None,
    )

    result = await service.execute(plan, context(position=position("long", 1.0, 0.5)))

    assert result.succeeded is True
    assert exchange.cancelled_algos == [("old-oco", "BTC/USDT:USDT")]
    assert exchange.ocos == []
