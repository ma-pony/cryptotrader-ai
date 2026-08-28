"""ExecutionService 的顺序成交与统一保护单语义。"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from cryptotrader.decision.models import ExecutionPlan, OrderIntent
from cryptotrader.execution.order import OrderManager
from cryptotrader.models import OrderStatus
from tests.factories.signal_fusion import context, position


class _Exchange:
    def __init__(
        self,
        statuses=("closed",),
        *,
        supports_protection_orders=True,
        old_algos=("old-oco",),
        oco_error=None,
        cancel_errors=(),
    ) -> None:
        self.statuses = iter(statuses)
        self._supports_protection_orders = supports_protection_orders
        self.orders = []
        self.cancelled_algos = []
        self.ocos = []
        self.pending_algos = list(old_algos)
        self.oco_error = oco_error
        self.cancel_errors = set(cancel_errors)
        self.events = []

    def supports_protection_orders(self):
        return self._supports_protection_orders

    async def place_order(self, order):
        self.orders.append(order)
        self.events.append(("order", order.side, order.amount, order.reduce_only))
        status = next(self.statuses)
        return {"id": f"order-{len(self.orders)}", "status": status, "reason": "rejected"}

    async def list_pending_algos(self, pair=None):
        return [{"algoId": algo_id} for algo_id in self.pending_algos]

    async def cancel_algo(self, algo_id, pair):
        self.events.append(("cancel_oco", algo_id))
        if algo_id in self.cancel_errors:
            raise RuntimeError(f"cannot cancel {algo_id}")
        self.cancelled_algos.append((algo_id, pair))
        if algo_id in self.pending_algos:
            self.pending_algos.remove(algo_id)

    async def place_algo_oco(self, pair, **kwargs):
        self.events.append(("place_oco", "new-oco"))
        self.ocos.append((pair, kwargs))
        if self.oco_error is not None:
            raise self.oco_error
        self.pending_algos.append("new-oco")
        return "new-oco"


class _DirectOrderManager:
    """Expose service behavior when OrderManager.place itself raises or reports fills."""

    def __init__(self, *outcomes) -> None:
        self.outcomes = iter(outcomes)
        self.orders = []

    async def place(self, order, exchange):
        self.orders.append(order)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        status, raw = outcome
        order.status = status
        order.exchange_id = f"direct-{len(self.orders)}"
        return order, {"id": order.exchange_id, "status": status.value, **raw}


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
        "created_as_of": datetime(2026, 1, 1, tzinfo=UTC),
    }
    assert result.algo_id == "new-oco"


@pytest.mark.asyncio
async def test_unsupported_live_protection_rejects_before_any_position_order():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange(supports_protection_orders=False)
    service = ExecutionService(OrderManager(), exchange)
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 1.0, False),),
        stop_loss=90.0,
        take_profit=120.0,
    )

    result = await service.execute(plan, context())

    assert result.succeeded is False
    assert "protection" in result.error.lower()
    assert exchange.orders == []
    assert exchange.events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_loss", "take_profit"),
    [
        (100.0, 120.0),
        (0.0, 120.0),
        (90.0, math.inf),
        (110.0, 120.0),
    ],
)
async def test_invalid_long_protection_prices_reject_before_orders(stop_loss, take_profit):
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange()
    service = ExecutionService(OrderManager(), exchange)
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 1.0, False),),
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    result = await service.execute(plan, context(price=100.0))

    assert result.succeeded is False
    assert "protection" in result.error.lower()
    assert exchange.orders == []


@pytest.mark.asyncio
async def test_invalid_short_protection_prices_reject_before_orders():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange()
    service = ExecutionService(OrderManager(), exchange)
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "sell", 1.0, False),),
        stop_loss=90.0,
        take_profit=80.0,
    )

    result = await service.execute(plan, context(price=100.0))

    assert result.succeeded is False
    assert exchange.orders == []


@pytest.mark.asyncio
async def test_existing_protection_is_cancelled_only_after_replacement_exists():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange(("closed",))
    service = ExecutionService(OrderManager(), exchange)
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 1.0, False),),
        stop_loss=90.0,
        take_profit=120.0,
    )

    result = await service.execute(plan, context())

    assert result.succeeded is True
    assert exchange.events == [
        ("order", "buy", 1.0, False),
        ("place_oco", "new-oco"),
        ("cancel_oco", "old-oco"),
    ]


@pytest.mark.asyncio
async def test_replacement_failure_compensates_fills_in_reverse_and_retains_old_protection():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange(
        ("closed", "closed", "closed", "closed"),
        oco_error=RuntimeError("replacement rejected"),
    )
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
    assert [(order.side, order.amount, order.reduce_only) for order in exchange.orders] == [
        ("sell", 1.0, True),
        ("sell", 2.0, False),
        ("buy", 2.0, True),
        ("buy", 1.0, False),
    ]
    assert exchange.cancelled_algos == []
    assert exchange.pending_algos == ["old-oco"]
    assert result.retained_algo_ids == ("old-oco",)


@pytest.mark.asyncio
async def test_old_protection_cancellation_failure_removes_replacement_and_restores_position():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange(
        ("closed", "closed"),
        cancel_errors=("old-oco",),
    )
    service = ExecutionService(OrderManager(), exchange)
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 1.0, False),),
        stop_loss=90.0,
        take_profit=120.0,
    )

    result = await service.execute(plan, context())

    assert result.succeeded is False
    assert [(order.side, order.amount, order.reduce_only) for order in exchange.orders] == [
        ("buy", 1.0, False),
        ("sell", 1.0, True),
    ]
    assert exchange.cancelled_algos == [("new-oco", "BTC/USDT:USDT")]
    assert exchange.pending_algos == ["old-oco"]
    assert result.retained_algo_ids == ("old-oco",)


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
async def test_failed_later_reversal_leg_compensates_prior_fill_and_retains_old_protection():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange(("closed", "failed", "closed"))
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
    assert [(order.side, order.amount, order.reduce_only) for order in exchange.orders] == [
        ("sell", 1.0, True),
        ("sell", 2.0, False),
        ("buy", 1.0, False),
    ]
    assert exchange.cancelled_algos == []
    assert exchange.ocos == []
    assert exchange.pending_algos == ["old-oco"]
    assert result.retained_algo_ids == ("old-oco",)


@pytest.mark.asyncio
async def test_later_order_manager_exception_compensates_known_fills_and_returns_audit():
    from cryptotrader.execution.service import ExecutionService

    manager = _DirectOrderManager(
        (OrderStatus.FILLED, {}),
        RuntimeError("manager state transition failed"),
        (OrderStatus.FILLED, {}),
    )
    exchange = _Exchange()
    service = ExecutionService(manager, exchange)
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
    assert "manager state transition failed" in result.error
    assert "original position restored" in result.error
    assert [(order.side, order.amount, order.reduce_only) for order in manager.orders] == [
        ("sell", 1.0, True),
        ("sell", 2.0, False),
        ("buy", 1.0, False),
    ]
    assert [(item.intent.side, item.filled_amount) for item in result.orders] == [
        ("sell", 1.0),
        ("buy", 1.0),
    ]
    assert result.retained_algo_ids == ("old-oco",)


@pytest.mark.asyncio
async def test_compensation_order_manager_exception_becomes_explicit_audited_failure():
    from cryptotrader.execution.service import ExecutionService

    manager = _DirectOrderManager(
        (OrderStatus.FILLED, {}),
        RuntimeError("compensation transport failed"),
    )
    exchange = _Exchange(oco_error=RuntimeError("replacement rejected"))
    service = ExecutionService(manager, exchange)
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 1.0, False),),
        stop_loss=90.0,
        take_profit=120.0,
    )

    result = await service.execute(plan, context())

    assert result.succeeded is False
    assert "position compensation failed" in result.error
    assert "compensation transport failed" in result.error
    assert [(item.intent.side, item.filled_amount) for item in result.orders] == [("buy", 1.0)]
    assert result.retained_algo_ids == ("old-oco",)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_status", [OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELLED])
async def test_actual_partial_fill_is_compensated_before_failure_returns(failed_status):
    from cryptotrader.execution.service import ExecutionService

    manager = _DirectOrderManager(
        (OrderStatus.FILLED, {}),
        (failed_status, {"filled": 0.5}),
        (OrderStatus.FILLED, {}),
        (OrderStatus.FILLED, {}),
    )
    exchange = _Exchange()
    service = ExecutionService(manager, exchange)
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
    assert [(order.side, order.amount, order.reduce_only) for order in manager.orders] == [
        ("sell", 1.0, True),
        ("sell", 2.0, False),
        ("buy", 0.5, True),
        ("buy", 1.0, False),
    ]
    assert [(item.status, item.filled_amount) for item in result.orders] == [
        ("filled", 1.0),
        (failed_status.value, 0.5),
        ("filled", 0.5),
        ("filled", 1.0),
    ]
    assert result.retained_algo_ids == ("old-oco",)


@pytest.mark.asyncio
async def test_replacement_cleanup_failure_reports_old_and_new_algos_as_unresolved():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange(("closed", "closed"), cancel_errors=("old-oco", "new-oco"))
    service = ExecutionService(OrderManager(), exchange)
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 1.0, False),),
        stop_loss=90.0,
        take_profit=120.0,
    )

    result = await service.execute(plan, context())

    assert result.succeeded is False
    assert exchange.pending_algos == ["old-oco", "new-oco"]
    assert result.retained_algo_ids == ("old-oco", "new-oco")


@pytest.mark.asyncio
async def test_multiple_old_algos_report_only_not_confirmed_cancelled_ids():
    from cryptotrader.execution.service import ExecutionService

    exchange = _Exchange(
        ("closed", "closed"),
        old_algos=("old-1", "old-2", "old-3"),
        cancel_errors=("old-2",),
    )
    service = ExecutionService(OrderManager(), exchange)
    plan = ExecutionPlan(
        intents=(OrderIntent("BTC/USDT:USDT", "buy", 1.0, False),),
        stop_loss=90.0,
        take_profit=120.0,
    )

    result = await service.execute(plan, context())

    assert result.succeeded is False
    assert exchange.pending_algos == ["old-2", "old-3"]
    assert result.retained_algo_ids == ("old-2", "old-3")


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
