"""Reconciliation scenario tests."""

import pytest
from cryptotrader.execution.reconcile import Reconciler
from cryptotrader.models import Order, OrderStatus


class MockExchange:
    def __init__(self, orders: dict):
        self._orders = orders

    async def get_order(self, order_id: str) -> dict:
        return self._orders.get(order_id, {})


@pytest.mark.asyncio
async def test_reconcile_mismatch():
    ex = MockExchange({"ord1": {"status": "closed"}})
    r = Reconciler(ex)
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000,
                  status=OrderStatus.SUBMITTED, exchange_id="ord1")
    mismatches = await r.reconcile([order])
    assert len(mismatches) == 1
    assert mismatches[0][1] == "filled"


@pytest.mark.asyncio
async def test_reconcile_no_mismatch():
    ex = MockExchange({"ord1": {"status": "closed"}})
    r = Reconciler(ex)
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000,
                  status=OrderStatus.FILLED, exchange_id="ord1")
    mismatches = await r.reconcile([order])
    assert len(mismatches) == 0


@pytest.mark.asyncio
async def test_reconcile_skips_no_exchange_id():
    ex = MockExchange({})
    r = Reconciler(ex)
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    mismatches = await r.reconcile([order])
    assert len(mismatches) == 0
