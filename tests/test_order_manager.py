"""Tests for OrderManager state transitions."""

import pytest

from cryptotrader.models import Order, OrderStatus
from cryptotrader.execution.order import OrderManager


@pytest.fixture
def mgr():
    return OrderManager()


def test_valid_transition_pending_to_submitted(mgr):
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    assert order.status == OrderStatus.PENDING
    mgr.transition(order, OrderStatus.SUBMITTED)
    assert order.status == OrderStatus.SUBMITTED


def test_valid_transition_submitted_to_filled(mgr):
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    mgr.transition(order, OrderStatus.SUBMITTED)
    mgr.transition(order, OrderStatus.FILLED)
    assert order.status == OrderStatus.FILLED


def test_valid_transition_submitted_to_partially_filled(mgr):
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    mgr.transition(order, OrderStatus.SUBMITTED)
    mgr.transition(order, OrderStatus.PARTIALLY_FILLED)
    assert order.status == OrderStatus.PARTIALLY_FILLED


def test_valid_transition_partially_filled_to_filled(mgr):
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    mgr.transition(order, OrderStatus.SUBMITTED)
    mgr.transition(order, OrderStatus.PARTIALLY_FILLED)
    mgr.transition(order, OrderStatus.FILLED)
    assert order.status == OrderStatus.FILLED


def test_invalid_transition_raises(mgr):
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    with pytest.raises(ValueError, match="Invalid transition"):
        mgr.transition(order, OrderStatus.FILLED)  # PENDING -> FILLED not allowed


def test_invalid_transition_filled_to_submitted(mgr):
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    mgr.transition(order, OrderStatus.SUBMITTED)
    mgr.transition(order, OrderStatus.FILLED)
    with pytest.raises(ValueError):
        mgr.transition(order, OrderStatus.SUBMITTED)  # FILLED has no transitions


@pytest.mark.asyncio
async def test_place_success(mgr):
    class MockExchange:
        async def place_order(self, order):
            return {"id": "abc123", "status": "closed"}
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    result = await mgr.place(order, MockExchange())
    assert result.status == OrderStatus.FILLED
    assert result.exchange_id == "abc123"


@pytest.mark.asyncio
async def test_place_partial_fill(mgr):
    class MockExchange:
        async def place_order(self, order):
            return {"id": "abc456", "status": "partially_filled"}
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    result = await mgr.place(order, MockExchange())
    assert result.status == OrderStatus.PARTIALLY_FILLED


@pytest.mark.asyncio
async def test_place_failure(mgr):
    class MockExchange:
        async def place_order(self, order):
            raise RuntimeError("Connection lost")
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    result = await mgr.place(order, MockExchange())
    assert result.status == OrderStatus.FAILED


@pytest.mark.asyncio
async def test_place_open_stays_submitted(mgr):
    """Exchange returns 'open' status — order should stay SUBMITTED, not jump to FILLED."""
    class MockExchange:
        async def place_order(self, order):
            return {"id": "abc789", "status": "open"}
    order = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    result = await mgr.place(order, MockExchange())
    assert result.status == OrderStatus.SUBMITTED
    assert result.exchange_id == "abc789"
