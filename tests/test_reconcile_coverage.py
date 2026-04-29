"""Tests for execution/reconcile.py — Reconciler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cryptotrader.execution.reconcile import Reconciler
from cryptotrader.models import Order, OrderStatus


def _make_order(exchange_id: str = "ex1", pair: str = "BTC/USDT", status: OrderStatus = OrderStatus.SUBMITTED) -> Order:
    o = MagicMock(spec=Order)
    o.exchange_id = exchange_id
    o.pair = pair
    o.status = status
    return o


class TestReconcile:
    @pytest.mark.asyncio
    async def test_no_mismatch(self):
        exchange = MagicMock()
        exchange.get_order = AsyncMock(return_value={"status": "open"})
        r = Reconciler(exchange)
        order = _make_order(status=OrderStatus.SUBMITTED)
        result = await r.reconcile([order])
        assert result == []

    @pytest.mark.asyncio
    async def test_mismatch_detected(self):
        exchange = MagicMock()
        exchange.get_order = AsyncMock(return_value={"status": "closed"})
        r = Reconciler(exchange)
        order = _make_order(status=OrderStatus.SUBMITTED)
        result = await r.reconcile([order])
        assert len(result) == 1
        assert result[0][1] == "filled"

    @pytest.mark.asyncio
    async def test_skip_no_exchange_id(self):
        exchange = MagicMock()
        r = Reconciler(exchange)
        order = _make_order(exchange_id="")
        result = await r.reconcile([order])
        assert result == []
        exchange.get_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_error(self):
        exchange = MagicMock()
        exchange.get_order = AsyncMock(side_effect=Exception("network"))
        r = Reconciler(exchange)
        order = _make_order()
        result = await r.reconcile([order])
        assert result == []

    @pytest.mark.asyncio
    async def test_unknown_remote_status(self):
        exchange = MagicMock()
        exchange.get_order = AsyncMock(return_value={"status": "unknown_status"})
        r = Reconciler(exchange)
        order = _make_order()
        result = await r.reconcile([order])
        assert result == []


class TestDetectOrphans:
    @pytest.mark.asyncio
    async def test_no_orphans(self):
        exchange = MagicMock()
        exchange.fetch_open_orders = AsyncMock(return_value=[{"id": "ex1"}])
        r = Reconciler(exchange)
        result = await r.detect_orphans({"ex1"})
        assert result == []

    @pytest.mark.asyncio
    async def test_orphan_found(self):
        exchange = MagicMock()
        exchange.fetch_open_orders = AsyncMock(return_value=[{"id": "orphan1"}, {"id": "ex1"}])
        r = Reconciler(exchange)
        result = await r.detect_orphans({"ex1"})
        assert len(result) == 1
        assert result[0]["id"] == "orphan1"

    @pytest.mark.asyncio
    async def test_fetch_error(self):
        exchange = MagicMock()
        exchange.fetch_open_orders = AsyncMock(side_effect=Exception("timeout"))
        r = Reconciler(exchange)
        result = await r.detect_orphans(set())
        assert result == []
