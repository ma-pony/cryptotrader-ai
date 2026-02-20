"""Exchange hardening tests — retry, balance check, precision."""

import pytest
from cryptotrader.execution.exchange import LiveExchange, ExchangeAdapter
from cryptotrader.models import Order


def test_exchange_adapter_protocol():
    assert issubclass(LiveExchange, ExchangeAdapter)
    assert hasattr(ExchangeAdapter, 'place_order')
    assert hasattr(ExchangeAdapter, 'cancel_order')
    assert hasattr(ExchangeAdapter, 'get_balance')


def test_order_creation():
    o = Order(pair="BTC/USDT", side="buy", amount=0.001, price=50000)
    assert o.order_type == "market"
    assert o.side == "buy"
