"""Smoke tests for data models."""

from datetime import datetime

import pandas as pd

from cryptotrader._compat import UTC
from cryptotrader.models import (
    VALID_TRANSITIONS,
    AgentAnalysis,
    DataSnapshot,
    MacroData,
    MarketData,
    NewsSentiment,
    OnchainData,
    Order,
    OrderStatus,
)


def test_market_data():
    md = MarketData(
        pair="BTC/USDT",
        ohlcv=pd.DataFrame({"close": [100, 101]}),
        ticker={"last": 101},
        funding_rate=0.0001,
        orderbook_imbalance=0.1,
        volatility=0.02,
    )
    assert md.pair == "BTC/USDT"
    assert md.funding_rate == 0.0001


def test_agent_analysis():
    a = AgentAnalysis(
        agent_id="tech",
        pair="BTC/USDT",
        direction="bullish",
        confidence=0.8,
        reasoning="test",
        key_factors=["ma_cross"],
    )
    assert a.direction == "bullish"
    assert a.confidence == 0.8


def test_order_status_transitions():
    assert OrderStatus.SUBMITTED in VALID_TRANSITIONS[OrderStatus.PENDING]
    assert OrderStatus.FILLED in VALID_TRANSITIONS[OrderStatus.SUBMITTED]
    assert OrderStatus.PENDING not in VALID_TRANSITIONS.get(OrderStatus.FILLED, set())


def test_order():
    o = Order(pair="BTC/USDT", side="buy", amount=0.1, price=50000)
    assert o.status == OrderStatus.PENDING
    assert o.order_type == "market"


def test_data_snapshot():
    snap = DataSnapshot(
        timestamp=datetime.now(UTC),
        pair="BTC/USDT",
        market=MarketData("BTC/USDT", pd.DataFrame(), {}, 0.0, 0.0, 0.0),
        onchain=OnchainData(),
        news=NewsSentiment(),
        macro=MacroData(),
    )
    assert snap.pair == "BTC/USDT"
