"""Smoke tests for data models."""

from datetime import UTC, datetime

import pandas as pd

from cryptotrader.models import (
    AgentAnalysis, CheckResult, DataSnapshot, DecisionCommit,
    GateResult, MacroData, MarketData, NewsSentiment, OnchainData,
    Order, OrderStatus, TradeVerdict, VALID_TRANSITIONS,
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
        agent_id="tech", pair="BTC/USDT",
        direction="bullish", confidence=0.8,
        reasoning="test", key_factors=["ma_cross"],
    )
    assert a.direction == "bullish"
    assert a.confidence == 0.8


def test_trade_verdict():
    v = TradeVerdict(action="long", confidence=0.7, position_scale=0.85)
    assert v.action == "long"


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


def test_gate_result():
    g = GateResult(passed=False, rejected_by="DailyLossLimit", reason="exceeded")
    assert not g.passed
    assert g.rejected_by == "DailyLossLimit"


def test_decision_commit():
    dc = DecisionCommit(
        hash="abc12345", parent_hash=None,
        timestamp=datetime.now(UTC), pair="BTC/USDT",
        snapshot_summary={}, analyses={}, debate_rounds=2,
    )
    assert dc.hash == "abc12345"
    assert dc.pnl is None
