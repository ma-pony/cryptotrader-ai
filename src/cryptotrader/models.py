"""All data models for CryptoTrader AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

import pandas as pd


# ── Data Layer Models (Section 5.2) ──

@dataclass
class MarketData:
    pair: str
    ohlcv: pd.DataFrame
    ticker: dict[str, Any]
    funding_rate: float
    orderbook_imbalance: float
    volatility: float


@dataclass
class OnchainData:
    exchange_netflow: float = 0.0
    whale_transfers: list[dict] = field(default_factory=list)
    open_interest: float = 0.0
    liquidations_24h: dict[str, float] = field(default_factory=dict)
    defi_tvl: float = 0.0
    defi_tvl_change_7d: float = 0.0


@dataclass
class NewsSentiment:
    headlines: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    key_events: list[str] = field(default_factory=list)
    social_buzz: float = 0.0


@dataclass
class MacroData:
    fed_rate: float = 0.0
    dxy: float = 0.0
    btc_dominance: float = 0.0
    fear_greed_index: int = 50


@dataclass
class DataSnapshot:
    timestamp: datetime
    pair: str
    market: MarketData
    onchain: OnchainData
    news: NewsSentiment
    macro: MacroData


# ── Intelligence Layer Models (Section 4.3, 4.5) ──

@dataclass
class AgentAnalysis:
    agent_id: str
    pair: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str
    key_factors: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    data_points: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class TradeVerdict:
    action: Literal["long", "short", "hold"]
    confidence: float = 0.0
    position_scale: float = 0.0
    divergence: float = 0.0
    reasoning: str = ""


# ── Risk Layer Models (Section 6) ──

@dataclass
class CheckResult:
    passed: bool
    reason: str = ""


@dataclass
class GateResult:
    passed: bool
    rejected_by: str = ""
    reason: str = ""


# ── Execution Layer Models (Section 7.1) ──

class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING: {OrderStatus.SUBMITTED, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.SUBMITTED: {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.FILLED, OrderStatus.CANCELLED},
}


@dataclass
class Order:
    pair: str
    side: Literal["buy", "sell"]
    amount: float
    price: float
    order_type: Literal["market", "limit"] = "market"
    status: OrderStatus = OrderStatus.PENDING
    exchange_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ── Decision Journal Models (Section 8.2) ──

@dataclass
class DecisionCommit:
    hash: str
    parent_hash: str | None
    timestamp: datetime
    pair: str
    snapshot_summary: dict[str, Any]
    analyses: dict[str, AgentAnalysis]
    debate_rounds: int
    challenges: list[dict] = field(default_factory=list)
    divergence: float = 0.0
    verdict: TradeVerdict | None = None
    risk_gate: GateResult | None = None
    order: Order | None = None
    fill_price: float | None = None
    slippage: float | None = None
    portfolio_after: dict[str, Any] = field(default_factory=dict)
    pnl: float | None = None
    retrospective: str | None = None
