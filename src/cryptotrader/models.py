"""All data models for CryptoTrader AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
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
    btc_tx_count: float = 0.0
    btc_active_addresses: float = 0.0
    btc_avg_fee_usd: float = 0.0
    btc_difficulty: float = 0.0
    # Track which providers returned real data vs fallback zeros
    data_quality: dict[str, bool] = field(default_factory=dict)


@dataclass
class NewsArticle:
    """A single news article with title, summary, and metadata."""

    title: str = ""
    summary: str = ""  # Lead paragraph or body excerpt (max ~500 chars)
    source: str = ""  # e.g. "coindesk", "cointelegraph"
    published: str = ""  # ISO date or human-readable date string


@dataclass
class NewsSentiment:
    headlines: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    key_events: list[str] = field(default_factory=list)
    social_buzz: float = 0.0
    articles: list[NewsArticle] = field(default_factory=list)


@dataclass
class MacroData:
    fed_rate: float = 0.0
    dxy: float = 0.0
    btc_dominance: float = 0.0
    fear_greed_index: int = 50
    etf_daily_net_inflow: float = 0.0
    etf_total_net_assets: float = 0.0
    etf_cum_net_inflow: float = 0.0
    vix: float = 0.0
    sp500: float = 0.0
    stablecoin_total_supply: float = 0.0
    btc_hashrate: float = 0.0
    yield_curve: float = 0.0
    m2_supply: float = 0.0
    cpi: float = 0.0
    etf_top_flows: list[dict] = field(default_factory=list)
    fear_greed_history: list[int] = field(default_factory=list)


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
    data_sufficiency: Literal["high", "medium", "low"] = "medium"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_mock: bool = False

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class TradeVerdict:
    action: Literal["long", "short", "hold", "close"]
    confidence: float = 0.0
    position_scale: float = 0.0
    divergence: float = 0.0
    reasoning: str = ""
    thesis: str = ""
    invalidation: str = ""

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.position_scale = max(0.0, min(1.0, self.position_scale))


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


class OrderStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING: {OrderStatus.SUBMITTED, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.SUBMITTED: {
        OrderStatus.FILLED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.FAILED,
    },
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

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError(f"Order amount must be > 0, got {self.amount}")
        if self.price < 0:
            raise ValueError(f"Order price must be >= 0, got {self.price}")


# ── Decision Journal Models (Section 8.2) ──


# ── Experience Memory Models ──


@dataclass
class ExperienceRule:
    """A single distilled experience rule from historical trading analysis."""

    pattern: str
    category: str  # "success_pattern" | "forbidden_zone"
    conditions: dict[str, list[str]] = field(default_factory=dict)  # {"regime_tags": [...]}
    rate: float = 0.0  # win_rate or loss_rate
    sample_count: int = 0
    maturity: str = "observation"  # "observation" | "hypothesis" | "rule"
    reason: str = ""
    regime_count: int = 1
    source: str = "live"  # "live" | "backtest" | "manual"
    source_session: str = ""  # backtest session ID


@dataclass
class ExperienceMemory:
    """Structured experience memory for an agent."""

    success_patterns: list[ExperienceRule] = field(default_factory=list)
    forbidden_zones: list[ExperienceRule] = field(default_factory=list)
    strategic_insights: list[str] = field(default_factory=list)
    updated_at: str = ""


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
    trace_id: str | None = None
