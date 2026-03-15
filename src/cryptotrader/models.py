"""All data models for CryptoTrader AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    import pandas as pd

# ── External API Response Schema Models (Section 7.3) ──

_NEWS_TITLE_MAX_LEN = 2000


class NewsHeadlineResponse(BaseModel):
    """Pydantic schema for validating a single news headline from external APIs.

    Enforces non-empty title and maximum length constraints so that malformed
    or injected payloads are rejected before reaching the LLM pipeline.
    """

    title: str
    source: str = ""
    published: str = ""

    @field_validator("title")
    @classmethod
    def title_must_be_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("title must not be empty or whitespace-only")
        if len(stripped) > _NEWS_TITLE_MAX_LEN:
            raise ValueError(f"title length {len(stripped)} exceeds maximum {_NEWS_TITLE_MAX_LEN}")
        return stripped


class OnchainMetricResponse(BaseModel):
    """Pydantic schema for validating a single on-chain metric from external APIs.

    Enforces non-empty metric_name and non-negative value so that corrupted
    or structurally invalid provider responses are rejected at ingestion.
    """

    metric_name: str
    value: float
    source: str = ""

    @field_validator("metric_name")
    @classmethod
    def metric_name_must_be_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("metric_name must not be empty or whitespace-only")
        return stripped

    @field_validator("value")
    @classmethod
    def value_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"value must be >= 0, got {v}")
        return v


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
    verdict_source: Literal["ai", "weighted", "hold_all_mock"] = "ai"

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


# ── Observability Value Objects ──


@dataclass
class ConsensusMetrics:
    """Snapshot of debate_gate consensus computation for a single decision cycle."""

    strength: float  # abs(mean_score) * (1 - pstdev)
    mean_score: float  # mean of all agent scores
    dispersion: float  # standard deviation of agent scores
    skip_threshold: float  # consensus_skip_threshold from config
    confusion_threshold: float  # confusion_skip_threshold from config


@dataclass
class NodeTraceEntry:
    """Execution trace record for a single pipeline node."""

    node: str
    duration_ms: int
    summary: str  # node output summary, max 200 chars


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
    # Observability fields (tasks 1.2)
    consensus_metrics: ConsensusMetrics | None = None
    verdict_source: Literal["ai", "weighted", "hold_all_mock"] = "ai"
    experience_memory: dict[str, Any] = field(default_factory=dict)
    node_trace: list[NodeTraceEntry] = field(default_factory=list)
    debate_skip_reason: str = ""
