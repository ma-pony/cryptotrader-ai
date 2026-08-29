"""All data models for CryptoTrader AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, field_validator

from cryptotrader._compat import UTC, StrEnum

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
    reduce_only: bool = False
    status: OrderStatus = OrderStatus.PENDING
    venue_order_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError(f"Order amount must be > 0, got {self.amount}")
        if self.price < 0:
            raise ValueError(f"Order price must be >= 0, got {self.price}")
