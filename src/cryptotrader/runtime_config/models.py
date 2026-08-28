"""Strict immutable document model for the database runtime configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cryptotrader.execution.models import ExecutionBook  # noqa: TC001
from cryptotrader.profiles.models import ComponentWeight, SignalProfile, validate_signal_profile
from cryptotrader.venues.models import VenueConnection  # noqa: TC001


class _FrozenConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SystemConfig(_FrozenConfigModel):
    active: bool = False


class MarketDataConfig(_FrozenConfigModel):
    source_id: str = "default"
    parameters: dict[str, Any] = Field(default_factory=dict)


class LlmRetryConfig(_FrozenConfigModel):
    max_attempts: int = 3
    retry_base_delay_s: float = 1.0
    retry_backoff_factor: float = 2.0
    retry_jitter: bool = True


class LlmModelCostConfig(_FrozenConfigModel):
    name: str
    input_usd_per_mtok: float = 0.0
    output_usd_per_mtok: float = 0.0


class LlmModelsConfig(_FrozenConfigModel):
    analysis: str = "gemini-3-flash"
    debate: str = "gemini-3-flash"
    committee_summary: str = "gpt-5.4"
    tech_agent: str = "gemini-3-flash"
    chain_agent: str = "gemini-3.1-pro"
    news_agent: str = "gemini-3.1-pro"
    macro_agent: str = "gemini-3.1-pro"
    fallback: str = "deepseek-chat"
    timeout_seconds: int = 90


class LlmConfig(_FrozenConfigModel):
    base_url: str = ""
    streaming_models: tuple[str, ...] = ()
    default_temperature: float = 0.2
    timeout: int = 120
    prompt_caching: bool = True
    retry: LlmRetryConfig = Field(default_factory=LlmRetryConfig)
    model_costs: tuple[LlmModelCostConfig, ...] = ()
    models: LlmModelsConfig = Field(default_factory=LlmModelsConfig)


class SignalComponentConfig(_FrozenConfigModel):
    component_id: str
    enabled: bool
    weight: float
    parameters: dict[str, Any] = Field(default_factory=dict)


class SignalConfig(_FrozenConfigModel):
    components: tuple[SignalComponentConfig, ...]
    neutral_threshold: float
    max_target_ratio: float
    atr_stop_multiplier: float
    reward_ratio: float
    hitl_required: bool = False

    def to_profile(self, revision: int) -> SignalProfile:
        return SignalProfile(
            revision=revision,
            components=tuple(
                ComponentWeight(component.component_id, component.enabled, component.weight)
                for component in self.components
            ),
            neutral_threshold=self.neutral_threshold,
            max_target_ratio=self.max_target_ratio,
            atr_stop_multiplier=self.atr_stop_multiplier,
            reward_ratio=self.reward_ratio,
            hitl_required=self.hitl_required,
        )


class PositionConfig(_FrozenConfigModel):
    max_single_pct: float = 0.50
    max_total_exposure_pct: float = 1.00
    max_margin_used_pct: float = 0.40
    max_correlated_positions: int = 2
    max_same_direction_positions: int = 3


class LossConfig(_FrozenConfigModel):
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_cvar_95: float = 0.05
    cvar_min_returns: int = 20


class CooldownConfig(_FrozenConfigModel):
    same_pair_minutes: int = 60
    post_loss_minutes: int = 120


class VolatilityConfig(_FrozenConfigModel):
    flash_crash_threshold: float = 0.05
    funding_rate_threshold: float = 0.005
    flash_crash_lookback: int = 10


class ExchangeCheckConfig(_FrozenConfigModel):
    max_api_latency_ms: int = 2000
    health_check_interval_s: int = 30


class RateLimitConfig(_FrozenConfigModel):
    max_trades_per_hour: int = 6
    max_trades_per_day: int = 20


class RiskConfig(_FrozenConfigModel):
    max_stop_loss_pct: float = 0.05
    position: PositionConfig = Field(default_factory=PositionConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    cooldown: CooldownConfig = Field(default_factory=CooldownConfig)
    volatility: VolatilityConfig = Field(default_factory=VolatilityConfig)
    exchange: ExchangeCheckConfig = Field(default_factory=ExchangeCheckConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class ExecutionConfig(_FrozenConfigModel):
    connections: tuple[VenueConnection, ...] = ()
    books: tuple[ExecutionBook, ...] = ()
    allocation_policy: str = "weighted"


class HitlConfig(_FrozenConfigModel):
    approval_ttl_minutes: int = 60


class SchedulerConfig(_FrozenConfigModel):
    enabled: bool = False
    pairs: tuple[str, ...] = ()
    interval_minutes: int = 240
    daily_summary_hour: int = 0


class TriggerConfig(_FrozenConfigModel):
    enabled: bool = False
    max_rules: int = 50
    ws_reconnect_max_s: int = 60
    funding_rate_poll_interval_minutes: int = 5


class TelegramConfig(_FrozenConfigModel):
    enabled: bool = False
    chat_id: str = ""


class NotificationConfig(_FrozenConfigModel):
    webhook_url: str = ""
    enabled: bool = True
    webhook_timeout: int = 5
    events: tuple[str, ...] = ("trade", "rejection", "circuit_breaker", "reconcile_mismatch", "daily_summary")
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


class InfrastructureConfig(_FrozenConfigModel):
    redis_url: str = ""


class RuntimeConfigDocument(_FrozenConfigModel):
    system: SystemConfig
    market_data: MarketDataConfig
    llm: LlmConfig = Field(default_factory=LlmConfig)
    signals: SignalConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig
    hitl: HitlConfig = Field(default_factory=HitlConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    triggers: TriggerConfig = Field(default_factory=TriggerConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)


@dataclass(frozen=True)
class RuntimeConfigSnapshot:
    revision: int
    document: RuntimeConfigDocument
    updated_at: datetime

    @property
    def setup_required(self) -> bool:
        return not self.document.system.active


def validate_runtime_document(
    document: RuntimeConfigDocument,
    installed_signal_ids: set[str],
    installed_adapter_ids: set[str],
) -> None:
    """Validate the entire runtime document without silently repairing it."""

    validate_signal_profile(document.signals.to_profile(revision=0), installed_signal_ids)
    _validate_connection_ids(document.execution.connections, installed_adapter_ids)
    _validate_book_weights(document.execution.books)
    _validate_capital_scopes(document.execution.connections, document.execution.books)
    _validate_unique_enabled_membership(document.execution.books)
    _validate_active_document(document)


def _validate_connection_ids(connections: tuple[VenueConnection, ...], installed_adapter_ids: set[str]) -> None:
    ids = [connection.id for connection in connections]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate venue connection id")
    missing = sorted({connection.adapter_id for connection in connections} - set(installed_adapter_ids))
    if missing:
        raise ValueError(f"uninstalled adapter ids: {', '.join(missing)}")


def _validate_book_weights(books: tuple[ExecutionBook, ...]) -> None:
    book_ids = [book.id for book in books]
    if len(book_ids) != len(set(book_ids)):
        raise ValueError("duplicate execution book id")
    for book in books:
        allocation_ids = [allocation.connection_id for allocation in book.allocations]
        if len(allocation_ids) != len(set(allocation_ids)):
            raise ValueError(f"duplicate connection allocation in book {book.id}")
        if not book.enabled:
            continue
        enabled = tuple(allocation for allocation in book.allocations if allocation.enabled)
        if not enabled:
            raise ValueError(f"enabled execution book {book.id} requires an enabled allocation")
        total_weight = math.fsum(allocation.weight for allocation in enabled)
        if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"enabled allocation weights must sum to 1.0, got {total_weight}")


def _validate_capital_scopes(connections: tuple[VenueConnection, ...], books: tuple[ExecutionBook, ...]) -> None:
    by_id = {connection.id: connection for connection in connections}
    simulated_environments = {"paper", "demo", "testnet"}
    for book in books:
        for allocation in book.allocations:
            connection = by_id.get(allocation.connection_id)
            if connection is None:
                raise ValueError(f"unknown connection_id: {allocation.connection_id}")
            expected_scope = "simulated" if connection.environment in simulated_environments else "real"
            if book.capital_scope != expected_scope:
                raise ValueError(
                    f"connection {connection.id} environment {connection.environment} is incompatible with "
                    f"book capital_scope {book.capital_scope}"
                )


def _validate_unique_enabled_membership(books: tuple[ExecutionBook, ...]) -> None:
    memberships: set[str] = set()
    for book in books:
        if not book.enabled:
            continue
        for allocation in book.allocations:
            if not allocation.enabled:
                continue
            if allocation.connection_id in memberships:
                raise ValueError("a connection can belong to only one enabled book")
            memberships.add(allocation.connection_id)


def _validate_active_document(document: RuntimeConfigDocument) -> None:
    if not document.system.active:
        return
    if not document.market_data.source_id.strip():
        raise ValueError("active document requires a resolvable market source")
    if not any(component.enabled for component in document.signals.components):
        raise ValueError("active document requires an enabled signal component")
    if not any(book.enabled for book in document.execution.books):
        raise ValueError("active document requires an enabled execution book")
    for connection in document.execution.connections:
        if connection.enabled and connection.environment != "paper" and not connection.credential_ref:
            raise ValueError(f"enabled {connection.environment} connection {connection.id} requires credential_ref")
