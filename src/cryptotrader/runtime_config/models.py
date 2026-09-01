"""Strict immutable document model for the database runtime configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from cryptotrader.execution.models import ExecutionBook  # noqa: TC001
from cryptotrader.profiles.models import ComponentWeight, SignalProfile, validate_signal_profile
from cryptotrader.signals.presentation import interval_delta
from cryptotrader.venues.models import VenueConnection, _contains_secret_parameter_key


class _FrozenConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


def _freeze_parameters(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_parameters(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_parameters(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_parameters(item) for item in value)
    return value


def _thaw_parameters(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_parameters(item) for key, item in value.items()}
    if isinstance(value, tuple | frozenset):
        return [_thaw_parameters(item) for item in value]
    return value


class _ParameterConfigModel(_FrozenConfigModel):
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _freeze_nested_parameters(self):
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))
        return self

    @field_serializer("parameters")
    def _serialize_parameters(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _thaw_parameters(value)


class SecurityConfig(_FrozenConfigModel):
    enabled: bool = False


class MarketDataConfig(_ParameterConfigModel):
    source_id: str = "default"
    timeframe: str = "1h"

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, value):
        interval_delta(value)
        return value


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


class SignalComponentConfig(_ParameterConfigModel):
    component_id: str
    enabled: bool
    weight: float


class SignalConfig(_FrozenConfigModel):
    components: tuple[SignalComponentConfig, ...]
    neutral_threshold: float
    max_target_ratio: float
    atr_stop_multiplier: float
    reward_ratio: float
    evaluation_interval: str | None = None

    @field_validator("evaluation_interval")
    @classmethod
    def validate_interval(cls, value):
        if value is not None:
            interval_delta(value)
        return value

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
        )


class PositionConfig(_FrozenConfigModel):
    max_single_pct: float = 0.50
    max_total_exposure_pct: float = 1.00
    max_margin_used_pct: float = 0.40


class LossConfig(_FrozenConfigModel):
    max_drawdown_pct: float = 0.10


class RiskConfig(_FrozenConfigModel):
    position: PositionConfig = Field(default_factory=PositionConfig)
    loss: LossConfig = Field(default_factory=LossConfig)


class ExecutionConfig(_FrozenConfigModel):
    pairs: tuple[str, ...] = ()
    connections: tuple[VenueConnection, ...] = ()
    books: tuple[ExecutionBook, ...] = ()
    # This is deliberately a runtime-document switch rather than an adapter
    # setting: a live credential must never be enough to make writes possible.
    live_order_execution_enabled: bool = False

    @field_validator("pairs")
    @classmethod
    def validate_pairs(cls, values):
        from cryptotrader.pair import Pair

        pairs = tuple(Pair.parse(value).canonical() for value in values)
        if len(pairs) != len(set(pairs)):
            raise ValueError("duplicate execution pair")
        return pairs

    @model_validator(mode="before")
    @classmethod
    def _replace_rejected_connection_inputs(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        connections = value.get("connections")
        if type(connections) not in {list, tuple}:
            return value
        safe_connections = list(connections)
        changed = False
        for index, connection in enumerate(connections):
            if not isinstance(connection, Mapping) or not _contains_secret_parameter_key(connection.get("parameters")):
                continue
            safe_connections[index] = {
                "id": "rejected-connection",
                "label": "Rejected connection",
                "adapter_id": "paper",
                "environment": "paper",
                "enabled": False,
                "credential_ref": None,
                "leverage": 1,
                "margin_mode": "isolated",
                "canary_only": False,
                "parameters": {"secret": "[redacted]"},  # pragma: allowlist secret
            }
            changed = True
        return {**value, "connections": safe_connections} if changed else value

    @field_serializer("connections")
    def _serialize_connections(self, connections: tuple[VenueConnection, ...]) -> list[dict[str, Any]]:
        return [
            {
                "id": connection.id,
                "label": connection.label,
                "adapter_id": connection.adapter_id,
                "environment": connection.environment,
                "enabled": connection.enabled,
                "credential_ref": connection.credential_ref,
                "leverage": connection.leverage,
                "margin_mode": connection.margin_mode,
                "canary_only": connection.canary_only,
                "parameters": _thaw_parameters(connection.parameters),
            }
            for connection in connections
        ]


class HitlConfig(_FrozenConfigModel):
    approval_ttl_minutes: int = 60


class SchedulerConfig(_FrozenConfigModel):
    automation_enabled: bool = False
    enabled: bool = False
    interval_minutes: int = 240
    daily_summary_hour: int = 0


class TriggerConfig(_FrozenConfigModel):
    enabled: bool = False
    max_rules: int = 50
    ws_reconnect_max_s: int = 60
    funding_rate_poll_interval_minutes: int = 5


class NotificationConfig(_FrozenConfigModel):
    webhook_url: str = ""
    enabled: bool = True
    webhook_timeout: int = 5
    events: tuple[
        Literal[
            "approval_pending",
            "execution_failed",
            "protection_failed",
            "risk_adjusted",
            "component_failed",
            "connection_failed",
            "daily_summary",
        ],
        ...,
    ] = ("daily_summary",)


class InfrastructureConfig(_FrozenConfigModel):
    redis_url: str = ""


class ObservabilityConfig(_FrozenConfigModel):
    otlp_endpoint: str = ""


class AccountsConfig(_FrozenConfigModel):
    sync_interval_seconds: int = Field(default=60, ge=1, le=86400, strict=True)


class RuntimeConfigDocument(_FrozenConfigModel):
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    market_data: MarketDataConfig
    llm: LlmConfig = Field(default_factory=LlmConfig)
    signals: SignalConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig
    hitl: HitlConfig = Field(default_factory=HitlConfig)
    accounts: AccountsConfig = Field(default_factory=AccountsConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    triggers: TriggerConfig = Field(default_factory=TriggerConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


@dataclass(frozen=True)
class RuntimeConfigSnapshot:
    revision: int
    document: RuntimeConfigDocument
    updated_at: datetime
    apply_status: str = "applied"
    applied_revision: int | None = None
    apply_error: str | None = None

    def __post_init__(self) -> None:
        if self.apply_status == "applied" and self.applied_revision is None:
            object.__setattr__(self, "applied_revision", self.revision)

    @property
    def operational(self) -> bool:
        return self.apply_status == "applied" and self.applied_revision == self.revision


def validate_runtime_document(
    document: RuntimeConfigDocument,
    installed_signal_ids: set[str],
    installed_adapter_ids: set[str],
    installed_market_source_ids: set[str],
) -> None:
    """Validate the entire runtime document without silently repairing it."""

    validate_signal_profile(document.signals.to_profile(revision=0), installed_signal_ids)
    _validate_connection_ids(document.execution.connections, installed_adapter_ids)
    _validate_book_weights(document.execution.books)
    _validate_capital_scopes(document.execution.connections, document.execution.books)
    _validate_unique_enabled_membership(document.execution.books)
    _validate_capabilities(document, installed_market_source_ids)


def _validate_connection_ids(connections: tuple[VenueConnection, ...], installed_adapter_ids: set[str]) -> None:
    ids = [connection.id for connection in connections]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate venue connection id")
    missing = sorted({connection.adapter_id for connection in connections} - set(installed_adapter_ids))
    if missing:
        raise ValueError(f"unregistered adapter ids: {', '.join(missing)}")
    from cryptotrader.configuration.catalog import require_environment

    for connection in connections:
        require_environment(connection.adapter_id, connection.environment)


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
    from cryptotrader.configuration.catalog import require_environment

    for book in books:
        for allocation in book.allocations:
            connection = by_id.get(allocation.connection_id)
            if connection is None:
                raise ValueError(f"unknown connection_id: {allocation.connection_id}")
            if connection.canary_only:
                raise ValueError(f"canary_only connection {connection.id} cannot be allocated to an execution book")
            expected_scope = require_environment(connection.adapter_id, connection.environment).capital_scope
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


def _validate_capabilities(document: RuntimeConfigDocument, installed_market_source_ids: set[str]) -> None:
    if document.market_data.source_id not in installed_market_source_ids:
        raise ValueError(f"unregistered market source: {document.market_data.source_id}")
    redis_url = document.infrastructure.redis_url.strip()
    if redis_url:
        parsed_redis_url = urlparse(redis_url)
        if parsed_redis_url.scheme not in {"redis", "rediss"} or not parsed_redis_url.hostname:
            raise ValueError("infrastructure.redis_url must use redis:// or rediss://")
