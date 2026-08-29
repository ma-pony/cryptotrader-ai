"""数据库 RuntimeConfig 的整文档 CAS 与严格脱敏响应。"""

from __future__ import annotations

import asyncio
from datetime import datetime  # noqa: TC003 - Pydantic resolves this response field at runtime.
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from api.routes.response_dto import JsonEntryOut, StrictOut, json_entries_out
from cryptotrader.execution.models import ExecutionBook  # noqa: TC001
from cryptotrader.runtime_config.models import (
    ExecutionConfig,
    HitlConfig,
    InfrastructureConfig,
    LlmConfig,
    MarketDataConfig,
    NotificationConfig,
    RiskConfig,
    RuntimeConfigDocument,
    SchedulerConfig,
    SignalConfig,
    SystemConfig,
    TriggerConfig,
    validate_runtime_document,
)
from cryptotrader.runtime_config.repository import CredentialState, RevisionConflict
from cryptotrader.venues.models import (
    ConnectionEnvironment,
    MarginMode,
    VenueConnection,
)

router = APIRouter(prefix="/api/config", tags=["config"])


class ConnectionAllocationOut(StrictOut):
    connection_id: str
    enabled: bool
    weight: float


class ExecutionBookOut(StrictOut):
    id: str
    label: str
    capital_scope: str
    enabled: bool
    hitl_required: bool
    allocations: list[ConnectionAllocationOut]


class VenueConnectionOut(StrictOut):
    id: str
    label: str
    adapter_id: str
    environment: ConnectionEnvironment
    enabled: bool
    credential_configured: bool
    credential_updated_at: datetime | None
    leverage: int
    margin_mode: MarginMode
    parameters: list[JsonEntryOut]


class ExecutionConfigOut(StrictOut):
    connections: list[VenueConnectionOut]
    books: list[ExecutionBookOut]
    allocation_policy: str


class SystemConfigOut(StrictOut):
    active: bool


class MarketDataConfigOut(StrictOut):
    source_id: str
    parameters: list[JsonEntryOut]


class LlmRetryConfigOut(StrictOut):
    max_attempts: int
    retry_base_delay_s: float
    retry_backoff_factor: float
    retry_jitter: bool


class LlmModelCostConfigOut(StrictOut):
    name: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float


class LlmModelsConfigOut(StrictOut):
    analysis: str
    debate: str
    committee_summary: str
    tech_agent: str
    chain_agent: str
    news_agent: str
    macro_agent: str
    fallback: str
    timeout_seconds: int


class LlmConfigOut(StrictOut):
    base_url: str
    streaming_models: list[str]
    default_temperature: float
    timeout: int
    prompt_caching: bool
    retry: LlmRetryConfigOut
    model_costs: list[LlmModelCostConfigOut]
    models: LlmModelsConfigOut


class SignalComponentConfigOut(StrictOut):
    component_id: str
    enabled: bool
    weight: float
    parameters: list[JsonEntryOut]


class SignalConfigOut(StrictOut):
    components: list[SignalComponentConfigOut]
    neutral_threshold: float
    max_target_ratio: float
    atr_stop_multiplier: float
    reward_ratio: float
    hitl_required: bool


class PositionConfigOut(StrictOut):
    max_single_pct: float
    max_total_exposure_pct: float
    max_margin_used_pct: float
    max_correlated_positions: int
    max_same_direction_positions: int


class LossConfigOut(StrictOut):
    max_daily_loss_pct: float
    max_drawdown_pct: float
    max_cvar_95: float
    cvar_min_returns: int


class CooldownConfigOut(StrictOut):
    same_pair_minutes: int
    post_loss_minutes: int


class VolatilityConfigOut(StrictOut):
    flash_crash_threshold: float
    funding_rate_threshold: float
    flash_crash_lookback: int


class ExchangeCheckConfigOut(StrictOut):
    max_api_latency_ms: int
    health_check_interval_s: int


class RateLimitConfigOut(StrictOut):
    max_trades_per_hour: int
    max_trades_per_day: int


class RiskConfigOut(StrictOut):
    max_stop_loss_pct: float
    position: PositionConfigOut
    loss: LossConfigOut
    cooldown: CooldownConfigOut
    volatility: VolatilityConfigOut
    exchange: ExchangeCheckConfigOut
    rate_limit: RateLimitConfigOut


class HitlConfigOut(StrictOut):
    approval_ttl_minutes: int


class SchedulerConfigOut(StrictOut):
    enabled: bool
    pairs: list[str]
    interval_minutes: int
    daily_summary_hour: int


class TriggerConfigOut(StrictOut):
    enabled: bool
    max_rules: int
    ws_reconnect_max_s: int
    funding_rate_poll_interval_minutes: int


class TelegramConfigOut(StrictOut):
    enabled: bool
    chat_id: str


class NotificationConfigOut(StrictOut):
    webhook_url: str
    enabled: bool
    webhook_timeout: int
    events: list[str]
    telegram: TelegramConfigOut


class InfrastructureConfigOut(StrictOut):
    redis_url: str


class RuntimeDocumentOut(StrictOut):
    system: SystemConfigOut
    market_data: MarketDataConfigOut
    llm: LlmConfigOut
    signals: SignalConfigOut
    risk: RiskConfigOut
    execution: ExecutionConfigOut
    hitl: HitlConfigOut
    scheduler: SchedulerConfigOut
    triggers: TriggerConfigOut
    notifications: NotificationConfigOut
    infrastructure: InfrastructureConfigOut


class RuntimeConfigOut(StrictOut):
    revision: int
    updated_at: datetime
    setup_required: bool
    document: RuntimeDocumentOut


class PutRuntimeConfigIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    expected_revision: int
    document: RuntimeDocumentIn


class VenueConnectionDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    id: str
    label: str
    adapter_id: str
    environment: ConnectionEnvironment
    enabled: bool
    leverage: int
    margin_mode: MarginMode
    parameters: dict[str, Any]


class ExecutionConfigIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    connections: tuple[VenueConnectionDocumentIn, ...] = ()
    books: tuple[ExecutionBook, ...] = ()
    allocation_policy: str = "weighted"


class RuntimeDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    system: SystemConfig
    market_data: MarketDataConfig
    llm: LlmConfig = LlmConfig()
    signals: SignalConfig
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfigIn
    hitl: HitlConfig = HitlConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    triggers: TriggerConfig = TriggerConfig()
    notifications: NotificationConfig = NotificationConfig()
    infrastructure: InfrastructureConfig = InfrastructureConfig()


def require_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or getattr(runtime, "repository", None) is None:
        raise HTTPException(status_code=503, detail="Runtime configuration is unavailable")
    return runtime


def validate_document(runtime, document: RuntimeConfigDocument) -> None:
    try:
        validate_runtime_document(
            document,
            set(runtime.signal_registry.installed_ids()),
            set(runtime.venue_registry.installed_ids()),
            set(runtime.market_registry.installed_ids()),
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Runtime configuration is invalid") from error


def server_credential_ref(connection_id: str, environment: ConnectionEnvironment) -> str | None:
    return None if environment == "paper" else f"venue-connection:{connection_id}"


def document_from_input(body: RuntimeDocumentIn, current: RuntimeConfigDocument) -> RuntimeConfigDocument:
    current_connections = {connection.id: connection for connection in current.execution.connections}
    try:
        connections = tuple(
            VenueConnection(
                id=item.id,
                label=item.label,
                adapter_id=item.adapter_id,
                environment=item.environment,
                enabled=item.enabled,
                credential_ref=(
                    current_connections[item.id].credential_ref
                    if item.id in current_connections
                    else server_credential_ref(item.id, item.environment)
                ),
                leverage=item.leverage,
                margin_mode=item.margin_mode,
                parameters=item.parameters,
            )
            for item in body.execution.connections
        )
        return RuntimeConfigDocument(
            system=body.system,
            market_data=body.market_data,
            llm=body.llm,
            signals=body.signals,
            risk=body.risk,
            execution=ExecutionConfig(
                connections=connections,
                books=body.execution.books,
                allocation_policy=body.execution.allocation_policy,
            ),
            hitl=body.hitl,
            scheduler=body.scheduler,
            triggers=body.triggers,
            notifications=body.notifications,
            infrastructure=body.infrastructure,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Runtime configuration is invalid") from error


def ensure_expected_revision(snapshot, expected_revision: int) -> None:
    if snapshot.revision != expected_revision:
        raise HTTPException(status_code=409, detail="Runtime configuration changed; reload and retry")


def validate_connection_lifecycle(
    current: RuntimeConfigDocument,
    replacement: RuntimeConfigDocument,
) -> None:
    current_connections = {connection.id: connection for connection in current.execution.connections}
    replacement_connections = {connection.id: connection for connection in replacement.execution.connections}
    if not current_connections.keys() <= replacement_connections.keys():
        raise HTTPException(status_code=422, detail="Existing venue connections cannot be deleted")
    for connection_id, existing in current_connections.items():
        candidate = replacement_connections[connection_id]
        if candidate.environment != existing.environment:
            raise HTTPException(status_code=422, detail="Connection environment cannot be changed")
        if existing.enabled and not candidate.enabled and _enabled_book_references(current, connection_id):
            raise HTTPException(status_code=422, detail="Enabled book still references this connection")


def _enabled_book_references(document: RuntimeConfigDocument, connection_id: str) -> bool:
    return any(
        book.enabled
        and any(allocation.enabled and allocation.connection_id == connection_id for allocation in book.allocations)
        for book in document.execution.books
    )


async def connection_out(repository, connection: VenueConnection) -> VenueConnectionOut:
    state = (
        await repository.credential_state(connection.credential_ref)
        if connection.credential_ref is not None
        else CredentialState("", False, None)
    )
    return VenueConnectionOut(
        id=connection.id,
        label=connection.label,
        adapter_id=connection.adapter_id,
        environment=connection.environment,
        enabled=connection.enabled,
        credential_configured=state.configured,
        credential_updated_at=state.updated_at,
        leverage=connection.leverage,
        margin_mode=connection.margin_mode,
        parameters=json_entries_out(connection.parameters),
    )


async def config_out(repository, snapshot) -> RuntimeConfigOut:
    document = snapshot.document
    connections = await asyncio.gather(
        *(connection_out(repository, connection) for connection in document.execution.connections)
    )
    books = [
        ExecutionBookOut(
            id=book.id,
            label=book.label,
            capital_scope=book.capital_scope,
            enabled=book.enabled,
            hitl_required=book.hitl_required,
            allocations=[
                ConnectionAllocationOut(
                    connection_id=allocation.connection_id,
                    enabled=allocation.enabled,
                    weight=allocation.weight,
                )
                for allocation in book.allocations
            ],
        )
        for book in document.execution.books
    ]
    return RuntimeConfigOut(
        revision=snapshot.revision,
        updated_at=snapshot.updated_at,
        setup_required=snapshot.setup_required,
        document=RuntimeDocumentOut(
            system=SystemConfigOut(active=document.system.active),
            market_data=MarketDataConfigOut(
                source_id=document.market_data.source_id,
                parameters=json_entries_out(document.market_data.parameters),
            ),
            llm=LlmConfigOut(
                base_url=document.llm.base_url,
                streaming_models=list(document.llm.streaming_models),
                default_temperature=document.llm.default_temperature,
                timeout=document.llm.timeout,
                prompt_caching=document.llm.prompt_caching,
                retry=LlmRetryConfigOut(
                    max_attempts=document.llm.retry.max_attempts,
                    retry_base_delay_s=document.llm.retry.retry_base_delay_s,
                    retry_backoff_factor=document.llm.retry.retry_backoff_factor,
                    retry_jitter=document.llm.retry.retry_jitter,
                ),
                model_costs=[
                    LlmModelCostConfigOut(
                        name=item.name,
                        input_usd_per_mtok=item.input_usd_per_mtok,
                        output_usd_per_mtok=item.output_usd_per_mtok,
                    )
                    for item in document.llm.model_costs
                ],
                models=LlmModelsConfigOut(
                    analysis=document.llm.models.analysis,
                    debate=document.llm.models.debate,
                    committee_summary=document.llm.models.committee_summary,
                    tech_agent=document.llm.models.tech_agent,
                    chain_agent=document.llm.models.chain_agent,
                    news_agent=document.llm.models.news_agent,
                    macro_agent=document.llm.models.macro_agent,
                    fallback=document.llm.models.fallback,
                    timeout_seconds=document.llm.models.timeout_seconds,
                ),
            ),
            signals=SignalConfigOut(
                components=[
                    SignalComponentConfigOut(
                        component_id=item.component_id,
                        enabled=item.enabled,
                        weight=item.weight,
                        parameters=json_entries_out(item.parameters),
                    )
                    for item in document.signals.components
                ],
                neutral_threshold=document.signals.neutral_threshold,
                max_target_ratio=document.signals.max_target_ratio,
                atr_stop_multiplier=document.signals.atr_stop_multiplier,
                reward_ratio=document.signals.reward_ratio,
                hitl_required=document.signals.hitl_required,
            ),
            risk=RiskConfigOut(
                max_stop_loss_pct=document.risk.max_stop_loss_pct,
                position=PositionConfigOut(
                    max_single_pct=document.risk.position.max_single_pct,
                    max_total_exposure_pct=document.risk.position.max_total_exposure_pct,
                    max_margin_used_pct=document.risk.position.max_margin_used_pct,
                    max_correlated_positions=document.risk.position.max_correlated_positions,
                    max_same_direction_positions=document.risk.position.max_same_direction_positions,
                ),
                loss=LossConfigOut(
                    max_daily_loss_pct=document.risk.loss.max_daily_loss_pct,
                    max_drawdown_pct=document.risk.loss.max_drawdown_pct,
                    max_cvar_95=document.risk.loss.max_cvar_95,
                    cvar_min_returns=document.risk.loss.cvar_min_returns,
                ),
                cooldown=CooldownConfigOut(
                    same_pair_minutes=document.risk.cooldown.same_pair_minutes,
                    post_loss_minutes=document.risk.cooldown.post_loss_minutes,
                ),
                volatility=VolatilityConfigOut(
                    flash_crash_threshold=document.risk.volatility.flash_crash_threshold,
                    funding_rate_threshold=document.risk.volatility.funding_rate_threshold,
                    flash_crash_lookback=document.risk.volatility.flash_crash_lookback,
                ),
                exchange=ExchangeCheckConfigOut(
                    max_api_latency_ms=document.risk.exchange.max_api_latency_ms,
                    health_check_interval_s=document.risk.exchange.health_check_interval_s,
                ),
                rate_limit=RateLimitConfigOut(
                    max_trades_per_hour=document.risk.rate_limit.max_trades_per_hour,
                    max_trades_per_day=document.risk.rate_limit.max_trades_per_day,
                ),
            ),
            execution=ExecutionConfigOut(
                connections=list(connections),
                books=books,
                allocation_policy=document.execution.allocation_policy,
            ),
            hitl=HitlConfigOut(approval_ttl_minutes=document.hitl.approval_ttl_minutes),
            scheduler=SchedulerConfigOut(
                enabled=document.scheduler.enabled,
                pairs=list(document.scheduler.pairs),
                interval_minutes=document.scheduler.interval_minutes,
                daily_summary_hour=document.scheduler.daily_summary_hour,
            ),
            triggers=TriggerConfigOut(
                enabled=document.triggers.enabled,
                max_rules=document.triggers.max_rules,
                ws_reconnect_max_s=document.triggers.ws_reconnect_max_s,
                funding_rate_poll_interval_minutes=document.triggers.funding_rate_poll_interval_minutes,
            ),
            notifications=NotificationConfigOut(
                webhook_url=document.notifications.webhook_url,
                enabled=document.notifications.enabled,
                webhook_timeout=document.notifications.webhook_timeout,
                events=list(document.notifications.events),
                telegram=TelegramConfigOut(
                    enabled=document.notifications.telegram.enabled,
                    chat_id=document.notifications.telegram.chat_id,
                ),
            ),
            infrastructure=InfrastructureConfigOut(redis_url=document.infrastructure.redis_url),
        ),
    )


async def replace_document(runtime, snapshot, expected_revision: int, document: RuntimeConfigDocument):
    ensure_expected_revision(snapshot, expected_revision)
    validate_connection_lifecycle(snapshot.document, document)
    validate_document(runtime, document)
    try:
        return await runtime.repository.replace(expected_revision, document)
    except RevisionConflict as error:
        raise HTTPException(status_code=409, detail="Runtime configuration changed; reload and retry") from error


@router.get("", response_model=RuntimeConfigOut)
async def get_config(request: Request) -> RuntimeConfigOut:
    runtime = require_runtime(request)
    snapshot = await runtime.repository.get_or_create()
    return await config_out(runtime.repository, snapshot)


@router.put("", response_model=RuntimeConfigOut)
async def put_config(body: PutRuntimeConfigIn, request: Request) -> RuntimeConfigOut:
    runtime = require_runtime(request)
    current = await runtime.repository.get_or_create()
    ensure_expected_revision(current, body.expected_revision)
    document = document_from_input(body.document, current.document)
    snapshot = await replace_document(runtime, current, body.expected_revision, document)
    await runtime.reload_for_cycle()
    refresh_owners = getattr(request.app.state, "refresh_runtime_owners", None)
    if refresh_owners is not None:
        await refresh_owners()
    return await config_out(runtime.repository, snapshot)
