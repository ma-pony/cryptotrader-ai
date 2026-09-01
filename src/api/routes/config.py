"""数据库 RuntimeConfig 的整文档 CAS 与严格脱敏响应。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from api.routes.response_dto import JsonEntryOut, JsonValueOut, StrictOut, json_entries_out, json_value_out
from cryptotrader.configuration.catalog import (
    ConfigurationCatalog,
    configuration_catalog,
    validate_configuration_parameters,
)
from cryptotrader.execution.models import ExecutionBook  # noqa: TC001
from cryptotrader.execution_ownership import wait_for_owned
from cryptotrader.runtime_config.models import (
    AccountsConfig,
    ExecutionConfig,
    HitlConfig,
    InfrastructureConfig,
    LlmConfig,
    MarketDataConfig,
    NotificationConfig,
    ObservabilityConfig,
    RiskConfig,
    RuntimeConfigDocument,
    RuntimeConfigSnapshot,
    SchedulerConfig,
    SecurityConfig,
    SignalConfig,
    TriggerConfig,
    validate_runtime_document,
)
from cryptotrader.runtime_config.repository import (
    API_ACCESS_CREDENTIAL_REF,
    LLM_GATEWAY_CREDENTIAL_REF,
    NEWS_PROVIDER_CREDENTIAL_REF,
    CredentialState,
    RevisionConflict,
)
from cryptotrader.runtime_config.secrets import TokenPayload
from cryptotrader.venues.models import (
    ConnectionEnvironment,
    MarginMode,
    VenueConnection,
)

if TYPE_CHECKING:
    from cryptotrader.configuration.fields import ConfigurationField, FieldOption, LocalizedText

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
    canary_only: bool
    parameters: list[JsonEntryOut]


class ExecutionConfigOut(StrictOut):
    pairs: list[str]
    connections: list[VenueConnectionOut]
    books: list[ExecutionBookOut]
    live_order_execution_enabled: bool


class MarketDataConfigOut(StrictOut):
    source_id: str
    timeframe: str
    parameters: list[JsonEntryOut]
    news_credential_configured: bool
    news_credential_updated_at: datetime | None


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
    gateway_credential_configured: bool
    gateway_credential_updated_at: datetime | None


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
    evaluation_interval: str | None


class PositionConfigOut(StrictOut):
    max_single_pct: float
    max_total_exposure_pct: float
    max_margin_used_pct: float


class LossConfigOut(StrictOut):
    max_drawdown_pct: float


class RiskConfigOut(StrictOut):
    position: PositionConfigOut
    loss: LossConfigOut


class HitlConfigOut(StrictOut):
    approval_ttl_minutes: int


class SchedulerConfigOut(StrictOut):
    automation_enabled: bool
    enabled: bool
    interval_minutes: int
    daily_summary_hour: int


class TriggerConfigOut(StrictOut):
    enabled: bool
    max_rules: int
    ws_reconnect_max_s: int
    funding_rate_poll_interval_minutes: int


class NotificationConfigOut(StrictOut):
    webhook_url: str
    enabled: bool
    webhook_timeout: int
    events: list[str]


class InfrastructureConfigOut(StrictOut):
    redis_url: str


class SecurityConfigOut(StrictOut):
    enabled: bool
    access_credential_configured: bool
    access_credential_updated_at: datetime | None


class ObservabilityConfigOut(StrictOut):
    otlp_endpoint: str


class RuntimeDocumentOut(StrictOut):
    accounts: AccountsConfig
    security: SecurityConfigOut
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
    observability: ObservabilityConfigOut


class RuntimeConfigOut(StrictOut):
    revision: int
    updated_at: datetime
    apply_status: str
    applied_revision: int | None
    apply_error: str | None
    document: RuntimeDocumentOut


class LocalizedTextOut(StrictOut):
    zh_CN: str  # noqa: N815 - transport contract preserves locale code spelling.
    en_US: str  # noqa: N815 - transport contract preserves locale code spelling.


class FieldOptionOut(StrictOut):
    value: str
    label: LocalizedTextOut


class PluginFieldOut(StrictOut):
    key: str
    label: LocalizedTextOut
    description: LocalizedTextOut
    kind: Literal["text", "number", "integer", "boolean", "select", "string_list"]
    default_value: JsonValueOut
    required: bool
    minimum: float | int | None
    maximum: float | int | None
    exclusive_minimum: float | int | None
    exclusive_maximum: float | int | None
    step: float | int | None
    unit: str | None
    advanced: bool
    options: list[FieldOptionOut]


class EnvironmentDefinitionOut(StrictOut):
    id: str
    label: LocalizedTextOut
    capital_scope: Literal["simulated", "real"]


class CredentialFieldOut(StrictOut):
    key: str
    label: LocalizedTextOut
    description: LocalizedTextOut
    required: bool


class VenueCapabilitiesDefinitionOut(StrictOut):
    market_types: list[str]
    native_protection: bool
    hedge_mode: bool
    reduce_only: bool
    supported_order_types: list[str]
    account_reads: list[str]
    exit_operations: list[str]
    history_initial_days: int | None
    unknown_fields: list[str]


class VenueDefinitionOut(StrictOut):
    id: str
    label: LocalizedTextOut
    description: LocalizedTextOut
    environment: EnvironmentDefinitionOut
    fields: list[PluginFieldOut]
    credential_fields: list[CredentialFieldOut]
    margin_modes: list[str]
    leverage_minimum: int
    leverage_maximum: int | None
    account_read: bool
    capabilities: VenueCapabilitiesDefinitionOut | None


class PluginDefinitionOut(StrictOut):
    id: str
    label: LocalizedTextOut
    description: LocalizedTextOut
    fields: list[PluginFieldOut]
    environments: list[EnvironmentDefinitionOut]
    credential_fields: list[CredentialFieldOut]
    margin_modes: list[str]


class ConfigurationCatalogOut(StrictOut):
    components: list[PluginDefinitionOut]
    venues: list[PluginDefinitionOut]
    market_sources: list[PluginDefinitionOut]


class PutRuntimeConfigIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    expected_revision: int
    document: RuntimeDocumentIn


class PutTokenIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)

    expected_revision: int
    token: str


class TokenMutationOut(StrictOut):
    revision: int
    configured: bool
    updated_at: datetime


class VenueConnectionDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    id: str
    label: str
    adapter_id: str
    environment: ConnectionEnvironment
    enabled: bool
    leverage: int
    margin_mode: MarginMode
    canary_only: bool
    parameters: dict[str, Any]


class ExecutionConfigIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    connections: tuple[VenueConnectionDocumentIn, ...] = ()
    pairs: tuple[str, ...] = ()
    books: tuple[ExecutionBook, ...] = ()
    live_order_execution_enabled: bool = False


class RuntimeDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    security: SecurityConfig = SecurityConfig()
    market_data: MarketDataConfig
    llm: LlmConfig = LlmConfig()
    signals: SignalConfig
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfigIn
    hitl: HitlConfig = HitlConfig()
    accounts: AccountsConfig = AccountsConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    triggers: TriggerConfig = TriggerConfig()
    notifications: NotificationConfig = NotificationConfig()
    infrastructure: InfrastructureConfig = InfrastructureConfig()
    observability: ObservabilityConfig = ObservabilityConfig()


def require_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or getattr(runtime, "repository", None) is None:
        raise HTTPException(status_code=503, detail="Runtime configuration is unavailable")
    return runtime


def validate_document(runtime, document: RuntimeConfigDocument) -> None:
    try:
        validate_runtime_document(
            document,
            set(runtime.signal_registry.registered_ids()),
            set(runtime.venue_registry.registered_ids()),
            set(runtime.market_registry.registered_ids()),
        )
        validate_configuration_parameters(document)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Runtime configuration is invalid") from error


def _localized_text_out(value: LocalizedText) -> LocalizedTextOut:
    return LocalizedTextOut(zh_CN=value.zh_CN, en_US=value.en_US)


def _field_option_out(value: FieldOption) -> FieldOptionOut:
    return FieldOptionOut(value=value.value, label=_localized_text_out(value.label))


def _plugin_field_out(value: ConfigurationField) -> PluginFieldOut:
    return PluginFieldOut(
        key=value.key,
        label=_localized_text_out(value.label),
        description=_localized_text_out(value.description),
        kind=value.kind,
        default_value=json_value_out(value.default_value),
        required=value.required,
        minimum=value.minimum,
        maximum=value.maximum,
        exclusive_minimum=value.exclusive_minimum,
        exclusive_maximum=value.exclusive_maximum,
        step=value.step,
        unit=value.unit,
        advanced=value.advanced,
        options=[_field_option_out(option) for option in value.options],
    )


def _plugin_definition_out(value) -> PluginDefinitionOut:
    return PluginDefinitionOut(
        id=value.id,
        label=_localized_text_out(value.label),
        description=_localized_text_out(value.description),
        fields=[_plugin_field_out(field) for field in value.fields],
        environments=[_environment_out(item) for item in value.environments],
        credential_fields=[_credential_field_out(item) for item in value.credential_fields],
        margin_modes=list(value.margin_modes),
    )


def _environment_out(value) -> EnvironmentDefinitionOut:
    return EnvironmentDefinitionOut(
        id=value.id, label=_localized_text_out(value.label), capital_scope=value.capital_scope
    )


def _credential_field_out(value) -> CredentialFieldOut:
    return CredentialFieldOut(
        key=value.key,
        label=_localized_text_out(value.label),
        description=_localized_text_out(value.description),
        required=value.required,
    )


def venue_definition(adapter_id: str, environment: str) -> VenueDefinitionOut:
    definition = configuration_catalog().require_venue(adapter_id)
    selected = definition.for_environment(environment)
    capabilities = selected.capabilities
    return VenueDefinitionOut(
        id=selected.id,
        label=_localized_text_out(selected.label),
        description=_localized_text_out(selected.description),
        environment=_environment_out(definition.require_environment(environment)),
        fields=[_plugin_field_out(field) for field in selected.fields],
        credential_fields=[_credential_field_out(field) for field in selected.credential_fields],
        margin_modes=list(selected.margin_modes),
        leverage_minimum=selected.leverage_minimum,
        leverage_maximum=selected.leverage_maximum,
        account_read=capabilities is not None and {"balances", "positions", "orders"} <= capabilities.account_reads,
        capabilities=VenueCapabilitiesDefinitionOut(
            market_types=sorted(capabilities.market_types),
            native_protection=capabilities.native_protection,
            hedge_mode=capabilities.hedge_mode,
            reduce_only=capabilities.reduce_only,
            supported_order_types=sorted(capabilities.supported_order_types),
            account_reads=sorted(capabilities.account_reads),
            exit_operations=sorted(capabilities.exit_operations),
            history_initial_days=capabilities.history_initial_days,
            unknown_fields=sorted(capabilities.unknown_fields),
        )
        if capabilities is not None
        else None,
    )


def configuration_catalog_out(catalog: ConfigurationCatalog) -> ConfigurationCatalogOut:
    return ConfigurationCatalogOut(
        components=[_plugin_definition_out(item) for _, item in sorted(catalog.components.items())],
        venues=[_plugin_definition_out(item) for _, item in sorted(catalog.venues.items())],
        market_sources=[_plugin_definition_out(item) for _, item in sorted(catalog.market_sources.items())],
    )


def server_credential_ref(connection_id: str, adapter_id: str, environment: ConnectionEnvironment) -> str | None:
    definition = configuration_catalog().require_venue(adapter_id).for_environment(environment)
    return f"venue-connection:{connection_id}" if definition.credential_fields else None


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
                    else server_credential_ref(item.id, item.adapter_id, item.environment)
                ),
                leverage=item.leverage,
                margin_mode=item.margin_mode,
                canary_only=item.canary_only,
                parameters=item.parameters,
            )
            for item in body.execution.connections
        )
        return RuntimeConfigDocument(
            security=body.security,
            market_data=body.market_data,
            llm=body.llm,
            signals=body.signals,
            risk=body.risk,
            execution=ExecutionConfig(
                pairs=body.execution.pairs,
                connections=connections,
                books=body.execution.books,
                live_order_execution_enabled=body.execution.live_order_execution_enabled,
            ),
            hitl=body.hitl,
            accounts=body.accounts,
            # Only the explicit runtime automation resource owns the total switch.
            scheduler=body.scheduler.model_copy(update={"automation_enabled": current.scheduler.automation_enabled}),
            triggers=body.triggers,
            notifications=body.notifications,
            infrastructure=body.infrastructure,
            observability=body.observability,
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
    for connection_id, existing in current_connections.items():
        candidate = replacement_connections.get(connection_id)
        if candidate is None:
            continue
        if candidate.environment != existing.environment:
            raise HTTPException(status_code=422, detail="Connection environment cannot be changed")
        if existing.enabled and not candidate.enabled and _enabled_book_references(replacement, connection_id):
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
        canary_only=connection.canary_only,
        parameters=json_entries_out(connection.parameters),
    )


async def config_out(repository, snapshot) -> RuntimeConfigOut:
    document = snapshot.document
    llm_credential, api_credential, news_credential = await asyncio.gather(
        repository.token_state(LLM_GATEWAY_CREDENTIAL_REF),
        repository.token_state(API_ACCESS_CREDENTIAL_REF),
        repository.token_state(NEWS_PROVIDER_CREDENTIAL_REF),
    )
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
        apply_status=snapshot.apply_status,
        applied_revision=snapshot.applied_revision,
        apply_error=snapshot.apply_error,
        document=RuntimeDocumentOut(
            security=SecurityConfigOut(
                enabled=document.security.enabled,
                access_credential_configured=api_credential.configured,
                access_credential_updated_at=api_credential.updated_at,
            ),
            market_data=MarketDataConfigOut(
                source_id=document.market_data.source_id,
                timeframe=document.market_data.timeframe,
                parameters=json_entries_out(document.market_data.parameters),
                news_credential_configured=news_credential.configured,
                news_credential_updated_at=news_credential.updated_at,
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
                gateway_credential_configured=llm_credential.configured,
                gateway_credential_updated_at=llm_credential.updated_at,
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
                evaluation_interval=document.signals.evaluation_interval,
            ),
            risk=RiskConfigOut(
                position=PositionConfigOut(
                    max_single_pct=document.risk.position.max_single_pct,
                    max_total_exposure_pct=document.risk.position.max_total_exposure_pct,
                    max_margin_used_pct=document.risk.position.max_margin_used_pct,
                ),
                loss=LossConfigOut(
                    max_drawdown_pct=document.risk.loss.max_drawdown_pct,
                ),
            ),
            execution=ExecutionConfigOut(
                pairs=list(document.execution.pairs),
                connections=list(connections),
                books=books,
                live_order_execution_enabled=document.execution.live_order_execution_enabled,
            ),
            hitl=HitlConfigOut(approval_ttl_minutes=document.hitl.approval_ttl_minutes),
            accounts=document.accounts,
            scheduler=SchedulerConfigOut(
                automation_enabled=document.scheduler.automation_enabled,
                enabled=document.scheduler.enabled,
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
            ),
            infrastructure=InfrastructureConfigOut(redis_url=document.infrastructure.redis_url),
            observability=ObservabilityConfigOut(otlp_endpoint=document.observability.otlp_endpoint),
        ),
    )


async def apply_document(
    runtime,
    snapshot,
    expected_revision: int,
    document: RuntimeConfigDocument,
    refresh_owners,
    clear_owners=None,
):
    """Apply one desired document without ever silently executing the old graph."""
    ensure_expected_revision(snapshot, expected_revision)
    validate_connection_lifecycle(snapshot.document, document)
    validate_document(runtime, document)
    await validate_activation_prerequisites(runtime.repository, document)
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot

    candidate_snapshot = RuntimeConfigSnapshot(
        expected_revision + 1,
        document,
        datetime.now().astimezone(),
        apply_status="applied",
        applied_revision=snapshot.applied_revision,
    )
    try:
        candidate = await runtime.prepare_candidate(candidate_snapshot)
    except Exception:
        raise HTTPException(status_code=503, detail="Runtime configuration cannot be applied") from None
    async with application_barrier(runtime):
        try:
            from api.routes.account_operations import operation_service
            from cryptotrader.accounts.operations import removal_ids

            if removal_ids(snapshot.document, document):
                async with operation_service(runtime).removal_guard(snapshot.document, document):
                    saved = await runtime.repository.replace(expected_revision, document)
            else:
                saved = await runtime.repository.replace(expected_revision, document)
        except RevisionConflict as error:
            await candidate.close()
            raise HTTPException(status_code=409, detail="Runtime configuration changed; reload and retry") from error
        except Exception:
            await candidate.close()
            raise HTTPException(
                status_code=409,
                detail="账户移除被拒绝，请先停用并确认最新账户无持仓、无挂单",  # noqa: RUF001
            ) from None
        return await publish_pending_snapshot(
            runtime,
            saved,
            refresh_owners,
            candidate,
            clear_owners=clear_owners,
        )


@asynccontextmanager
async def application_barrier(runtime):
    """Use the Runtime application protocol; test doubles supply the same boundary."""
    async with runtime.application_barrier():
        yield


async def publish_pending_snapshot(runtime, pending, refresh_owners, candidate=None, *, clear_owners=None):
    """Publish an already persisted desired revision, or leave that revision explicitly failed."""
    try:
        prepared = candidate or await runtime.prepare_candidate(pending)
        await runtime.publish_candidate(prepared, pending)
        if refresh_owners is not None:
            await refresh_owners(pending)
        local_applied = RuntimeConfigSnapshot(
            pending.revision,
            pending.document,
            pending.updated_at,
            apply_status="applied",
            applied_revision=pending.revision,
        )
        await runtime.activate_applied(local_applied)
        return await runtime.repository.mark_applied(pending.revision)
    except BaseException as error:
        cleanup_error: BaseException | None = None
        try:
            await wait_for_owned(
                asyncio.create_task(_fail_pending_application(runtime, pending, candidate, clear_owners))
            )
        except BaseException as cleanup_failure:
            cleanup_error = cleanup_failure
        if isinstance(error, asyncio.CancelledError):
            raise error
        if isinstance(cleanup_error, asyncio.CancelledError):
            raise cleanup_error from None
        raise HTTPException(status_code=503, detail="Runtime configuration cannot be applied") from None


async def _fail_pending_application(runtime, pending, candidate, clear_owners) -> None:
    """Complete fail-closed cleanup independent of request cancellation."""
    cleanup_incomplete = False
    if clear_owners is not None:
        try:
            await clear_owners()
        except BaseException:
            cleanup_incomplete = True
    if candidate is not None:
        try:
            await candidate.close()
        except BaseException:
            cleanup_incomplete = True
    error = "runtime application failed"
    if cleanup_incomplete:
        error = "runtime application failed: cleanup incomplete"
    try:
        failed = await runtime.repository.mark_failed(pending.revision, error)
    except BaseException:
        from cryptotrader.runtime_config.models import RuntimeConfigSnapshot

        failed = RuntimeConfigSnapshot(
            pending.revision,
            pending.document,
            pending.updated_at,
            apply_status="failed",
            applied_revision=pending.applied_revision,
            apply_error=error,
        )
    try:
        await runtime.fail_closed(failed)
    except BaseException:
        # Runtime fail-closed owns its own graph.  The desired revision remains
        # failed even when a session's teardown reports an operational error.
        return


async def validate_activation_prerequisites(repository, document: RuntimeConfigDocument) -> None:
    required = []
    if document.security.enabled:
        required.append(API_ACCESS_CREDENTIAL_REF)
    for credential_ref in required:
        try:
            token = await repository.reveal_token(credential_ref)
        except Exception:
            raise HTTPException(status_code=422, detail="Required runtime credential is not configured") from None
        if not token.token.strip():
            raise HTTPException(status_code=422, detail="Required runtime credential is not configured")


@router.get("", response_model=RuntimeConfigOut)
async def get_config(request: Request) -> RuntimeConfigOut:
    runtime = require_runtime(request)
    snapshot = await runtime.repository.get_or_create()
    return await config_out(runtime.repository, snapshot)


@router.get("/catalog", response_model=ConfigurationCatalogOut)
async def get_configuration_catalog() -> ConfigurationCatalogOut:
    try:
        return configuration_catalog_out(configuration_catalog())
    except (TypeError, ValueError):
        raise HTTPException(status_code=503, detail="Backend-registered configuration is unavailable") from None


@router.put("", response_model=RuntimeConfigOut)
async def put_config(body: PutRuntimeConfigIn, request: Request) -> RuntimeConfigOut:
    runtime = require_runtime(request)
    current = await runtime.repository.get_or_create()
    ensure_expected_revision(current, body.expected_revision)
    document = document_from_input(body.document, current.document)
    refresh_owners = getattr(request.app.state, "refresh_runtime_owners", None)
    snapshot = await apply_document(
        runtime,
        current,
        body.expected_revision,
        document,
        refresh_owners,
        getattr(request.app.state, "clear_runtime_owners", None),
    )
    return await config_out(runtime.repository, snapshot)


@router.get("/catalog/venues/{adapter_id}", response_model=VenueDefinitionOut)
async def get_venue_definition(adapter_id: str, environment: str) -> VenueDefinitionOut:
    try:
        return venue_definition(adapter_id, environment)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Venue environment is invalid") from None


async def _put_runtime_token(
    body: PutTokenIn,
    request: Request,
    credential_ref: str,
) -> TokenMutationOut:
    runtime = require_runtime(request)
    current = await runtime.repository.get_or_create()
    ensure_expected_revision(current, body.expected_revision)
    refresh_owners = getattr(request.app.state, "refresh_runtime_owners", None)
    async with application_barrier(runtime):
        try:
            snapshot = await runtime.repository.put_token(
                body.expected_revision,
                credential_ref,
                TokenPayload(token=body.token),
            )
        except RevisionConflict as error:
            raise HTTPException(status_code=409, detail="Runtime configuration changed; reload and retry") from error
        await publish_pending_snapshot(
            runtime,
            snapshot,
            refresh_owners,
            clear_owners=getattr(request.app.state, "clear_runtime_owners", None),
        )
    return TokenMutationOut(revision=snapshot.revision, configured=True, updated_at=snapshot.updated_at)


@router.put("/credentials/llm-gateway", response_model=TokenMutationOut)
async def put_llm_gateway_token(body: PutTokenIn, request: Request) -> TokenMutationOut:
    return await _put_runtime_token(body, request, LLM_GATEWAY_CREDENTIAL_REF)


@router.put("/credentials/api-access", response_model=TokenMutationOut)
async def put_api_access_token(body: PutTokenIn, request: Request) -> TokenMutationOut:
    return await _put_runtime_token(body, request, API_ACCESS_CREDENTIAL_REF)


@router.put("/credentials/news-provider", response_model=TokenMutationOut)
async def put_news_provider_token(body: PutTokenIn, request: Request) -> TokenMutationOut:
    return await _put_runtime_token(body, request, NEWS_PROVIDER_CREDENTIAL_REF)
