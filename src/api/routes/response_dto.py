"""Strict credential-safe response DTOs shared by runtime APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from api.routes.portfolio_books import (
    BookPortfolioOut,
    BookRiskStateOut,
    ConnectionPortfolioOut,
    book_portfolio_out,
    connection_portfolio_out,
)
from cryptotrader.pair import Pair
from cryptotrader.signals.presentation import EvaluationReference, ResultBlock, SignalUsage  # noqa: TC001


class StrictOut(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSignalOut(StrictOut):
    component_id: str
    direction: str
    confidence: float
    reasoning: str
    details: list[JsonEntryOut]
    blocks: list[ResultBlock]
    evaluation_reference: EvaluationReference | None
    status: Literal["completed", "skipped", "failed"]
    duration_ms: int | None
    usage: SignalUsage | None
    cost: Decimal | None


class ComponentContributionOut(StrictOut):
    component_id: str
    weight: float
    signed_score: float
    weighted_score: float


class FusedSignalOut(StrictOut):
    score: float
    reasoning: str
    contributions: list[ComponentContributionOut]


class TargetPositionOut(StrictOut):
    side: str
    size_ratio: float


class SharedSignalsOut(StrictOut):
    components: list[ComponentSignalOut]
    fused: FusedSignalOut | None
    target_position: TargetPositionOut | None


class BookHitlOut(StrictOut):
    approval_id: str | None
    status: str
    config_revision: int


class BookExecutionSummaryOut(StrictOut):
    status: str
    requires_attention: bool
    reallocated: bool


class CycleConnectionOut(StrictOut):
    connection_id: str
    portfolio_before: ConnectionPortfolioOut | None
    portfolio_after: ConnectionPortfolioOut | None
    risk: ConnectionRiskDecisionOut | None
    plan: ConnectionExecutionPlanOut | None
    execution: ConnectionExecutionResultOut | None
    unavailable: bool


class BookFailureOut(StrictOut):
    stage: str


class BookCycleOut(StrictOut):
    book_id: str
    capital_scope: Literal["simulated", "real"]
    config_revision: int
    pair: str
    market_type: str
    status: str
    hitl: BookHitlOut
    failure: BookFailureOut | None
    requested_target_exposure: str | None
    target_exposure: str | None
    risk: BookRiskDecisionOut | None
    ready: bool | None
    errors: list[str]
    execution: BookExecutionSummaryOut | None
    portfolio_before: BookPortfolioOut | None
    portfolio_after: BookPortfolioOut | None
    portfolio_after_available: bool | None
    reconciliation_required: bool | None
    connections: list[CycleConnectionOut]


class CycleOut(StrictOut):
    cycle_id: str
    config_revision: int
    market_data_source_id: str
    shared_signals: SharedSignalsOut
    books: list[BookCycleOut]
    cycle_status: str
    execution_status: str
    requires_attention: bool
    created_at: datetime


class PaginatedCyclesOut(StrictOut):
    items: list[CycleOut]
    total: int
    page: int
    size: int
    has_next: bool


class JsonEntryOut(StrictOut):
    key: str
    value: JsonValueOut


class JsonValueOut(StrictOut):
    kind: Literal["null", "boolean", "number", "string", "datetime", "pair", "array", "object"]
    boolean_value: bool | None = None
    number_value: str | None = None
    string_value: str | None = None
    datetime_value: datetime | None = None
    pair_value: str | None = None
    items: list[JsonValueOut] = []
    entries: list[JsonEntryOut] = []


JsonEntryOut.model_rebuild()


def json_entries_out(value: Mapping[str, Any]) -> list[JsonEntryOut]:
    return [JsonEntryOut(key=str(key), value=json_value_out(item)) for key, item in sorted(value.items())]


def json_value_out(value: Any) -> JsonValueOut:  # noqa: C901
    if value is None:
        return JsonValueOut(kind="null")
    if type(value) is bool:
        return JsonValueOut(kind="boolean", boolean_value=value)
    if isinstance(value, int | float | Decimal):
        return JsonValueOut(kind="number", number_value=str(value))
    if type(value) is str:
        return JsonValueOut(kind="string", string_value=value)
    if isinstance(value, datetime):
        return JsonValueOut(kind="datetime", datetime_value=value)
    if isinstance(value, Pair):
        return JsonValueOut(kind="pair", pair_value=value.canonical())
    if isinstance(value, Mapping):
        return JsonValueOut(kind="object", entries=json_entries_out(value))
    if isinstance(value, list | tuple):
        return JsonValueOut(kind="array", items=[json_value_out(item) for item in value])
    if isinstance(value, set | frozenset):
        return JsonValueOut(kind="array", items=[json_value_out(item) for item in sorted(value, key=repr)])
    if is_dataclass(value):
        return JsonValueOut(
            kind="object",
            entries=json_entries_out({field.name: getattr(value, field.name) for field in fields(value)}),
        )
    return JsonValueOut(kind="string", string_value=str(value))


class PairSymbolOut(StrictOut):
    symbol: str


class VenueCapabilitiesOut(StrictOut):
    market_types: list[str]
    native_protection: bool
    hedge_mode: bool
    reduce_only: bool
    supported_order_types: list[str]
    account_reads: list[str]
    exit_operations: list[str]
    history_initial_days: int | None
    unknown_fields: list[str]


class VenueQuoteOut(StrictOut):
    pair: PairSymbolOut
    bid: str
    ask: str
    last: str


class ConnectionTargetOut(StrictOut):
    book_id: str
    connection_id: str
    weight: str
    book_equity: str | None
    target_exposure: str
    target_signed_notional: str


class BookRiskDecisionOut(StrictOut):
    passed: bool
    requested_target_exposure: str
    capped_target_exposure: str
    connection_weights: list[str]
    connection_targets: list[ConnectionTargetOut]
    rejected_by: str
    reason: str
    cap_source: str
    state: BookRiskStateOut | None


class ConnectionRiskDecisionOut(StrictOut):
    connection_id: str
    passed: bool
    risk_increase: bool
    reason: str
    operation: str


class ConnectionExecutionPlanOut(StrictOut):
    book_id: str
    connection_id: str
    pair: PairSymbolOut
    current_signed_notional: str
    target_signed_notional: str
    delta_signed_notional: str
    current_signed_amount: str
    target_signed_amount: str
    delta_signed_amount: str
    post_fill_signed_amount: str
    quote: VenueQuoteOut
    execution_price: str
    amount: str
    side: str
    reduce_only: bool
    market_type: str
    stop_loss: str | None
    take_profit: str | None
    old_protection_ids: list[str]
    capabilities: VenueCapabilitiesOut


class BookExecutionProposalOut(StrictOut):
    version: Literal[1] = 1
    book_id: str
    capital_scope: Literal["simulated", "real"]
    config_revision: int
    pair: PairSymbolOut
    requested_target_exposure: str
    target_exposure: str
    risk: BookRiskDecisionOut
    connection_risks: list[ConnectionRiskDecisionOut]
    connection_plans: list[ConnectionExecutionPlanOut]
    unavailable_connections: list[str]
    errors: list[str]
    ready: bool


class NormalizedOrderOut(StrictOut):
    id: str
    pair: PairSymbolOut
    side: str
    order_type: str
    amount: str
    filled_amount: str
    average_price: str | None
    status: str
    reduce_only: bool
    client_order_id: str | None


class ProtectionStateOut(StrictOut):
    actual_order_ids: list[str]
    protection_ids: list[str]
    pair: PairSymbolOut
    position_side: str
    amount: str
    stop_loss: str | None
    take_profit: str | None
    active: bool
    triggered: bool


class ExecutionPositionOut(StrictOut):
    pair: PairSymbolOut
    signed_amount: str
    signed_notional: str
    entry_price: str | None


class ExecutionFinalPositionOut(StrictOut):
    position: ExecutionPositionOut
    protected: bool
    protection_ids: list[str]
    protections: list[ProtectionStateOut]


class CompensationResultOut(StrictOut):
    attempted: bool
    succeeded: bool
    order: NormalizedOrderOut | None
    operation: str
    safe_signed_amount: str | None
    required_protection: ProtectionStateOut | None


class ConnectionExecutionResultOut(StrictOut):
    book_id: str
    connection_id: str
    pair: PairSymbolOut
    target_signed_notional: str
    target_signed_amount: str
    status: str
    orders: list[NormalizedOrderOut]
    protection: ProtectionStateOut | None
    compensation: CompensationResultOut
    final_position: ExecutionFinalPositionOut | None
    error_operation: str
    requires_attention: bool
    trace: list[str]
    execution_quote: VenueQuoteOut | None
    quantity_frozen: bool | None


def pair_out(value) -> PairSymbolOut:
    return PairSymbolOut(symbol=value.canonical())


def capabilities_out(value) -> VenueCapabilitiesOut:
    return VenueCapabilitiesOut(
        market_types=sorted(value.market_types),
        native_protection=value.native_protection,
        hedge_mode=value.hedge_mode,
        reduce_only=value.reduce_only,
        supported_order_types=sorted(value.supported_order_types),
        account_reads=sorted(value.account_reads),
        exit_operations=sorted(value.exit_operations),
        history_initial_days=value.history_initial_days,
        unknown_fields=sorted(value.unknown_fields),
    )


def quote_out(value) -> VenueQuoteOut:
    return VenueQuoteOut(pair=pair_out(value.pair), bid=str(value.bid), ask=str(value.ask), last=str(value.last))


def connection_target_out(value) -> ConnectionTargetOut:
    return ConnectionTargetOut(
        book_id=value.book_id,
        connection_id=value.connection_id,
        weight=str(value.weight),
        book_equity=None if value.book_equity is None else str(value.book_equity),
        target_exposure=str(value.target_exposure),
        target_signed_notional=str(value.target_signed_notional),
    )


def book_risk_out(value) -> BookRiskDecisionOut:
    from api.routes.portfolio_books import risk_state_out

    state = risk_state_out(value.state)
    return BookRiskDecisionOut(
        passed=value.passed,
        requested_target_exposure=str(value.requested_target_exposure),
        capped_target_exposure=str(value.capped_target_exposure),
        connection_weights=[str(item) for item in value.connection_weights],
        connection_targets=[connection_target_out(item) for item in value.connection_targets],
        rejected_by=value.rejected_by,
        reason=value.reason,
        cap_source=value.cap_source,
        state=state,
    )


def connection_risk_out(value) -> ConnectionRiskDecisionOut:
    return ConnectionRiskDecisionOut(
        connection_id=value.connection_id,
        passed=value.passed,
        risk_increase=value.risk_increase,
        reason=value.reason,
        operation=value.operation,
    )


def plan_out(value) -> ConnectionExecutionPlanOut:
    return ConnectionExecutionPlanOut(
        book_id=value.book_id,
        connection_id=value.connection_id,
        pair=pair_out(value.pair),
        current_signed_notional=str(value.current_signed_notional),
        target_signed_notional=str(value.target_signed_notional),
        delta_signed_notional=str(value.delta_signed_notional),
        current_signed_amount=str(value.current_signed_amount),
        target_signed_amount=str(value.target_signed_amount),
        delta_signed_amount=str(value.delta_signed_amount),
        post_fill_signed_amount=str(value.post_fill_signed_amount),
        quote=quote_out(value.quote),
        execution_price=str(value.execution_price),
        amount=str(value.amount),
        side=value.side,
        reduce_only=value.reduce_only,
        market_type=value.market_type,
        stop_loss=None if value.stop_loss is None else str(value.stop_loss),
        take_profit=None if value.take_profit is None else str(value.take_profit),
        old_protection_ids=list(value.old_protection_ids),
        capabilities=capabilities_out(value.capabilities),
    )


def proposal_out(value) -> BookExecutionProposalOut:
    return BookExecutionProposalOut(
        book_id=value.book_id,
        capital_scope=value.capital_scope,
        config_revision=value.config_revision,
        pair=pair_out(value.pair),
        requested_target_exposure=str(value.requested_target_exposure),
        target_exposure=str(value.target_exposure),
        risk=book_risk_out(value.risk),
        connection_risks=[connection_risk_out(item) for item in value.connection_risks],
        connection_plans=[plan_out(item) for item in value.connection_plans],
        unavailable_connections=list(value.unavailable_connections),
        errors=list(value.errors),
        ready=value.ready,
    )


def order_out(value) -> NormalizedOrderOut:
    return NormalizedOrderOut(
        id=value.id,
        pair=pair_out(value.pair),
        side=value.side,
        order_type=value.order_type,
        amount=str(value.amount),
        filled_amount=str(value.filled_amount),
        average_price=None if value.average_price is None else str(value.average_price),
        status=value.status,
        reduce_only=value.reduce_only,
        client_order_id=value.client_order_id,
    )


def protection_out(value) -> ProtectionStateOut:
    return ProtectionStateOut(
        protection_ids=list(value.protection_ids),
        actual_order_ids=list(value.actual_order_ids),
        pair=pair_out(value.pair),
        position_side=value.position_side,
        amount=str(value.amount),
        stop_loss=None if value.stop_loss is None else str(value.stop_loss),
        take_profit=None if value.take_profit is None else str(value.take_profit),
        active=value.active,
        triggered=value.triggered,
    )


def execution_position_out(value) -> ExecutionPositionOut:
    return ExecutionPositionOut(
        pair=pair_out(value.pair),
        signed_amount=str(value.signed_amount),
        signed_notional=str(value.signed_notional),
        entry_price=None if value.entry_price is None else str(value.entry_price),
    )


def final_position_out(value) -> ExecutionFinalPositionOut:
    return ExecutionFinalPositionOut(
        position=execution_position_out(value.position),
        protected=value.protected,
        protection_ids=list(value.protection_ids),
        protections=[protection_out(item) for item in value.protections],
    )


def compensation_out(value) -> CompensationResultOut:
    return CompensationResultOut(
        attempted=value.attempted,
        succeeded=value.succeeded,
        order=None if value.order is None else order_out(value.order),
        operation=value.operation,
        safe_signed_amount=None if value.safe_signed_amount is None else str(value.safe_signed_amount),
        required_protection=(None if value.required_protection is None else protection_out(value.required_protection)),
    )


def connection_execution_out(value) -> ConnectionExecutionResultOut:
    return ConnectionExecutionResultOut(
        book_id=value.book_id,
        connection_id=value.connection_id,
        pair=pair_out(value.pair),
        target_signed_notional=str(value.target_signed_notional),
        target_signed_amount=str(value.target_signed_amount),
        status=value.status,
        orders=[order_out(item) for item in value.orders],
        protection=None if value.protection is None else protection_out(value.protection),
        compensation=compensation_out(value.compensation),
        final_position=None if value.final_position is None else final_position_out(value.final_position),
        error_operation=value.error_operation,
        requires_attention=value.requires_attention,
        trace=list(value.trace),
        execution_quote=None if value.execution_quote is None else quote_out(value.execution_quote),
        quantity_frozen=value.quantity_frozen,
    )


def _connection_ids(book) -> tuple[str, ...]:
    ordered: list[str] = []
    sources = []
    if book.portfolio_before is not None:
        sources.append(item.connection_id for item in book.portfolio_before.connections)
    if book.proposal is not None:
        sources.append(item.connection_id for item in book.proposal.risk.connection_targets)
        sources.append(item.connection_id for item in book.proposal.connection_plans)
        sources.append(iter(book.proposal.unavailable_connections))
    if book.execution is not None:
        sources.append(item.connection_id for item in book.execution.connection_results)
    if book.portfolio_after is not None:
        sources.append(item.connection_id for item in book.portfolio_after.connections)
    for source in sources:
        for connection_id in source:
            if connection_id not in ordered:
                ordered.append(connection_id)
    return tuple(ordered)


def _book_out(book) -> BookCycleOut:
    proposal = book.proposal
    before = (
        {item.connection_id: connection_portfolio_out(item) for item in book.portfolio_before.connections}
        if book.portfolio_before is not None
        else {}
    )
    after = (
        {item.connection_id: connection_portfolio_out(item) for item in book.portfolio_after.connections}
        if book.portfolio_after is not None
        else {}
    )
    risks = (
        {item.connection_id: connection_risk_out(item) for item in proposal.connection_risks}
        if proposal is not None
        else {}
    )
    plans = {item.connection_id: plan_out(item) for item in proposal.connection_plans} if proposal is not None else {}
    executions = (
        {item.connection_id: connection_execution_out(item) for item in book.execution.connection_results}
        if book.execution is not None
        else {}
    )
    unavailable = set(proposal.unavailable_connections if proposal is not None else ())
    connections = [
        CycleConnectionOut(
            connection_id=connection_id,
            portfolio_before=before.get(connection_id),
            portfolio_after=after.get(connection_id),
            risk=risks.get(connection_id),
            plan=plans.get(connection_id),
            execution=executions.get(connection_id),
            unavailable=connection_id in unavailable,
        )
        for connection_id in _connection_ids(book)
    ]
    execution = (
        BookExecutionSummaryOut(
            status=book.execution.status,
            requires_attention=book.execution.requires_attention,
            reallocated=book.execution.reallocated,
        )
        if book.execution is not None
        else None
    )
    return BookCycleOut(
        book_id=book.book_id,
        capital_scope=book.capital_scope,
        config_revision=book.config_revision,
        pair=book.pair.canonical(),
        market_type=book.pair.market_type,
        status=book.status,
        hitl=BookHitlOut(
            approval_id=book.hitl.approval_id,
            status=book.hitl.status,
            config_revision=book.hitl.config_revision,
        ),
        failure=BookFailureOut(stage=book.failure.stage) if book.failure is not None else None,
        requested_target_exposure=str(proposal.requested_target_exposure) if proposal is not None else None,
        target_exposure=str(proposal.target_exposure) if proposal is not None else None,
        risk=book_risk_out(proposal.risk) if proposal is not None else None,
        ready=proposal.ready if proposal is not None else None,
        errors=list(proposal.errors if proposal is not None else ()),
        execution=execution,
        portfolio_before=book_portfolio_out(book.portfolio_before) if book.portfolio_before is not None else None,
        portfolio_after=book_portfolio_out(book.portfolio_after) if book.portfolio_after is not None else None,
        portfolio_after_available=book.portfolio_after_available,
        reconciliation_required=book.reconciliation_required,
        connections=connections,
    )


def cycle_out(record) -> CycleOut:
    """Project one persisted canonical journal record without runtime lookups."""
    return CycleOut(
        cycle_id=record.cycle_id,
        config_revision=record.config_revision,
        market_data_source_id=record.market_data_source_id,
        shared_signals=SharedSignalsOut(
            components=[
                ComponentSignalOut(
                    component_id=item.component_id,
                    direction=item.direction,
                    confidence=item.confidence,
                    reasoning=item.reasoning,
                    details=json_entries_out(item.details),
                    blocks=list(item.blocks),
                    evaluation_reference=item.evaluation_reference,
                    status=item.status,
                    duration_ms=item.duration_ms,
                    usage=item.usage,
                    cost=item.cost,
                )
                for item in record.component_signals
            ],
            fused=(
                FusedSignalOut(
                    score=record.fused_signal.score,
                    reasoning=record.fused_signal.reasoning,
                    contributions=[
                        ComponentContributionOut(
                            component_id=item.component_id,
                            weight=item.weight,
                            signed_score=item.signed_score,
                            weighted_score=item.weighted_score,
                        )
                        for item in record.fused_signal.contributions
                    ],
                )
                if record.fused_signal is not None
                else None
            ),
            target_position=(
                TargetPositionOut(
                    side=record.target_position.side,
                    size_ratio=record.target_position.size_ratio,
                )
                if record.target_position is not None
                else None
            ),
        ),
        books=[_book_out(item) for item in record.book_results],
        cycle_status=record.cycle_status,
        execution_status=record.execution_status,
        requires_attention=record.requires_attention,
        created_at=record.created_at,
    )


for _model in (
    ComponentSignalOut,
    CycleConnectionOut,
    BookCycleOut,
    CycleOut,
):
    _model.model_rebuild()
