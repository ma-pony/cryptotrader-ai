"""Strict credential-safe response DTOs shared by runtime APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from cryptotrader.pair import Pair


class StrictOut(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class VenueQuoteOut(StrictOut):
    pair: PairSymbolOut
    bid: str
    ask: str
    last: str


class ConnectionTargetOut(StrictOut):
    book_id: str
    connection_id: str
    weight: str
    book_equity: str
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


class ProtectionStateOut(StrictOut):
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


def pair_out(value) -> PairSymbolOut:
    return PairSymbolOut(symbol=value.canonical())


def capabilities_out(value) -> VenueCapabilitiesOut:
    return VenueCapabilitiesOut(
        market_types=sorted(value.market_types),
        native_protection=value.native_protection,
        hedge_mode=value.hedge_mode,
        reduce_only=value.reduce_only,
        supported_order_types=sorted(value.supported_order_types),
    )


def quote_out(value) -> VenueQuoteOut:
    return VenueQuoteOut(pair=pair_out(value.pair), bid=str(value.bid), ask=str(value.ask), last=str(value.last))


def connection_target_out(value) -> ConnectionTargetOut:
    return ConnectionTargetOut(
        book_id=value.book_id,
        connection_id=value.connection_id,
        weight=str(value.weight),
        book_equity=str(value.book_equity),
        target_exposure=str(value.target_exposure),
        target_signed_notional=str(value.target_signed_notional),
    )


def book_risk_out(value) -> BookRiskDecisionOut:
    return BookRiskDecisionOut(
        passed=value.passed,
        requested_target_exposure=str(value.requested_target_exposure),
        capped_target_exposure=str(value.capped_target_exposure),
        connection_weights=[str(item) for item in value.connection_weights],
        connection_targets=[connection_target_out(item) for item in value.connection_targets],
        rejected_by=value.rejected_by,
        reason=value.reason,
        cap_source=value.cap_source,
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
    )


def protection_out(value) -> ProtectionStateOut:
    return ProtectionStateOut(
        protection_ids=list(value.protection_ids),
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
    )
