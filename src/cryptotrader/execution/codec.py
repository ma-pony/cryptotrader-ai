"""Book execution DTO 的显式、版本化 JSON codec。"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from cryptotrader.execution.models import (
    BookExecutionProposal,
    BookExecutionResult,
    CompensationResult,
    ConnectionExecutionPlan,
    ConnectionExecutionResult,
    ConnectionTarget,
    ExecutionFinalPosition,
)
from cryptotrader.pair import Pair
from cryptotrader.risk.models import BookRiskDecision, ConnectionRiskDecision
from cryptotrader.venues.models import (
    ConnectionPosition,
    NormalizedOrder,
    ProtectionState,
    VenueCapabilities,
    VenueQuote,
)

_CODEC_VERSION = 1


def _require_version(value: Any, label: str) -> None:
    if type(value) is not int or value != _CODEC_VERSION:
        raise ValueError(f"unsupported {label} codec version")


def _object(value: Any, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("invalid encoded object")
    return value


def _array(value: Any) -> list[Any]:
    if type(value) is not list:
        raise ValueError("invalid encoded array")
    return value


def _decimal(value: Any) -> Decimal:
    if type(value) is not str:
        raise ValueError("invalid encoded decimal")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError("invalid encoded decimal")
    return result


def _optional_decimal(value: Any) -> Decimal | None:
    return None if value is None else _decimal(value)


def _pair_payload(pair: Pair) -> dict[str, str]:
    return {"symbol": pair.canonical()}


def _pair_from_payload(value: Any) -> Pair:
    payload = _object(value, {"symbol"})
    if type(payload["symbol"]) is not str:
        raise ValueError("invalid encoded pair")
    return Pair.parse(payload["symbol"])


def _capabilities_payload(value: VenueCapabilities) -> dict[str, Any]:
    return {
        "market_types": sorted(value.market_types),
        "native_protection": value.native_protection,
        "hedge_mode": value.hedge_mode,
        "reduce_only": value.reduce_only,
        "supported_order_types": sorted(value.supported_order_types),
    }


def _capabilities_from_payload(value: Any) -> VenueCapabilities:
    payload = _object(
        value,
        {"market_types", "native_protection", "hedge_mode", "reduce_only", "supported_order_types"},
    )
    market_types = _array(payload["market_types"])
    order_types = _array(payload["supported_order_types"])
    return VenueCapabilities(
        frozenset(market_types),
        payload["native_protection"],
        payload["hedge_mode"],
        payload["reduce_only"],
        frozenset(order_types),
    )


def _quote_payload(value: VenueQuote) -> dict[str, Any]:
    return {
        "pair": _pair_payload(value.pair),
        "bid": str(value.bid),
        "ask": str(value.ask),
        "last": str(value.last),
    }


def _quote_from_payload(value: Any) -> VenueQuote:
    payload = _object(value, {"pair", "bid", "ask", "last"})
    return VenueQuote(
        _pair_from_payload(payload["pair"]),
        _decimal(payload["bid"]),
        _decimal(payload["ask"]),
        _decimal(payload["last"]),
    )


def _target_payload(value: ConnectionTarget) -> dict[str, Any]:
    return {
        "book_id": value.book_id,
        "connection_id": value.connection_id,
        "weight": str(value.weight),
        "book_equity": str(value.book_equity),
        "target_exposure": str(value.target_exposure),
        "target_signed_notional": str(value.target_signed_notional),
    }


def _target_from_payload(value: Any) -> ConnectionTarget:
    payload = _object(
        value,
        {
            "book_id",
            "connection_id",
            "weight",
            "book_equity",
            "target_exposure",
            "target_signed_notional",
        },
    )
    return ConnectionTarget(
        payload["book_id"],
        payload["connection_id"],
        _decimal(payload["weight"]),
        _decimal(payload["book_equity"]),
        _decimal(payload["target_exposure"]),
        _decimal(payload["target_signed_notional"]),
    )


def _book_risk_payload(value: BookRiskDecision) -> dict[str, Any]:
    return {
        "passed": value.passed,
        "requested_target_exposure": str(value.requested_target_exposure),
        "capped_target_exposure": str(value.capped_target_exposure),
        "connection_weights": [str(item) for item in value.connection_weights],
        "connection_targets": [_target_payload(item) for item in value.connection_targets],
        "rejected_by": value.rejected_by,
        "reason": value.reason,
        "cap_source": value.cap_source,
    }


def _book_risk_from_payload(value: Any) -> BookRiskDecision:
    payload = _object(
        value,
        {
            "passed",
            "requested_target_exposure",
            "capped_target_exposure",
            "connection_weights",
            "connection_targets",
            "rejected_by",
            "reason",
            "cap_source",
        },
    )
    return BookRiskDecision(
        payload["passed"],
        _decimal(payload["requested_target_exposure"]),
        _decimal(payload["capped_target_exposure"]),
        tuple(_decimal(item) for item in _array(payload["connection_weights"])),
        tuple(_target_from_payload(item) for item in _array(payload["connection_targets"])),
        payload["rejected_by"],
        payload["reason"],
        payload["cap_source"],
    )


def _connection_risk_payload(value: ConnectionRiskDecision) -> dict[str, Any]:
    return {
        "connection_id": value.connection_id,
        "passed": value.passed,
        "risk_increase": value.risk_increase,
        "reason": value.reason,
        "operation": value.operation,
    }


def _connection_risk_from_payload(value: Any) -> ConnectionRiskDecision:
    payload = _object(value, {"connection_id", "passed", "risk_increase", "reason", "operation"})
    return ConnectionRiskDecision(
        payload["connection_id"],
        payload["passed"],
        payload["risk_increase"],
        payload["reason"],
        payload["operation"],
    )


def _plan_payload(value: ConnectionExecutionPlan) -> dict[str, Any]:
    decimal_fields = (
        "current_signed_notional",
        "target_signed_notional",
        "delta_signed_notional",
        "current_signed_amount",
        "target_signed_amount",
        "delta_signed_amount",
        "post_fill_signed_amount",
        "execution_price",
        "amount",
    )
    payload: dict[str, Any] = {field_name: str(getattr(value, field_name)) for field_name in decimal_fields}
    payload.update(
        {
            "book_id": value.book_id,
            "connection_id": value.connection_id,
            "pair": _pair_payload(value.pair),
            "quote": _quote_payload(value.quote),
            "side": value.side,
            "reduce_only": value.reduce_only,
            "market_type": value.market_type,
            "stop_loss": None if value.stop_loss is None else str(value.stop_loss),
            "take_profit": None if value.take_profit is None else str(value.take_profit),
            "old_protection_ids": list(value.old_protection_ids),
            "capabilities": _capabilities_payload(value.capabilities),
        }
    )
    return payload


_PLAN_KEYS = {
    "book_id",
    "connection_id",
    "pair",
    "current_signed_notional",
    "target_signed_notional",
    "delta_signed_notional",
    "current_signed_amount",
    "target_signed_amount",
    "delta_signed_amount",
    "post_fill_signed_amount",
    "quote",
    "execution_price",
    "amount",
    "side",
    "reduce_only",
    "market_type",
    "stop_loss",
    "take_profit",
    "old_protection_ids",
    "capabilities",
}


def _plan_from_payload(value: Any) -> ConnectionExecutionPlan:
    payload = _object(value, _PLAN_KEYS)
    return ConnectionExecutionPlan(
        payload["book_id"],
        payload["connection_id"],
        _pair_from_payload(payload["pair"]),
        _decimal(payload["current_signed_notional"]),
        _decimal(payload["target_signed_notional"]),
        _decimal(payload["delta_signed_notional"]),
        _decimal(payload["current_signed_amount"]),
        _decimal(payload["target_signed_amount"]),
        _decimal(payload["delta_signed_amount"]),
        _decimal(payload["post_fill_signed_amount"]),
        _quote_from_payload(payload["quote"]),
        _decimal(payload["execution_price"]),
        _decimal(payload["amount"]),
        payload["side"],
        payload["reduce_only"],
        payload["market_type"],
        _optional_decimal(payload["stop_loss"]),
        _optional_decimal(payload["take_profit"]),
        tuple(_array(payload["old_protection_ids"])),
        _capabilities_from_payload(payload["capabilities"]),
    )


def book_execution_proposal_payload(value: BookExecutionProposal) -> dict[str, Any]:
    return {
        "version": _CODEC_VERSION,
        "book_id": value.book_id,
        "capital_scope": value.capital_scope,
        "config_revision": value.config_revision,
        "pair": _pair_payload(value.pair),
        "requested_target_exposure": str(value.requested_target_exposure),
        "target_exposure": str(value.target_exposure),
        "risk": _book_risk_payload(value.risk),
        "connection_risks": [_connection_risk_payload(item) for item in value.connection_risks],
        "connection_plans": [_plan_payload(item) for item in value.connection_plans],
        "unavailable_connections": list(value.unavailable_connections),
        "errors": list(value.errors),
        "ready": value.ready,
    }


_PROPOSAL_KEYS = {
    "version",
    "book_id",
    "capital_scope",
    "config_revision",
    "pair",
    "requested_target_exposure",
    "target_exposure",
    "risk",
    "connection_risks",
    "connection_plans",
    "unavailable_connections",
    "errors",
    "ready",
}


def book_execution_proposal_from_payload(value: Any) -> BookExecutionProposal:
    payload = _object(value, _PROPOSAL_KEYS)
    _require_version(payload["version"], "proposal")
    return BookExecutionProposal(
        payload["book_id"],
        payload["capital_scope"],
        payload["config_revision"],
        _pair_from_payload(payload["pair"]),
        _decimal(payload["requested_target_exposure"]),
        _decimal(payload["target_exposure"]),
        _book_risk_from_payload(payload["risk"]),
        tuple(_connection_risk_from_payload(item) for item in _array(payload["connection_risks"])),
        tuple(_plan_from_payload(item) for item in _array(payload["connection_plans"])),
        tuple(_array(payload["unavailable_connections"])),
        tuple(_array(payload["errors"])),
        payload["ready"],
    )


def _order_payload(value: NormalizedOrder) -> dict[str, Any]:
    return {
        "id": value.id,
        "pair": _pair_payload(value.pair),
        "side": value.side,
        "order_type": value.order_type,
        "amount": str(value.amount),
        "filled_amount": str(value.filled_amount),
        "average_price": None if value.average_price is None else str(value.average_price),
        "status": value.status,
        "reduce_only": value.reduce_only,
    }


def _order_from_payload(value: Any) -> NormalizedOrder:
    payload = _object(
        value,
        {"id", "pair", "side", "order_type", "amount", "filled_amount", "average_price", "status", "reduce_only"},
    )
    return NormalizedOrder(
        payload["id"],
        _pair_from_payload(payload["pair"]),
        payload["side"],
        payload["order_type"],
        _decimal(payload["amount"]),
        _decimal(payload["filled_amount"]),
        _optional_decimal(payload["average_price"]),
        payload["status"],
        payload["reduce_only"],
    )


def _protection_payload(value: ProtectionState) -> dict[str, Any]:
    return {
        "protection_ids": list(value.protection_ids),
        "pair": _pair_payload(value.pair),
        "position_side": value.position_side,
        "amount": str(value.amount),
        "stop_loss": None if value.stop_loss is None else str(value.stop_loss),
        "take_profit": None if value.take_profit is None else str(value.take_profit),
        "active": value.active,
        "triggered": value.triggered,
    }


def _protection_from_payload(value: Any) -> ProtectionState:
    payload = _object(
        value,
        {"protection_ids", "pair", "position_side", "amount", "stop_loss", "take_profit", "active", "triggered"},
    )
    return ProtectionState(
        tuple(_array(payload["protection_ids"])),
        _pair_from_payload(payload["pair"]),
        payload["position_side"],
        _decimal(payload["amount"]),
        _optional_decimal(payload["stop_loss"]),
        _optional_decimal(payload["take_profit"]),
        payload["active"],
        payload["triggered"],
    )


def _position_payload(value: ConnectionPosition) -> dict[str, Any]:
    return {
        "pair": _pair_payload(value.pair),
        "signed_amount": str(value.signed_amount),
        "signed_notional": str(value.signed_notional),
        "entry_price": None if value.entry_price is None else str(value.entry_price),
    }


def _position_from_payload(value: Any) -> ConnectionPosition:
    payload = _object(value, {"pair", "signed_amount", "signed_notional", "entry_price"})
    return ConnectionPosition(
        _pair_from_payload(payload["pair"]),
        _decimal(payload["signed_amount"]),
        _decimal(payload["signed_notional"]),
        _optional_decimal(payload["entry_price"]),
    )


def _final_position_payload(value: ExecutionFinalPosition) -> dict[str, Any]:
    return {
        "position": _position_payload(value.position),
        "protected": value.protected,
        "protection_ids": list(value.protection_ids),
        "protections": [_protection_payload(item) for item in value.protections],
    }


def _final_position_from_payload(value: Any) -> ExecutionFinalPosition:
    payload = _object(value, {"position", "protected", "protection_ids", "protections"})
    return ExecutionFinalPosition(
        _position_from_payload(payload["position"]),
        payload["protected"],
        tuple(_array(payload["protection_ids"])),
        tuple(_protection_from_payload(item) for item in _array(payload["protections"])),
    )


def _compensation_payload(value: CompensationResult) -> dict[str, Any]:
    return {
        "attempted": value.attempted,
        "succeeded": value.succeeded,
        "order": None if value.order is None else _order_payload(value.order),
        "operation": value.operation,
        "safe_signed_amount": None if value.safe_signed_amount is None else str(value.safe_signed_amount),
        "required_protection": (
            None if value.required_protection is None else _protection_payload(value.required_protection)
        ),
    }


def _compensation_from_payload(value: Any) -> CompensationResult:
    payload = _object(
        value,
        {"attempted", "succeeded", "order", "operation", "safe_signed_amount", "required_protection"},
    )
    return CompensationResult(
        payload["attempted"],
        payload["succeeded"],
        None if payload["order"] is None else _order_from_payload(payload["order"]),
        payload["operation"],
        _optional_decimal(payload["safe_signed_amount"]),
        (None if payload["required_protection"] is None else _protection_from_payload(payload["required_protection"])),
    )


def _connection_result_payload(value: ConnectionExecutionResult) -> dict[str, Any]:
    return {
        "book_id": value.book_id,
        "connection_id": value.connection_id,
        "pair": _pair_payload(value.pair),
        "target_signed_notional": str(value.target_signed_notional),
        "target_signed_amount": str(value.target_signed_amount),
        "status": value.status,
        "orders": [_order_payload(item) for item in value.orders],
        "protection": None if value.protection is None else _protection_payload(value.protection),
        "compensation": _compensation_payload(value.compensation),
        "final_position": None if value.final_position is None else _final_position_payload(value.final_position),
        "error_operation": value.error_operation,
        "requires_attention": value.requires_attention,
        "trace": list(value.trace),
        "execution_quote": None if value.execution_quote is None else _quote_payload(value.execution_quote),
    }


def _connection_result_from_payload(value: Any) -> ConnectionExecutionResult:
    payload = _object(
        value,
        {
            "book_id",
            "connection_id",
            "pair",
            "target_signed_notional",
            "target_signed_amount",
            "status",
            "orders",
            "protection",
            "compensation",
            "final_position",
            "error_operation",
            "requires_attention",
            "trace",
            "execution_quote",
        },
    )
    return ConnectionExecutionResult(
        payload["book_id"],
        payload["connection_id"],
        _pair_from_payload(payload["pair"]),
        _decimal(payload["target_signed_notional"]),
        _decimal(payload["target_signed_amount"]),
        payload["status"],
        tuple(_order_from_payload(item) for item in _array(payload["orders"])),
        None if payload["protection"] is None else _protection_from_payload(payload["protection"]),
        _compensation_from_payload(payload["compensation"]),
        None if payload["final_position"] is None else _final_position_from_payload(payload["final_position"]),
        payload["error_operation"],
        payload["requires_attention"],
        tuple(_array(payload["trace"])),
        None if payload["execution_quote"] is None else _quote_from_payload(payload["execution_quote"]),
    )


def book_execution_result_payload(value: BookExecutionResult) -> dict[str, Any]:
    return {
        "version": _CODEC_VERSION,
        "proposal": book_execution_proposal_payload(value.proposal),
        "connection_results": [_connection_result_payload(item) for item in value.connection_results],
        "status": value.status,
        "requires_attention": value.requires_attention,
        "reallocated": value.reallocated,
    }


def book_execution_result_from_payload(value: Any) -> BookExecutionResult:
    payload = _object(
        value,
        {"version", "proposal", "connection_results", "status", "requires_attention", "reallocated"},
    )
    _require_version(payload["version"], "result")
    return BookExecutionResult(
        book_execution_proposal_from_payload(payload["proposal"]),
        tuple(_connection_result_from_payload(item) for item in _array(payload["connection_results"])),
        payload["status"],
        payload["requires_attention"],
        payload["reallocated"],
    )
