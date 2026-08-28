"""一次 TradingCycle 的不可变审计记录。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from cryptotrader.decision.models import TargetPosition
from cryptotrader.execution.models import BookExecutionResult
from cryptotrader.pair import Pair
from cryptotrader.signals.fusion import ComponentContribution, FusedSignal
from cryptotrader.signals.models import ComponentSignal

if TYPE_CHECKING:
    from cryptotrader.decision.models import CycleStatus


@dataclass(frozen=True)
class TradingCycleRecord:
    cycle_id: str
    created_at: datetime
    pair: str
    status: CycleStatus
    profile_revision: int
    profile_snapshot: Mapping[str, Any]
    context_summary: Mapping[str, Any]
    component_signals: tuple[Mapping[str, Any], ...]
    component_error: Mapping[str, str] | None
    fused_signal: Mapping[str, Any] | None
    target_position: Mapping[str, Any] | None
    trade_plan: Mapping[str, Any] | None
    hitl_result: Mapping[str, Any] | None
    risk_result: Mapping[str, Any] | None
    execution_result: Mapping[str, Any] | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.cycle_id:
            raise ValueError("cycle_id is required")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if not self.pair:
            raise ValueError("pair is required")
        if self.profile_revision < 1:
            raise ValueError("profile_revision must be positive")
        if self.profile_snapshot.get("revision") != self.profile_revision:
            raise ValueError("profile_snapshot revision must match profile_revision")
        available = self.context_summary.get("available")
        if available is True:
            required = {
                "pair",
                "as_of",
                "mode",
                "exchange_id",
                "market_type",
                "equity",
                "current_price",
                "atr",
                "current_position",
                "portfolio",
            }
        elif available is False:
            required = {"pair", "as_of", "mode", "exchange_id"}
        else:
            raise ValueError("context_summary must declare availability")
        missing = required - self.context_summary.keys()
        if missing:
            raise ValueError(f"context_summary missing fields: {sorted(missing)}")


MultiVenueCycleStatus = Literal[
    "ready",
    "awaiting_approval",
    "approval_rejected",
    "completed",
    "partial",
    "failed",
    "no_change",
    "component_failed",
    "cycle_failed",
    "risk_rejected",
    "cancelled",
]
MultiVenueExecutionStatus = Literal["not_started", "completed", "partial", "failed"]
_CYCLE_STATUSES = frozenset(
    {
        "ready",
        "awaiting_approval",
        "approval_rejected",
        "completed",
        "partial",
        "failed",
        "no_change",
        "component_failed",
        "cycle_failed",
        "risk_rejected",
        "cancelled",
    }
)
_EXECUTION_STATUSES = frozenset({"not_started", "completed", "partial", "failed"})


def _freeze_detail(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError("component signal detail keys must be non-empty strings")
            frozen[key] = _freeze_detail(item)
        return MappingProxyType(frozen)
    if type(value) in {list, tuple}:
        return tuple(_freeze_detail(item) for item in value)
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Decimal) and value.is_finite():
        return value
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value
    if isinstance(value, Pair):
        return value
    raise ValueError("component signal details must be explicit JSON-safe domain values")


def _strict_signal(signal: ComponentSignal) -> ComponentSignal:
    if not isinstance(signal, ComponentSignal):
        raise ValueError("component_signals must contain ComponentSignal values")
    if type(signal.confidence) is not float or not math.isfinite(signal.confidence):
        raise ValueError("component signal confidence must be a finite float")
    if type(signal.reasoning) is not str:
        raise ValueError("component signal reasoning must be a string")
    return ComponentSignal(
        signal.component_id,
        signal.direction,
        signal.confidence,
        signal.reasoning,
        _freeze_detail(signal.details),
    )


def _validate_fused_signal(value: FusedSignal | None) -> None:
    if value is None:
        return
    if not isinstance(value, FusedSignal) or type(value.score) is not float or not math.isfinite(value.score):
        raise ValueError("fused_signal must be a finite FusedSignal")
    if type(value.reasoning) is not str:
        raise ValueError("fused signal reasoning must be a string")
    if type(value.contributions) is not tuple or not all(
        isinstance(item, ComponentContribution) for item in value.contributions
    ):
        raise ValueError("fused signal contributions must be a tuple")
    for contribution in value.contributions:
        if type(contribution.component_id) is not str or not contribution.component_id.strip():
            raise ValueError("fused signal contribution requires component_id")
        if any(
            type(getattr(contribution, field_name)) is not float or not math.isfinite(getattr(contribution, field_name))
            for field_name in ("weight", "signed_score", "weighted_score")
        ):
            raise ValueError("fused signal contribution values must be finite floats")


@dataclass(frozen=True)
class MultiVenueCycleRecord:
    """新主链的一次平台无关信号与逐资金池执行审计。"""

    cycle_id: str
    config_revision: int
    market_data_source_id: str
    component_signals: tuple[ComponentSignal, ...]
    fused_signal: FusedSignal | None
    target_position: TargetPosition | None
    book_results: tuple[BookExecutionResult, ...]
    cycle_status: MultiVenueCycleStatus
    execution_status: MultiVenueExecutionStatus
    requires_attention: bool
    created_at: datetime

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_signals()
        self._validate_results()
        self._validate_status()

    def _validate_identity(self) -> None:
        if type(self.cycle_id) is not str or not self.cycle_id.strip():
            raise ValueError("cycle_id must be a non-empty string")
        if type(self.config_revision) is not int or self.config_revision < 0:
            raise ValueError("config_revision must be a non-negative integer")
        if type(self.market_data_source_id) is not str or not self.market_data_source_id.strip():
            raise ValueError("market_data_source_id must be a non-empty string")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != timedelta(0):
            raise ValueError("created_at must be UTC-aware")

    def _validate_signals(self) -> None:
        if type(self.component_signals) is not tuple:
            raise ValueError("component_signals must be a tuple")
        object.__setattr__(self, "component_signals", tuple(_strict_signal(item) for item in self.component_signals))
        _validate_fused_signal(self.fused_signal)
        if self.target_position is not None and not isinstance(self.target_position, TargetPosition):
            raise ValueError("target_position must be a TargetPosition or None")
        if self.target_position is not None and self.target_position.side not in {"long", "short", "flat"}:
            raise ValueError("target_position has an unsupported side")
        if self.target_position is not None and self.fused_signal is None:
            raise ValueError("target_position requires fused_signal")

    def _validate_results(self) -> None:
        if type(self.book_results) is not tuple or not all(
            isinstance(item, BookExecutionResult) for item in self.book_results
        ):
            raise ValueError("book_results must be a tuple of BookExecutionResult")
        book_ids = tuple(item.book_id for item in self.book_results)
        if len(book_ids) != len(set(book_ids)):
            raise ValueError("book_results must have unique book IDs")
        if any(item.proposal.config_revision != self.config_revision for item in self.book_results):
            raise ValueError("book result revision must match config_revision")

    def _validate_status(self) -> None:
        if self.cycle_status not in _CYCLE_STATUSES:
            raise ValueError("unsupported cycle_status")
        if self.execution_status not in _EXECUTION_STATUSES:
            raise ValueError("unsupported execution_status")
        if type(self.requires_attention) is not bool or self.requires_attention != any(
            item.requires_attention for item in self.book_results
        ):
            raise ValueError("requires_attention must equal the OR of book results")
