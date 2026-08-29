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
from cryptotrader.execution.models import BookExecutionProposal, BookExecutionResult
from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import BookPortfolioSnapshot
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
                "market_data_source_id",
                "market_type",
                "equity",
                "current_price",
                "atr",
                "current_position",
                "portfolio",
            }
        elif available is False:
            required = {"pair", "as_of", "mode", "market_data_source_id"}
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
_BOOK_CYCLE_STATUSES = frozenset({"ready", "awaiting_approval", "approval_rejected", "completed", "partial", "failed"})
_HITL_STATUSES = frozenset({"not_required", "pending", "rejected", "invalidated", "executed"})
_EMPTY_BOOK_CYCLE_STATUSES = frozenset({"no_change", "component_failed", "cycle_failed", "cancelled"})
_PREPARATION_STAGES = frozenset({"portfolio", "allocation", "risk", "planning"})


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
class BookHitlSnapshot:
    """Credential-free approval identity recorded with one book outcome."""

    approval_id: str | None
    status: Literal["not_required", "pending", "rejected", "invalidated", "executed"]
    config_revision: int

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in _HITL_STATUSES:
            raise ValueError("unsupported HITL status")
        if type(self.config_revision) is not int or self.config_revision < 0:
            raise ValueError("HITL config_revision must be a non-negative integer")
        if self.status == "not_required":
            if self.approval_id is not None:
                raise ValueError("not_required HITL must not have approval_id")
        elif type(self.approval_id) is not str or not self.approval_id.strip():
            raise ValueError("HITL state requires approval_id")


@dataclass(frozen=True)
class BookPreparationFailure:
    """A safe category for a book that failed before execution was possible."""

    stage: Literal["portfolio", "allocation", "risk", "planning"]

    def __post_init__(self) -> None:
        if type(self.stage) is not str or self.stage not in _PREPARATION_STAGES:
            raise ValueError("unsupported preparation failure stage")


@dataclass(frozen=True)
class BookCycleResult:
    """One book's state closed over preparation, approval, execution, and portfolios."""

    book_id: str
    capital_scope: str
    config_revision: int
    pair: Pair
    proposal: BookExecutionProposal | None
    portfolio_before: BookPortfolioSnapshot | None
    hitl: BookHitlSnapshot
    execution: BookExecutionResult | None
    failure: BookPreparationFailure | None
    portfolio_after: BookPortfolioSnapshot | None
    portfolio_after_available: bool | None
    status: Literal["ready", "awaiting_approval", "approval_rejected", "completed", "partial", "failed"]

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_proposal()
        if self.portfolio_before is not None:
            self._validate_portfolio_identity(self.portfolio_before, "portfolio_before")
            if self.proposal is not None:
                self._validate_before_closure()
        if not isinstance(self.hitl, BookHitlSnapshot) or self.hitl.config_revision != self.config_revision:
            raise ValueError("HITL identity must match book config_revision")
        self._validate_state()

    def _validate_identity(self) -> None:
        if type(self.book_id) is not str or not self.book_id.strip():
            raise ValueError("book_id must be a non-empty string")
        if type(self.capital_scope) is not str or self.capital_scope not in {"simulated", "real"}:
            raise ValueError("capital_scope must be simulated or real")
        if type(self.config_revision) is not int or self.config_revision < 0:
            raise ValueError("config_revision must be a non-negative integer")
        if not isinstance(self.pair, Pair):
            raise ValueError("pair must be a Pair")
        if type(self.status) is not str or self.status not in _BOOK_CYCLE_STATUSES:
            raise ValueError("unsupported book cycle status")

    def _validate_proposal(self) -> None:
        if self.proposal is None:
            return
        if not isinstance(self.proposal, BookExecutionProposal):
            raise ValueError("proposal must be a BookExecutionProposal or None")
        if (
            self.proposal.book_id != self.book_id
            or self.proposal.capital_scope != self.capital_scope
            or self.proposal.config_revision != self.config_revision
            or self.proposal.pair != self.pair
        ):
            raise ValueError("proposal identity must match the book cycle")

    def _validate_portfolio_identity(self, value: object, field_name: str) -> None:
        if not isinstance(value, BookPortfolioSnapshot):
            raise ValueError(f"{field_name} must be a BookPortfolioSnapshot")
        if value.book_id != self.book_id or value.capital_scope != self.capital_scope:
            raise ValueError(f"{field_name} identity must match the book cycle")
        if any(item.position.pair != self.pair for item in value.connections):
            raise ValueError(f"{field_name} positions must match book pair")

    def _validate_before_closure(self) -> None:
        assert self.proposal is not None
        assert self.portfolio_before is not None
        expected_ids = tuple(target.connection_id for target in self.proposal.risk.connection_targets)
        actual_ids = tuple(item.connection_id for item in self.portfolio_before.connections)
        if actual_ids != expected_ids:
            raise ValueError("portfolio_before connections must match the complete risk target set")
        if any(
            target.book_equity != self.portfolio_before.total_equity for target in self.proposal.risk.connection_targets
        ):
            raise ValueError("risk target book_equity must match portfolio_before total_equity")
        before_by_id = {item.connection_id: item for item in self.portfolio_before.connections}
        if any(
            before_by_id[plan.connection_id].position.signed_amount != plan.current_signed_amount
            or before_by_id[plan.connection_id].position.signed_notional != plan.current_signed_notional
            for plan in self.proposal.connection_plans
        ):
            raise ValueError("portfolio_before current position must match proposal plan current values")

    def _validate_state(self) -> None:
        if self.failure is not None:
            self._validate_preparation_failure()
            return
        if not isinstance(self.proposal, BookExecutionProposal) or self.proposal.ready is not True:
            raise ValueError("normal book cycle state requires a ready proposal")
        if self.portfolio_before is None:
            raise ValueError("normal book cycle state requires portfolio_before")
        expected_hitl = self._expected_hitl_states()
        if self.hitl.status not in expected_hitl:
            raise ValueError(f"{self.status} requires {sorted(expected_hitl)} HITL state")
        if self.status in {"completed", "partial", "failed"}:
            self._validate_terminal_state()
        elif self.execution is not None:
            raise ValueError("pre-execution book state must not contain execution")
        elif self.portfolio_after is not None or self.portfolio_after_available is not None:
            raise ValueError("pre-execution book state must not contain portfolio_after")

    def _validate_preparation_failure(self) -> None:
        if not isinstance(self.failure, BookPreparationFailure):
            raise ValueError("failure must be a BookPreparationFailure or None")
        if self.status != "failed" or self.execution is not None:
            raise ValueError("preparation failure must be failed without execution")
        if self.hitl.status != "not_required":
            raise ValueError("preparation failure must not enter HITL")
        if self.portfolio_after is not None or self.portfolio_after_available is not None:
            raise ValueError("preparation failure must not contain portfolio_after")
        if self.failure.stage == "portfolio":
            if self.proposal is not None or self.portfolio_before is not None:
                raise ValueError("portfolio failure must precede proposal and portfolio snapshot")
            return
        if self.portfolio_before is None:
            raise ValueError("post-portfolio preparation failure requires portfolio_before")
        if self.failure.stage == "allocation" and self.proposal is not None:
            raise ValueError("allocation failure must precede proposal")
        if self.proposal is not None and self.proposal.ready is not False:
            raise ValueError("preparation failure proposal must be non-ready")
        self._validate_risk_failure_evidence()

    def _validate_risk_failure_evidence(self) -> None:
        assert self.failure is not None
        if self.failure.stage == "risk" and self.proposal is not None:
            risk = self.proposal.risk
            if risk.passed or not risk.rejected_by.strip() or not risk.reason.strip():
                raise ValueError("risk preparation failure requires rejected risk evidence")

    def _expected_hitl_states(self) -> set[str]:
        if self.status == "ready":
            return {"not_required"}
        if self.status == "awaiting_approval":
            return {"pending"}
        if self.status == "approval_rejected":
            return {"rejected", "invalidated"}
        return {"not_required", "executed"}

    def _validate_terminal_state(self) -> None:
        if not isinstance(self.execution, BookExecutionResult):
            raise ValueError("terminal book state requires execution")
        if self.execution.proposal != self.proposal:
            raise ValueError("execution must close over the same proposal")
        if self.execution.status != self.status:
            raise ValueError("book status must match execution status")
        if type(self.portfolio_after_available) is not bool:
            raise ValueError("terminal book state requires explicit portfolio_after availability")
        if self.portfolio_after_available:
            self._validate_portfolio_identity(self.portfolio_after, "portfolio_after")
            self._validate_after_closure()
        elif self.portfolio_after is not None:
            raise ValueError("unavailable portfolio_after must be None")

    def _validate_after_closure(self) -> None:
        assert self.execution is not None
        assert self.portfolio_before is not None
        assert self.portfolio_after is not None
        before_by_id = {item.connection_id: item for item in self.portfolio_before.connections}
        after_by_id = {item.connection_id: item for item in self.portfolio_after.connections}
        if tuple(after_by_id) != tuple(before_by_id):
            raise ValueError("portfolio_after connections must match portfolio_before")
        results_by_id = {item.connection_id: item for item in self.execution.connection_results}
        for connection_id, before in before_by_id.items():
            result = results_by_id.get(connection_id)
            if result is None:
                expected = before.position
            elif result.final_position is not None:
                expected = result.final_position.position
            else:
                continue
            if after_by_id[connection_id].position != expected:
                raise ValueError("portfolio_after position must match final_position or portfolio_before")

    @property
    def connection_ids(self) -> tuple[str, ...]:
        if self.portfolio_before is not None:
            return tuple(item.connection_id for item in self.portfolio_before.connections)
        if self.proposal is not None:
            return tuple(item.connection_id for item in self.proposal.risk.connection_targets)
        return ()


@dataclass(frozen=True)
class MultiVenueCycleRecord:
    """新主链的一次平台无关信号与逐资金池执行审计。"""

    cycle_id: str
    config_revision: int
    market_data_source_id: str
    component_signals: tuple[ComponentSignal, ...]
    fused_signal: FusedSignal | None
    target_position: TargetPosition | None
    book_results: tuple[BookCycleResult, ...]
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
        component_ids = tuple(item.component_id for item in self.component_signals)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component signals must have unique component IDs")
        _validate_fused_signal(self.fused_signal)
        self._validate_target_and_fusion(component_ids)

    def _validate_target_and_fusion(self, component_ids: tuple[str, ...]) -> None:
        if self.target_position is not None and not isinstance(self.target_position, TargetPosition):
            raise ValueError("target_position must be a TargetPosition or None")
        if self.target_position is not None and self.target_position.side not in {"long", "short", "flat"}:
            raise ValueError("target_position has an unsupported side")
        if self.target_position is not None and (
            type(self.target_position.size_ratio) is not float or not math.isfinite(self.target_position.size_ratio)
        ):
            raise ValueError("target_position size_ratio must be an exact finite float")
        if (self.target_position is None) != (self.fused_signal is None):
            raise ValueError("target_position and fused_signal must exist together")
        if self.fused_signal is not None:
            self._validate_fusion_math(component_ids)
            self._validate_target_direction()

    def _validate_fusion_math(self, component_ids: tuple[str, ...]) -> None:
        assert self.fused_signal is not None
        contribution_ids = tuple(item.component_id for item in self.fused_signal.contributions)
        if contribution_ids != component_ids or len(contribution_ids) != len(set(contribution_ids)):
            raise ValueError("fused contribution IDs must exactly match component IDs")
        if any(not 0.0 <= item.weight <= 1.0 for item in self.fused_signal.contributions):
            raise ValueError("fused contribution weight must be in [0, 1]")
        if not math.isclose(
            math.fsum(item.weight for item in self.fused_signal.contributions),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("fused contribution weights must sum to one")
        expected_scores = {
            signal.component_id: {
                "long": signal.confidence,
                "short": -signal.confidence,
                "neutral": 0.0,
            }[signal.direction]
            for signal in self.component_signals
        }
        if any(
            contribution.signed_score != expected_scores[contribution.component_id]
            for contribution in self.fused_signal.contributions
        ):
            raise ValueError("fused signed_score must match its component signal")
        if any(
            not math.isclose(
                contribution.weighted_score,
                contribution.weight * contribution.signed_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for contribution in self.fused_signal.contributions
        ):
            raise ValueError("weighted_score must equal weight * signed_score")
        if not math.isclose(
            self.fused_signal.score,
            sum(item.weighted_score for item in self.fused_signal.contributions),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("fused score must equal the contribution sum")

    def _validate_target_direction(self) -> None:
        assert self.fused_signal is not None
        assert self.target_position is not None
        score = self.fused_signal.score
        if self.target_position.side == "long" and score <= 0:
            raise ValueError("long target_position requires positive fused score")
        if self.target_position.side == "short" and score >= 0:
            raise ValueError("short target_position requires negative fused score")
        if score == 0.0 and self.target_position.side != "flat":
            raise ValueError("zero fused score requires flat target_position")

    def _validate_results(self) -> None:
        if type(self.book_results) is not tuple or not all(
            isinstance(item, BookCycleResult) for item in self.book_results
        ):
            raise ValueError("book_results must be a tuple of BookCycleResult")
        book_ids = tuple(item.book_id for item in self.book_results)
        if len(book_ids) != len(set(book_ids)):
            raise ValueError("book_results must have unique book IDs")
        if any(item.config_revision != self.config_revision for item in self.book_results):
            raise ValueError("book result revision must match config_revision")
        if self.book_results:
            pairs = {item.pair for item in self.book_results}
            if len(pairs) != 1:
                raise ValueError("book proposal pair must be identical")
            connection_ids = tuple(connection_id for item in self.book_results for connection_id in item.connection_ids)
            if len(connection_ids) != len(set(connection_ids)):
                raise ValueError("book proposals must have globally unique connection IDs")

    def _validate_status(self) -> None:
        if self.cycle_status not in _CYCLE_STATUSES:
            raise ValueError("unsupported cycle_status")
        if self.execution_status not in _EXECUTION_STATUSES:
            raise ValueError("unsupported execution_status")
        if type(self.requires_attention) is not bool:
            raise ValueError("requires_attention must be a boolean")
        if not self.book_results:
            self._validate_empty_book_status()
            return

        expected_cycle = self._expected_cycle_status()
        if self.cycle_status != expected_cycle:
            raise ValueError("cycle_status must be derived from book states")
        expected_execution = self._expected_execution_status()
        if self.execution_status != expected_execution:
            raise ValueError("execution_status must be derived from book states")
        expected_attention = any(
            item.execution is not None and item.execution.requires_attention for item in self.book_results
        )
        if self.requires_attention != expected_attention:
            raise ValueError("requires_attention must equal the OR of execution results")

    def _validate_empty_book_status(self) -> None:
        if self.cycle_status not in _EMPTY_BOOK_CYCLE_STATUSES:
            raise ValueError("risk_rejected and execution cycle_status require per-book evidence")
        if self.execution_status != "not_started" or self.requires_attention:
            raise ValueError("empty-book cycle must not report execution or attention")
        if self.cycle_status == "component_failed" and (
            self.fused_signal is not None or self.target_position is not None
        ):
            raise ValueError("component_failed must not contain fused signal or target")
        if self.cycle_status == "no_change" and (self.fused_signal is None or self.target_position is None):
            raise ValueError("no_change requires fused signal and target evidence")

    def _expected_execution_status(self) -> str:
        terminal = tuple(item for item in self.book_results if item.execution is not None)
        if not terminal:
            return "not_started"
        execution_statuses = tuple(item.execution.status for item in terminal)
        if all(status == "failed" for status in execution_statuses):
            return "failed"
        if len(terminal) != len(self.book_results):
            return "partial"
        if all(status == "completed" for status in execution_statuses):
            return "completed"
        return "partial"

    def _expected_cycle_status(self) -> str:
        statuses = tuple(item.status for item in self.book_results)
        for state in ("awaiting_approval", "approval_rejected", "ready"):
            if state in statuses:
                return state
        preparation_failures = tuple(item.failure for item in self.book_results if item.failure is not None)
        if all(status == "failed" for status in statuses):
            if len(preparation_failures) == len(self.book_results) and all(
                failure.stage == "risk" for failure in preparation_failures
            ):
                return "risk_rejected"
            return "failed"
        if all(status == "completed" for status in statuses):
            return "completed"
        return "partial"
