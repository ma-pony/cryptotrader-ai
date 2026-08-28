"""资金池协调器保留计划顺序、目标权重和真实部分成功语义。"""

import asyncio
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from cryptotrader.execution.models import (
    BookExecutionProposal,
    CompensationResult,
    ConnectionExecutionResult,
    ConnectionTarget,
    ExecutionFinalPosition,
)
from cryptotrader.risk.models import BookRiskDecision, ConnectionRiskDecision
from cryptotrader.venues.models import ConnectionPosition, NormalizedOrder, ProtectionState
from cryptotrader.venues.protocol import VenueOperationError
from tests.test_execution_service import _venue_plan


def _proposal() -> BookExecutionProposal:
    plans = (
        replace(_venue_plan("0", "0.4", old_protection_ids=()), connection_id="first"),
        replace(_venue_plan("0", "0.6", old_protection_ids=()), connection_id="second"),
    )
    targets = (
        ConnectionTarget("simulation", "first", Decimal("0.4"), Decimal("100"), Decimal("1"), Decimal("40")),
        ConnectionTarget("simulation", "second", Decimal("0.6"), Decimal("100"), Decimal("1"), Decimal("60")),
    )
    risk = BookRiskDecision(True, Decimal("1"), Decimal("1"), (Decimal("0.4"), Decimal("0.6")), targets)
    return BookExecutionProposal(
        "simulation",
        "simulated",
        9,
        plans[0].pair,
        Decimal("1"),
        Decimal("1"),
        risk,
        (
            ConnectionRiskDecision("first", True, True),
            ConnectionRiskDecision("second", True, True),
        ),
        plans,
        (),
        (),
        True,
    )


class _Service:
    def __init__(
        self,
        connection_id: str,
        result: ConnectionExecutionResult,
        gate: asyncio.Event | None = None,
    ) -> None:
        self.connection_id = connection_id
        self.result = result
        self.gate = gate
        self.started = False

    async def execute(self, plan):
        self.started = True
        if self.gate is not None:
            await self.gate.wait()
        return self.result


def _result(proposal: BookExecutionProposal, index: int, status: str) -> ConnectionExecutionResult:
    plan = proposal.connection_plans[index]
    if status == "failed":
        return ConnectionExecutionResult.failed(plan, "place_order")
    protection = ProtectionState(
        (f"protection-{plan.connection_id}",),
        plan.pair,
        "long",
        plan.target_signed_amount,
        plan.stop_loss,
        plan.take_profit,
        True,
        False,
    )
    final = ExecutionFinalPosition(
        ConnectionPosition(
            plan.pair,
            plan.target_signed_amount,
            plan.target_signed_amount * plan.quote.last,
            plan.quote.last,
        ),
        True,
        protection.protection_ids,
    )
    return ConnectionExecutionResult(
        plan.book_id,
        plan.connection_id,
        plan.pair,
        plan.target_signed_notional,
        plan.target_signed_amount,
        "completed",
        (),
        protection,
        CompensationResult(False, False),
        final,
        "",
        False,
        ("pre_read", "reconcile"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (("completed", "completed"), "completed"),
        (("completed", "failed"), "partial"),
        (("failed", "failed"), "failed"),
    ],
)
async def test_coordinator_aggregates_connection_outcomes_without_reallocation(statuses, expected):
    from cryptotrader.execution.coordinator import ExecutionCoordinator

    proposal = _proposal()
    services = {
        plan.connection_id: _Service(plan.connection_id, _result(proposal, index, status))
        for index, (plan, status) in enumerate(zip(proposal.connection_plans, statuses, strict=True))
    }

    result = await ExecutionCoordinator(services).execute(proposal)

    assert result.status == expected
    assert result.reallocated is False
    assert tuple(item.connection_id for item in result.connection_results) == ("first", "second")
    assert result.target_weights == (Decimal("0.4"), Decimal("0.6"))


@pytest.mark.asyncio
async def test_coordinator_starts_planned_connections_concurrently_and_preserves_proposal_order():
    from cryptotrader.execution.coordinator import ExecutionCoordinator

    proposal = _proposal()
    gate = asyncio.Event()
    services = {
        plan.connection_id: _Service(plan.connection_id, _result(proposal, index, "completed"), gate)
        for index, plan in enumerate(proposal.connection_plans)
    }

    task = asyncio.create_task(
        ExecutionCoordinator({"second": services["second"], "first": services["first"]}).execute(proposal)
    )
    for _ in range(20):
        if all(service.started for service in services.values()):
            break
        await asyncio.sleep(0)
    assert all(service.started for service in services.values())
    gate.set()
    result = await task
    assert tuple(item.connection_id for item in result.connection_results) == ("first", "second")


@pytest.mark.asyncio
async def test_coordinator_rejects_non_ready_proposal_before_any_service_side_effect():
    from cryptotrader.execution.coordinator import ExecutionCoordinator

    proposal = _proposal()
    not_ready = replace(proposal, connection_plans=(), ready=False)
    service = _Service("first", _result(proposal, 0, "completed"))

    with pytest.raises(ValueError, match="ready"):
        await ExecutionCoordinator({"first": service}).execute(not_ready)
    assert service.started is False


@pytest.mark.asyncio
async def test_coordinator_redacts_explicit_venue_failure_without_losing_sibling_result():
    from cryptotrader.execution.coordinator import ExecutionCoordinator

    proposal = _proposal()

    class _FailingService:
        connection_id = "second"

        async def execute(self, plan):
            raise VenueOperationError("RAW_SECRET_COORDINATOR")

    services = {
        "first": _Service("first", _result(proposal, 0, "completed")),
        "second": _FailingService(),
    }
    result = await ExecutionCoordinator(services).execute(proposal)

    assert result.status == "partial"
    assert tuple(item.status for item in result.connection_results) == ("completed", "failed")
    assert result.connection_results[1].error_operation == "execute"
    assert result.requires_attention is True
    assert "RAW_SECRET" not in repr(result)


@pytest.mark.asyncio
async def test_coordinator_counts_unavailable_target_as_failed_book_outcome():
    from cryptotrader.execution.coordinator import ExecutionCoordinator

    proposal = _proposal()
    partial_proposal = replace(
        proposal,
        connection_risks=(
            proposal.connection_risks[0],
            ConnectionRiskDecision(
                "second",
                False,
                False,
                "connection unavailable",
                "list_open_state",
            ),
        ),
        connection_plans=(proposal.connection_plans[0],),
        unavailable_connections=("second",),
        errors=("connection second: list_open_state failed",),
    )
    service = _Service("first", _result(partial_proposal, 0, "completed"))

    result = await ExecutionCoordinator({"first": service}).execute(partial_proposal)

    assert result.status == "partial"
    assert tuple(item.connection_id for item in result.connection_results) == ("first",)


def test_result_models_reject_status_order_and_mutation_inconsistency():
    from cryptotrader.execution.models import BookExecutionResult

    proposal = _proposal()
    completed = _result(proposal, 0, "completed")
    failed = _result(proposal, 1, "failed")
    result = BookExecutionResult(proposal, (completed, failed), "partial", False)

    with pytest.raises(FrozenInstanceError):
        result.status = "completed"
    with pytest.raises(ValueError, match="status"):
        replace(result, status="completed")
    with pytest.raises(ValueError, match="order"):
        replace(result, connection_results=(failed, completed))
    with pytest.raises(ValueError, match="operation"):
        replace(failed, error_operation="RAW_SECRET")

    partial_order = NormalizedOrder(
        "partial",
        completed.pair,
        "buy",
        "market",
        Decimal("0.4"),
        Decimal("0.2"),
        Decimal("100"),
        "partially_filled",
        False,
    )
    with pytest.raises(ValueError, match="complete fills"):
        replace(completed, orders=(partial_order,))
