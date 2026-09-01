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
from tests.test_execution_service import (
    SPOT_CAPABILITIES,
    SPOT_PAIR,
    _venue_plan,
    _VenueSession,
)


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


async def test_coordinator_preserves_actual_spot_frozen_receipts_after_price_change():
    from cryptotrader.execution.coordinator import ExecutionCoordinator
    from cryptotrader.execution.service import VenueExecutionService
    from cryptotrader.venues.models import VenueQuote

    original = _proposal()
    proposal = replace(
        original,
        pair=SPOT_PAIR,
        connection_plans=tuple(
            replace(
                plan,
                pair=SPOT_PAIR,
                market_type="spot",
                capabilities=SPOT_CAPABILITIES,
                quote=replace(plan.quote, pair=SPOT_PAIR),
                stop_loss=None,
                take_profit=None,
            )
            for plan in original.connection_plans
        ),
    )
    services = {}
    for plan in proposal.connection_plans:
        session = _VenueSession("0", quote=VenueQuote(SPOT_PAIR, Decimal("99"), Decimal("99"), Decimal("99")))
        session.connection_id = plan.connection_id
        session.connection = replace(session.connection, id=plan.connection_id)
        session.capabilities = SPOT_CAPABILITIES
        services[plan.connection_id] = VenueExecutionService(session, connection=session.connection)
    result = await ExecutionCoordinator(services).execute(proposal, frozen=True)
    assert result.status == "completed"
    assert [item.orders[0].amount for item in result.connection_results] == [Decimal("0.4"), Decimal("0.6")]
    assert all(item.quantity_frozen for item in result.connection_results)
    assert [item.final_position.position.signed_amount for item in result.connection_results] == [
        Decimal("0.4"),
        Decimal("0.6"),
    ]


def _result(proposal: BookExecutionProposal, index: int, status: str) -> ConnectionExecutionResult:
    plan = proposal.connection_plans[index]
    if status == "failed":
        return ConnectionExecutionResult.failed(plan, "place_order", execution_quote=plan.quote)
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
        (protection,),
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
        plan.quote,
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
async def test_coordinator_executes_real_spot_service_without_protection_path():
    from cryptotrader.execution.coordinator import ExecutionCoordinator
    from cryptotrader.execution.service import VenueExecutionService
    from cryptotrader.venues.models import VenueQuote

    quote = VenueQuote(SPOT_PAIR, Decimal("99"), Decimal("100"), Decimal("99.5"))
    plan = replace(
        _venue_plan("0", "1", old_protection_ids=()),
        pair=SPOT_PAIR,
        quote=quote,
        execution_price=Decimal("100"),
        market_type="spot",
        stop_loss=None,
        take_profit=None,
        capabilities=SPOT_CAPABILITIES,
    )
    target = ConnectionTarget("simulation", "paper-a", Decimal("1"), Decimal("100"), Decimal("1"), Decimal("100"))
    risk = BookRiskDecision(True, Decimal("1"), Decimal("1"), (Decimal("1"),), (target,))
    proposal = BookExecutionProposal(
        "simulation",
        "simulated",
        9,
        SPOT_PAIR,
        Decimal("1"),
        Decimal("1"),
        risk,
        (ConnectionRiskDecision("paper-a", True, True),),
        (plan,),
        (),
        (),
        True,
    )
    session = _VenueSession("0", quote=quote)
    session.capabilities = SPOT_CAPABILITIES

    result = await ExecutionCoordinator(
        {"paper-a": VenueExecutionService(session, connection=session.connection)}
    ).execute(proposal)

    assert result.status == "completed"
    assert result.connection_results[0].protection is None
    assert "replace_protection" not in session.calls


@pytest.mark.asyncio
async def test_coordinator_audits_spot_protection_precondition_without_losing_sibling():
    from cryptotrader.execution.coordinator import ExecutionCoordinator
    from cryptotrader.execution.service import VenueExecutionService
    from cryptotrader.venues.models import VenueQuote

    quote = VenueQuote(SPOT_PAIR, Decimal("99"), Decimal("100"), Decimal("99.5"))
    base = _proposal()
    plans = tuple(
        replace(
            plan,
            pair=SPOT_PAIR,
            quote=quote,
            execution_price=Decimal("100"),
            market_type="spot",
            stop_loss=None,
            take_profit=None,
            capabilities=SPOT_CAPABILITIES,
        )
        for plan in base.connection_plans
    )
    proposal = replace(base, pair=SPOT_PAIR, connection_plans=plans)
    external_oco = ProtectionState(
        ("external-oco",),
        SPOT_PAIR,
        "long",
        Decimal("1"),
        Decimal("90"),
        Decimal("120"),
        True,
        False,
    )
    protected = _VenueSession("0", protections=(external_oco,), quote=quote)
    protected.connection_id = "first"
    protected.connection = replace(protected.connection, id="first")
    protected.capabilities = SPOT_CAPABILITIES
    sibling = _VenueSession("0", quote=quote)
    sibling.connection_id = "second"
    sibling.connection = replace(sibling.connection, id="second")
    sibling.capabilities = SPOT_CAPABILITIES

    result = await ExecutionCoordinator(
        {
            "first": VenueExecutionService(protected, connection=protected.connection),
            "second": VenueExecutionService(sibling, connection=sibling.connection),
        }
    ).execute(proposal)

    assert result.status == "partial"
    assert tuple(item.status for item in result.connection_results) == ("failed", "completed")
    assert result.connection_results[0].error_operation == "precondition"
    assert result.connection_results[0].requires_attention is True
    assert protected.calls == ["list_open_state"]
    assert sibling.signed_amount == Decimal("0.6")


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
@pytest.mark.parametrize("bad_outcome", [RuntimeError("RAW_SECRET_UNEXPECTED"), object()])
async def test_coordinator_preserves_success_when_sibling_raises_or_returns_invalid_result(bad_outcome):
    from cryptotrader.execution.coordinator import ExecutionCoordinator

    proposal = _proposal()

    class _UnexpectedService:
        connection_id = "second"

        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, plan):
            self.calls += 1
            if isinstance(bad_outcome, BaseException):
                raise bad_outcome
            return bad_outcome

    successful = _Service("first", _result(proposal, 0, "completed"))
    unexpected = _UnexpectedService()

    result = await ExecutionCoordinator({"first": successful, "second": unexpected}).execute(proposal)

    assert result.status == "partial"
    assert tuple(item.status for item in result.connection_results) == ("completed", "failed")
    assert result.connection_results[0].final_position is not None
    assert result.connection_results[1].final_position is None
    assert result.connection_results[1].requires_attention is True
    assert unexpected.calls == 1
    assert "RAW_SECRET" not in repr(result)


@pytest.mark.asyncio
async def test_coordinator_does_not_convert_non_exception_base_exception_to_connection_failure():
    from cryptotrader.execution.coordinator import ExecutionCoordinator

    proposal = _proposal()

    class _Fatal(BaseException):
        pass

    class _FatalService:
        connection_id = "second"

        async def execute(self, plan):
            raise _Fatal

    with pytest.raises(_Fatal):
        await ExecutionCoordinator(
            {
                "first": _Service("first", _result(proposal, 0, "completed")),
                "second": _FatalService(),
            }
        ).execute(proposal)


@pytest.mark.asyncio
async def test_coordinator_converts_semantically_mismatched_result_for_only_that_connection():
    from cryptotrader.execution.coordinator import ExecutionCoordinator

    proposal = _proposal()
    wrong_connection = _result(proposal, 0, "completed")

    result = await ExecutionCoordinator(
        {
            "first": _Service("first", _result(proposal, 0, "completed")),
            "second": _Service("second", wrong_connection),
        }
    ).execute(proposal)

    assert result.status == "partial"
    assert tuple(item.status for item in result.connection_results) == ("completed", "failed")
    assert result.connection_results[1].connection_id == "second"
    assert result.connection_results[1].final_position is None
    assert result.connection_results[1].requires_attention is True


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


@pytest.mark.asyncio
async def test_coordinator_counts_risk_rejected_target_without_plan_as_failed():
    from cryptotrader.execution.coordinator import ExecutionCoordinator

    proposal = _proposal()
    partial_proposal = replace(
        proposal,
        connection_risks=(
            proposal.connection_risks[0],
            ConnectionRiskDecision("second", False, False, "risk rejected", "risk_gate"),
        ),
        connection_plans=(proposal.connection_plans[0],),
        errors=("connection second: risk_gate failed",),
    )
    service = _Service("first", _result(partial_proposal, 0, "completed"))

    result = await ExecutionCoordinator({"first": service}).execute(partial_proposal)

    assert result.status == "partial"


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


def test_completed_result_requires_latest_execution_quote():
    proposal = _proposal()
    completed = _result(proposal, 0, "completed")

    quoted = replace(completed, execution_quote=proposal.connection_plans[0].quote)

    assert quoted.execution_quote == proposal.connection_plans[0].quote
    with pytest.raises(ValueError, match="quote"):
        replace(quoted, execution_quote=None)


@pytest.mark.parametrize(("active", "triggered"), [(False, False), (True, True)])
def test_completed_result_rejects_inactive_or_triggered_protection(active, triggered):
    proposal = _proposal()
    completed = _result(proposal, 0, "completed")
    invalid = replace(completed.protection, active=active, triggered=triggered)

    with pytest.raises(ValueError, match="protection"):
        replace(completed, protection=invalid)


def test_completed_result_rejects_target_amount_outside_quote_implied_band():
    proposal = _proposal()
    completed = _result(proposal, 0, "completed")
    invalid_protection = replace(completed.protection, amount=Decimal("999"))
    invalid_position = ExecutionFinalPosition(
        ConnectionPosition(completed.pair, Decimal("999"), Decimal("99900"), Decimal("100")),
        True,
        invalid_protection.protection_ids,
        (invalid_protection,),
    )

    with pytest.raises(ValueError, match="notional"):
        replace(
            completed,
            target_signed_amount=Decimal("999"),
            protection=invalid_protection,
            final_position=invalid_position,
        )


def test_completed_result_requires_exact_final_protection_ids():
    proposal = _proposal()
    completed = _result(proposal, 0, "completed")

    with pytest.raises(ValueError, match="exact"):
        replace(
            completed.final_position,
            protection_ids=(*completed.final_position.protection_ids, "ambiguous-extra"),
        )


def test_completed_derivative_result_requires_returned_replacement_group():
    proposal = _proposal()
    completed = _result(proposal, 0, "completed")

    with pytest.raises(ValueError, match="replacement protection"):
        replace(completed, protection=None)


def test_final_position_protected_flag_requires_matching_active_group():
    proposal = _proposal()
    completed = _result(proposal, 0, "completed")
    wrong_side = replace(completed.protection, position_side="short")

    with pytest.raises(ValueError, match="protected"):
        replace(
            completed.final_position,
            protection_ids=wrong_side.protection_ids,
            protections=(wrong_side,),
        )
