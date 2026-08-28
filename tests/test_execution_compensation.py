"""同连接风险补偿只撤销本次新增的实际成交量。"""

from dataclasses import replace
from decimal import Decimal

import pytest

from cryptotrader.venues.models import ProtectionState
from cryptotrader.venues.protocol import VenueOperationError
from tests.test_execution_service import _venue_plan, _venue_protection, _VenueSession


@pytest.mark.asyncio
async def test_open_success_protection_failure_compensates_actual_fill_on_same_connection():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("1", protections=(_venue_protection("1"),), failures=("replace_protection",))
    result = await VenueExecutionService(session).execute(_venue_plan("1", "2"))

    assert result.status == "failed"
    assert result.compensation.attempted is True
    assert result.compensation.succeeded is True
    assert result.compensation.order is not None
    assert result.compensation.order.amount == Decimal("1")
    assert result.compensation.order.reduce_only is True
    assert session.signed_amount == Decimal("1")
    assert result.requires_attention is False


@pytest.mark.asyncio
async def test_failed_compensation_marks_unprotected_residual_for_attention():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession(
        "1",
        protections=(_venue_protection("1"),),
        failures=("replace_protection", "place_order"),
    )
    result = await VenueExecutionService(session).execute(_venue_plan("1", "2"))

    assert result.status == "failed"
    assert result.compensation.attempted is True
    assert result.compensation.succeeded is False
    assert result.requires_attention is True
    assert result.final_position is not None
    assert result.final_position.protected is False
    assert "RAW_SECRET" not in repr(result)


@pytest.mark.asyncio
async def test_successful_compensation_result_is_closed_over_explicit_safe_state():
    from cryptotrader.execution.models import ExecutionFinalPosition
    from cryptotrader.execution.service import VenueExecutionService
    from cryptotrader.venues.models import ConnectionPosition

    session = _VenueSession("1", protections=(_venue_protection("1"),), failures=("replace_protection",))
    result = await VenueExecutionService(session).execute(_venue_plan("1", "2"))

    assert result.compensation.succeeded is True
    assert result.compensation.safe_signed_amount == Decimal("1")
    assert result.compensation.required_protection is not None
    wrong_protection = replace(result.compensation.required_protection, amount=Decimal("999"))
    wrong_final = ExecutionFinalPosition(
        ConnectionPosition(result.pair, Decimal("999"), Decimal("99900"), Decimal("100")),
        True,
        wrong_protection.protection_ids,
        (wrong_protection,),
    )
    with pytest.raises(ValueError, match="safe amount"):
        replace(result, final_position=wrong_final)


@pytest.mark.asyncio
async def test_partial_risk_increase_compensates_only_reported_fill():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession(
        "1",
        protections=(_venue_protection("1"),),
        partial_fill=Decimal("0.25"),
    )
    result = await VenueExecutionService(session).execute(_venue_plan("1", "2"))

    assert result.status == "failed"
    assert result.compensation.succeeded is True
    assert result.compensation.order is not None
    assert result.compensation.order.amount == Decimal("0.25")
    assert session.signed_amount == Decimal("1")
    assert result.requires_attention is False


@pytest.mark.asyncio
async def test_order_transport_error_reconciles_and_compensates_observed_new_risk():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("1", protections=(_venue_protection("1"),))
    original_place = session.place_order
    first = True

    async def fill_then_raise(intent):
        nonlocal first
        if first:
            first = False
            await original_place(intent)
            raise VenueOperationError("RAW_SECRET_AFTER_FILL")
        return await original_place(intent)

    session.place_order = fill_then_raise
    result = await VenueExecutionService(session).execute(_venue_plan("1", "2"))

    assert result.status == "failed"
    assert result.compensation.succeeded is True
    assert result.compensation.order is not None
    assert result.compensation.order.amount == Decimal("1")
    assert session.signed_amount == Decimal("1")
    assert result.requires_attention is False
    assert "RAW_SECRET" not in repr(result)


@pytest.mark.asyncio
async def test_flip_open_transport_error_after_fill_compensates_back_to_flat():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("1", protections=(_venue_protection("1"),))
    original_place = session.place_order
    calls = 0

    async def second_fill_then_raise(intent):
        nonlocal calls
        calls += 1
        if calls == 2:
            await original_place(intent)
            raise VenueOperationError("RAW_SECRET_FLIP_AFTER_FILL")
        return await original_place(intent)

    session.place_order = second_fill_then_raise
    result = await VenueExecutionService(session).execute(_venue_plan("1", "-2"))

    assert result.status == "failed"
    assert result.compensation.succeeded is True
    assert result.compensation.order is not None
    assert result.compensation.order.amount == Decimal("2")
    assert session.signed_amount == 0
    assert result.requires_attention is False


@pytest.mark.asyncio
async def test_extra_active_protection_triggers_compensation_and_exact_prior_spec_restore():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("1", protections=(_venue_protection("1"),))
    replace_calls = 0

    async def append_then_restore(spec):
        nonlocal replace_calls
        replace_calls += 1
        protection = ProtectionState(
            (f"replacement-{replace_calls}",),
            spec.pair,
            spec.position_side,
            spec.amount,
            spec.stop_loss,
            spec.take_profit,
            True,
            False,
        )
        if replace_calls == 1:
            session.protections += (protection,)
        else:
            session.protections = (protection,)
        return protection

    session.replace_protection = append_then_restore
    result = await VenueExecutionService(session).execute(_venue_plan("1", "2"))

    assert result.status == "failed"
    assert result.error_operation == "protection_mismatch"
    assert result.compensation.succeeded is True
    assert result.compensation.safe_signed_amount == Decimal("1")
    assert result.compensation.required_protection == session.protections[0]
    assert replace_calls == 2
    assert session.signed_amount == Decimal("1")
    assert len(session.protections) == 1
    assert session.protections[0].amount == Decimal("1")
    assert result.requires_attention is False


@pytest.mark.asyncio
async def test_extra_active_protection_surviving_rollback_requires_attention():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("1", protections=(_venue_protection("1"),))
    replace_calls = 0

    async def always_append(spec):
        nonlocal replace_calls
        replace_calls += 1
        protection = ProtectionState(
            (f"ambiguous-{replace_calls}",),
            spec.pair,
            spec.position_side,
            spec.amount,
            spec.stop_loss,
            spec.take_profit,
            True,
            False,
        )
        session.protections += (protection,)
        return protection

    session.replace_protection = always_append
    result = await VenueExecutionService(session).execute(_venue_plan("1", "2"))

    assert result.status == "failed"
    assert result.compensation.attempted is True
    assert result.compensation.succeeded is False
    assert result.requires_attention is True
    assert replace_calls == 2
    assert len(session.protections) > 1
