"""同连接风险补偿只撤销本次新增的实际成交量。"""

from decimal import Decimal

import pytest

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
