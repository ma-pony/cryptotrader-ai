"""HITL API 通过 TradingCycle 原子恢复或拒绝冻结计划。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from cryptotrader.decision.models import CycleOutcome, TargetPosition
from cryptotrader.hitl.store import ApprovalStateError, ApprovalStore
from tests.factories.signal_fusion import context, request, trade_plan


class _Cycle:
    def __init__(self, mode="paper", approvals=None) -> None:
        self.mode = mode
        self.approvals = approvals or ApprovalStore()
        self.resume_approved = AsyncMock()
        self.reject_approval = AsyncMock()


def _request_for(cycle, *, cycles=None):
    state = SimpleNamespace(
        trading_cycle=cycle,
        trading_cycles=cycles or {cycle.mode: cycle},
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


async def _seed(cycle, approval_id="approval-1", mode="paper"):
    return await cycle.approvals.create(
        cycle_id="cycle-1",
        cycle_request=request(mode=mode),
        profile_revision=3,
        signal_context=context(),
        plan=trade_plan(TargetPosition("long", 0.4), stop_loss=90.0, take_profit=120.0),
        approval_id=approval_id,
    )


@pytest.mark.asyncio
async def test_pending_api_exposes_frozen_target_plan():
    from api.routes.hitl import list_pending

    cycle = _Cycle()
    await _seed(cycle)

    result = await list_pending(_request_for(cycle))

    assert len(result) == 1
    assert result[0].cycle_id == "cycle-1"
    assert result[0].profile_revision == 3
    assert result[0].trade_plan["target"] == {"side": "long", "size_ratio": 0.4}


@pytest.mark.asyncio
async def test_approve_api_resumes_cycle_instead_of_only_flipping_database_status():
    from api.routes.hitl import HitlRespondIn, respond_approval

    cycle = _Cycle()
    await _seed(cycle)
    cycle.resume_approved.return_value = CycleOutcome("cycle-1", "completed", 3)

    result = await respond_approval(
        "approval-1",
        HitlRespondIn(decision="approve"),
        _request_for(cycle),
    )

    cycle.resume_approved.assert_awaited_once_with("approval-1", decision_by="web")
    assert result.cycle_status == "completed"


@pytest.mark.asyncio
async def test_approve_api_resumes_with_the_cycle_matching_the_frozen_request_mode():
    from api.routes.hitl import HitlRespondIn, respond_approval

    approvals = ApprovalStore()
    live_cycle = _Cycle("live", approvals)
    paper_cycle = _Cycle("paper", approvals)
    await _seed(live_cycle, mode="paper")
    paper_cycle.resume_approved.return_value = CycleOutcome("cycle-1", "completed", 3)

    result = await respond_approval(
        "approval-1",
        HitlRespondIn(decision="approve"),
        _request_for(live_cycle, cycles={"live": live_cycle, "paper": paper_cycle}),
    )

    paper_cycle.resume_approved.assert_awaited_once_with("approval-1", decision_by="web")
    live_cycle.resume_approved.assert_not_awaited()
    assert result.cycle_status == "completed"


@pytest.mark.asyncio
async def test_reject_api_terminates_cycle_without_resume():
    from api.routes.hitl import HitlRespondIn, respond_approval

    cycle = _Cycle()
    await _seed(cycle)
    cycle.reject_approval.return_value = CycleOutcome("cycle-1", "approval_rejected", 3)

    result = await respond_approval(
        "approval-1",
        HitlRespondIn(decision="reject"),
        _request_for(cycle),
    )

    cycle.reject_approval.assert_awaited_once_with("approval-1", decision_by="web")
    cycle.resume_approved.assert_not_awaited()
    assert result.cycle_status == "approval_rejected"


@pytest.mark.asyncio
async def test_already_decided_approval_returns_conflict():
    from api.routes.hitl import HitlRespondIn, respond_approval

    cycle = _Cycle()
    await _seed(cycle)
    cycle.resume_approved.side_effect = ApprovalStateError("approval is not pending")

    with pytest.raises(HTTPException) as error:
        await respond_approval(
            "approval-1",
            HitlRespondIn(decision="approve"),
            _request_for(cycle),
        )

    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_unknown_approval_returns_not_found():
    from api.routes.hitl import get_approval

    cycle = _Cycle()

    with pytest.raises(HTTPException) as error:
        await get_approval("missing", _request_for(cycle))

    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_database_store_round_trips_typed_snapshot_and_claims_once(tmp_path):
    store = ApprovalStore(f"sqlite+aiosqlite:///{tmp_path / 'approvals.db'}")
    original = await store.create(
        cycle_id="cycle-1",
        cycle_request=request(),
        profile_revision=3,
        signal_context=context(),
        plan=trade_plan(TargetPosition("short", 0.4), stop_loss=110.0, take_profit=80.0),
        approval_id="approval-1",
    )

    loaded = await store.get("approval-1")
    approved = await store.approve("approval-1", decision_by="web")

    assert loaded == original
    assert approved.status == "approved"
    assert approved.plan.target.side == "short"
    with pytest.raises(ApprovalStateError, match="not pending"):
        await store.approve("approval-1", decision_by="web")
