"""HITL API exposes and executes revision-bound book proposals."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from cryptotrader.decision.models import CycleOutcome, TargetPosition
from cryptotrader.hitl.store import ApprovalStateError, BookApprovalStore
from cryptotrader.pair import Pair
from tests.test_multi_venue_journal import _proposal_for

PAIR = Pair.parse("BTC/USDT:USDT")
NOW = datetime.now(UTC)


class _Cycle:
    def __init__(self, approvals=None) -> None:
        self.approvals = approvals or BookApprovalStore()
        self.execute_approved = AsyncMock(side_effect=self._execute)
        self.reject_approval = AsyncMock(side_effect=self._reject)

    async def _execute(self, approval_id):
        await self.approvals.claim_for_execution(approval_id, current_revision=9)
        return _outcome("completed")

    async def _reject(self, approval_id):
        await self.approvals.reject(approval_id)
        return _outcome("approval_rejected")


def _outcome(status: str) -> CycleOutcome:
    return CycleOutcome("cycle-1", 9, TargetPosition("long", 1.0), (), status, "not_started", False)


def _request_for(cycle):
    runtime = SimpleNamespace(cycle=cycle)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime=runtime)))


async def _seed(cycle, approval_id="approval-1"):
    proposal = _proposal_for("live", "real", ("live-first", "live-second"), PAIR)
    return await cycle.approvals.create(
        proposal,
        cycle_id="cycle-1",
        approval_id=approval_id,
        created_at=NOW,
    )


@pytest.mark.asyncio
async def test_pending_api_exposes_frozen_book_proposal():
    from api.routes.hitl import list_pending

    cycle = _Cycle()
    await _seed(cycle)

    result = await list_pending(_request_for(cycle))

    assert len(result) == 1
    assert result[0].cycle_id == "cycle-1"
    assert result[0].book_id == "live"
    assert result[0].config_revision == 9
    assert result[0].proposal["book_id"] == "live"


@pytest.mark.asyncio
async def test_approve_api_executes_original_proposal_on_the_unique_runtime_cycle():
    from api.routes.hitl import HitlRespondIn, respond_approval

    cycle = _Cycle()
    await _seed(cycle)

    result = await respond_approval("approval-1", HitlRespondIn(decision="approve"), _request_for(cycle))

    cycle.execute_approved.assert_awaited_once_with("approval-1")
    assert result.status == "executed"
    assert result.cycle_status == "completed"


@pytest.mark.asyncio
async def test_revision_invalidation_response_never_reports_executed():
    from api.routes.hitl import HitlRespondIn, respond_approval

    cycle = _Cycle()
    await _seed(cycle)

    async def invalidate_after_approval(approval_id):
        current = await cycle.approvals.get(approval_id)
        cycle.approvals.records[0] = replace(current, status="invalidated")
        return _outcome("approval_rejected")

    cycle.execute_approved.side_effect = invalidate_after_approval

    result = await respond_approval("approval-1", HitlRespondIn(decision="approve"), _request_for(cycle))

    assert result.status == "invalidated"
    assert result.cycle_status == "approval_rejected"


@pytest.mark.asyncio
async def test_reject_api_updates_same_cycle_without_execution():
    from api.routes.hitl import HitlRespondIn, respond_approval

    cycle = _Cycle()
    await _seed(cycle)

    result = await respond_approval("approval-1", HitlRespondIn(decision="reject"), _request_for(cycle))

    cycle.reject_approval.assert_awaited_once_with("approval-1")
    cycle.execute_approved.assert_not_awaited()
    assert result.cycle_status == "approval_rejected"


@pytest.mark.asyncio
async def test_already_decided_approval_returns_conflict():
    from api.routes.hitl import HitlRespondIn, respond_approval

    cycle = _Cycle()
    await _seed(cycle)
    await cycle.approvals.reject("approval-1")

    with pytest.raises(HTTPException) as error:
        await respond_approval("approval-1", HitlRespondIn(decision="approve"), _request_for(cycle))

    assert error.value.status_code == 409


@pytest.mark.asyncio
async def test_unknown_approval_returns_not_found():
    from api.routes.hitl import get_approval

    with pytest.raises(HTTPException) as error:
        await get_approval("missing", _request_for(_Cycle()))

    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_database_store_round_trips_proposal_and_claims_once(tmp_path):
    store = BookApprovalStore(f"sqlite+aiosqlite:///{tmp_path / 'book-approvals.db'}")
    proposal = _proposal_for("live", "real", ("live-first", "live-second"), PAIR)
    original = await store.create(proposal, cycle_id="cycle-1", approval_id="approval-1", created_at=NOW)

    loaded = await store.get("approval-1")
    await store.approve("approval-1")
    claimed = await store.claim_for_execution("approval-1", current_revision=proposal.config_revision)

    assert loaded == original
    assert claimed == proposal
    with pytest.raises(ApprovalStateError, match="already claimed"):
        await store.claim_for_execution("approval-1", current_revision=proposal.config_revision)
