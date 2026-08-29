"""HITL API 返回完整资金池 proposal 与最终真实周期状态。"""

# ruff: noqa: F401, F811 - 导入 fixture 供本模块的 pytest 参数解析。

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

from cryptotrader.decision.models import CycleOutcome, TargetPosition
from cryptotrader.hitl.store import ApprovalStateError, BookApprovalStore
from cryptotrader.pair import Pair
from tests.test_multi_venue_journal import _proposal_for
from tests.test_runtime_config_api import api_harness

PAIR = Pair.parse("BTC/USDT:USDT")
NOW = datetime.now(UTC)


def _outcome(status: str, execution_status: str = "not_started", requires_attention: bool = False) -> CycleOutcome:
    return CycleOutcome("cycle-1", 9, TargetPosition("long", 1.0), (), status, execution_status, requires_attention)


class _Cycle:
    def __init__(self) -> None:
        self.approvals = BookApprovalStore()
        self.execute_approved = AsyncMock(side_effect=self._execute)
        self.reject_approval = AsyncMock(side_effect=self._reject)

    async def _execute(self, approval_id):
        await self.approvals.claim_for_execution(approval_id, current_revision=9)
        return _outcome("completed", "completed")

    async def _reject(self, approval_id):
        await self.approvals.reject(approval_id)
        return _outcome("approval_rejected")


async def _seed(cycle: _Cycle, approval_id: str = "approval-1"):
    return await cycle.approvals.create(
        _proposal_for("live", "real", ("live-first", "live-second"), PAIR),
        cycle_id="cycle-1",
        approval_id=approval_id,
        created_at=NOW,
    )


async def test_pending_api_exposes_frozen_book_proposal(api_harness):
    cycle = _Cycle()
    api_harness.runtime.cycle = cycle
    await _seed(cycle)

    response = await api_harness.client.get("/api/hitl/pending")

    assert response.status_code == 200
    item = response.json()[0]
    assert item["cycle_id"] == "cycle-1"
    assert item["book_id"] == "live"
    assert item["config_revision"] == 9
    assert item["proposal"]["book_id"] == "live"
    assert item["proposal"]["pair"] == {"symbol": "BTC/USDT:USDT"}
    assert item["proposal"]["risk"]["connection_targets"][0]["connection_id"] == "live-first"
    assert item["proposal"]["connection_plans"][0]["connection_id"] == "live-first"


async def test_approve_api_returns_final_approval_and_cycle_state(api_harness):
    cycle = _Cycle()
    api_harness.runtime.cycle = cycle
    await _seed(cycle)

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "approve"},
    )

    assert response.status_code == 200
    cycle.execute_approved.assert_awaited_once_with("approval-1")
    assert response.json() == {
        "approval_id": "approval-1",
        "cycle_id": "cycle-1",
        "approval_status": "executed",
        "cycle_status": "completed",
        "execution_status": "completed",
        "requires_attention": False,
    }


async def test_revision_invalidation_response_never_reports_executed(api_harness):
    cycle = _Cycle()
    api_harness.runtime.cycle = cycle
    await _seed(cycle)

    async def invalidate_after_approval(approval_id):
        current = await cycle.approvals.get(approval_id)
        cycle.approvals.records[0] = replace(current, status="invalidated", decided_at=datetime.now(UTC))
        return _outcome("approval_rejected")

    cycle.execute_approved.side_effect = invalidate_after_approval

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "approve"},
    )

    assert response.status_code == 200
    assert response.json()["approval_status"] == "invalidated"
    assert response.json()["cycle_status"] == "approval_rejected"


async def test_reject_api_updates_same_cycle_without_execution(api_harness):
    cycle = _Cycle()
    api_harness.runtime.cycle = cycle
    await _seed(cycle)

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "reject"},
    )

    assert response.status_code == 200
    cycle.reject_approval.assert_awaited_once_with("approval-1")
    cycle.execute_approved.assert_not_awaited()
    assert response.json()["approval_status"] == "rejected"
    assert response.json()["cycle_status"] == "approval_rejected"


async def test_already_decided_approval_returns_conflict(api_harness):
    cycle = _Cycle()
    api_harness.runtime.cycle = cycle
    await _seed(cycle)
    await cycle.approvals.reject("approval-1")

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "approve"},
    )

    assert response.status_code == 409


async def test_unknown_approval_returns_not_found(api_harness):
    api_harness.runtime.cycle = _Cycle()

    response = await api_harness.client.get("/api/hitl/missing")

    assert response.status_code == 404


async def test_approval_state_error_returns_fixed_detail_without_internal_marker(api_harness):
    cycle = _Cycle()
    api_harness.runtime.cycle = cycle
    await _seed(cycle)
    marker = "approval-1 internal-state-marker"
    cycle.execute_approved.side_effect = ApprovalStateError(marker)

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "approve"},
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Approval state conflict"}
    assert marker not in response.text
    assert "approval-1" not in response.text


async def test_lookup_error_returns_fixed_detail_without_internal_marker(api_harness):
    cycle = _Cycle()
    api_harness.runtime.cycle = cycle
    await _seed(cycle)
    marker = "approval-1 internal-lookup-marker"
    cycle.reject_approval.side_effect = LookupError(marker)

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "reject"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Approval request not found"}
    assert marker not in response.text
    assert "approval-1" not in response.text


async def test_hitl_request_rejects_unknown_fields(api_harness):
    api_harness.runtime.cycle = _Cycle()

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "approve", "unexpected": True},
    )

    assert response.status_code == 422
