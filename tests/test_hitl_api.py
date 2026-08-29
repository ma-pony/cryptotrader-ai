"""HITL API 返回完整资金池 proposal 与最终真实周期状态。"""

# ruff: noqa: F401, F811 - 导入 fixture 供本模块的 pytest 参数解析。

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from cryptotrader.cycle_events import MultiplexedCycleEventSink, NullCycleEventSink
from cryptotrader.decision.models import CycleOutcome, TargetPosition
from cryptotrader.hitl.store import ApprovalStateError, BookApprovalStore
from cryptotrader.pair import Pair
from cryptotrader.runtime import Runtime
from tests.runtime_lease import static_cycle_lease
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


def _mount_cycle(api_harness, cycle: _Cycle) -> None:
    api_harness.runtime.cycle = cycle
    api_harness.runtime.cycle_lease = static_cycle_lease(cycle)


async def test_pending_api_exposes_frozen_book_proposal(api_harness):
    cycle = _Cycle()
    _mount_cycle(api_harness, cycle)
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
    _mount_cycle(api_harness, cycle)
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


async def test_cancelled_approve_response_waits_for_owned_execution_terminal(api_harness):
    from api.routes.hitl import HitlRespondIn, respond_approval

    cycle = _Cycle()
    _mount_cycle(api_harness, cycle)
    await _seed(cycle)
    execution_started = asyncio.Event()
    release_execution = asyncio.Event()
    execution_terminal = asyncio.Event()
    child_cancelled = False

    async def blocking_execute(approval_id):
        nonlocal child_cancelled
        await cycle.approvals.claim_for_execution(approval_id, current_revision=9)
        execution_started.set()
        try:
            await release_execution.wait()
        except asyncio.CancelledError:
            child_cancelled = True
            raise
        execution_terminal.set()
        return _outcome("completed", "completed")

    cycle.execute_approved.side_effect = blocking_execute
    request = type(
        "Request",
        (),
        {"app": type("App", (), {"state": type("State", (), {"runtime": api_harness.runtime})()})()},
    )()
    responding = asyncio.create_task(respond_approval("approval-1", HitlRespondIn(decision="approve"), request))
    await execution_started.wait()

    responding.cancel()
    await asyncio.sleep(0)
    responding.cancel()
    assert responding.done() is False
    assert child_cancelled is False

    release_execution.set()
    with pytest.raises(asyncio.CancelledError):
        await responding
    approval = await cycle.approvals.get("approval-1")
    assert execution_terminal.is_set()
    assert child_cancelled is False
    assert approval.status == "executed"


async def test_approved_response_resumes_after_ordinary_execution_failure(api_harness):
    from api.routes.hitl import HitlRespondIn, respond_approval

    cycle = _Cycle()
    _mount_cycle(api_harness, cycle)
    await _seed(cycle)
    request = type(
        "Request",
        (),
        {"app": type("App", (), {"state": type("State", (), {"runtime": api_harness.runtime})()})()},
    )()
    cycle.execute_approved.side_effect = RuntimeError("execution failed before claim")

    with pytest.raises(RuntimeError, match="execution failed before claim"):
        await respond_approval("approval-1", HitlRespondIn(decision="approve"), request)
    approved = await cycle.approvals.get("approval-1")
    assert approved.status == "approved"

    cycle.execute_approved.side_effect = cycle._execute
    response = await respond_approval("approval-1", HitlRespondIn(decision="approve"), request)

    assert response.approval_status == "executed"
    assert response.execution_status == "completed"
    assert cycle.execute_approved.await_count == 2


async def test_revision_invalidation_response_never_reports_executed(api_harness):
    cycle = _Cycle()
    _mount_cycle(api_harness, cycle)
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
    _mount_cycle(api_harness, cycle)
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
    _mount_cycle(api_harness, cycle)
    await _seed(cycle)
    await cycle.approvals.reject("approval-1")

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "approve"},
    )

    assert response.status_code == 409


@pytest.mark.parametrize("terminal_status", ["invalidated", "executed"])
async def test_terminal_approval_never_repeats_execution(api_harness, terminal_status):
    cycle = _Cycle()
    _mount_cycle(api_harness, cycle)
    await _seed(cycle)
    if terminal_status == "invalidated":
        await cycle.approvals.invalidate("approval-1")
    else:
        await cycle.approvals.approve("approval-1")
        await cycle.approvals.claim_for_execution("approval-1", current_revision=9)
    cycle.execute_approved.reset_mock()

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "approve"},
    )

    assert response.status_code == 409
    cycle.execute_approved.assert_not_awaited()


async def test_unknown_approval_returns_not_found(api_harness):
    _mount_cycle(api_harness, _Cycle())

    response = await api_harness.client.get("/api/hitl/missing")

    assert response.status_code == 404


async def test_inactive_runtime_returns_fixed_unavailable_for_respond_and_detail(api_harness):
    api_harness.runtime.cycle = None

    respond = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "approve"},
    )
    detail = await api_harness.client.get("/api/hitl/approval-1")

    assert respond.status_code == 503
    assert respond.json() == {"detail": "Trading runtime is not active"}
    assert detail.status_code == 503
    assert detail.json() == {"detail": "Trading runtime is not active"}


async def test_concurrent_runtime_close_during_lease_acquisition_returns_fixed_unavailable(api_harness):
    cycle = _Cycle()
    await _seed(cycle)
    runtime = Runtime(
        snapshot=api_harness.runtime.snapshot,
        repository=api_harness.runtime.repository,
        cycle=cycle,
        sessions={},
        signal_registry=object(),
        market_registry=object(),
        venue_registry=object(),
        events=MultiplexedCycleEventSink(NullCycleEventSink()),
    )
    lease_entered = asyncio.Event()
    original_lease = runtime.cycle_lease

    @asynccontextmanager
    async def observed_lease():
        lease_entered.set()
        async with original_lease() as leased_cycle:
            yield leased_cycle

    runtime.cycle_lease = observed_lease
    api_harness.runtime = runtime
    from api.main import app

    app.state.runtime = runtime
    await runtime._lifecycle_lock.acquire()
    responding = asyncio.create_task(
        api_harness.client.post(
            "/api/hitl/approval-1/respond",
            json={"decision": "approve"},
        )
    )
    await lease_entered.wait()
    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    runtime._lifecycle_lock.release()

    response = await responding
    await closing
    assert response.status_code == 503
    assert response.json() == {"detail": "Trading runtime is not active"}
    cycle.execute_approved.assert_not_awaited()


async def test_execution_runtime_error_remains_internal_failure_not_unavailable(api_harness):
    cycle = _Cycle()
    _mount_cycle(api_harness, cycle)
    await _seed(cycle)
    cycle.execute_approved.side_effect = RuntimeError("execution body failed")

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "approve"},
    )

    assert response.status_code == 500
    assert response.status_code != 503


async def test_approval_state_error_returns_fixed_detail_without_internal_marker(api_harness):
    cycle = _Cycle()
    _mount_cycle(api_harness, cycle)
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
    _mount_cycle(api_harness, cycle)
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
    _mount_cycle(api_harness, _Cycle())

    response = await api_harness.client.post(
        "/api/hitl/approval-1/respond",
        json={"decision": "approve", "unexpected": True},
    )

    assert response.status_code == 422


async def test_approve_lease_pins_session_during_claim_reload_and_runtime_close(api_harness):
    from api.routes.hitl import HitlRespondIn, respond_approval

    cycle = _Cycle()
    await _seed(cycle)
    claimed = asyncio.Event()
    release = asyncio.Event()

    async def execute_after_claim(approval_id):
        await cycle.approvals.claim_for_execution(approval_id, current_revision=9)
        claimed.set()
        await release.wait()
        return _outcome("completed", "completed")

    cycle.execute_approved.side_effect = execute_after_claim

    class Session:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    old_session = Session()
    replacement_session = Session()
    runtime = Runtime(
        snapshot=api_harness.runtime.snapshot,
        repository=api_harness.runtime.repository,
        cycle=cycle,
        sessions={"old": old_session},
        signal_registry=object(),
        market_registry=object(),
        venue_registry=object(),
        events=MultiplexedCycleEventSink(NullCycleEventSink()),
    )
    reload_calls = 0

    async def reload_graph():
        nonlocal reload_calls
        reload_calls += 1
        if reload_calls == 2:
            runtime.sessions = {"replacement": replacement_session}
            await runtime._retire_sessions((old_session,))
        return cycle

    runtime._reload_for_cycle_locked = reload_graph
    request = type("Request", (), {"app": type("App", (), {"state": type("State", (), {"runtime": runtime})()})()})()

    responding = asyncio.create_task(respond_approval("approval-1", HitlRespondIn(decision="approve"), request))
    await claimed.wait()
    await runtime.reload_for_cycle()
    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)

    assert old_session.close_calls == 0
    assert replacement_session.close_calls == 0
    assert closing.done() is False

    release.set()
    response = await responding
    await closing
    assert response.approval_status == "executed"
    assert old_session.close_calls == 1
    assert replacement_session.close_calls == 1
