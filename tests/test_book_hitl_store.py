"""资金池审批必须绑定精确 proposal、revision, 并且只能领取一次。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from cryptotrader.execution.models import BookExecutionProposal
from tests.test_execution_coordinator import _proposal


def _book_proposal(*, config_revision: int = 7) -> BookExecutionProposal:
    return replace(_proposal(), config_revision=config_revision)


def _non_ready_proposal(*, config_revision: int = 7) -> BookExecutionProposal:
    return replace(_book_proposal(config_revision=config_revision), connection_plans=(), ready=False)


@pytest.mark.parametrize(
    ("status", "decided", "claimed"),
    [
        ("pending", False, False),
        ("approved", True, False),
        ("rejected", True, False),
        ("invalidated", True, False),
        ("executed", True, True),
    ],
)
def test_book_approval_is_strictly_immutable_and_validates_time_state(status, decided, claimed):
    from cryptotrader.hitl.models import BookApproval

    created_at = datetime(2026, 8, 29, 1, tzinfo=UTC)
    approval = BookApproval(
        "approval-1",
        "cycle-1",
        "simulation",
        7,
        _book_proposal(),
        status,
        created_at,
        created_at + timedelta(seconds=1) if decided else None,
        created_at + timedelta(seconds=2) if claimed else None,
    )

    with pytest.raises(FrozenInstanceError):
        approval.status = "rejected"


def test_book_approval_rejects_identity_and_time_mismatches():
    from cryptotrader.hitl.models import BookApproval

    now = datetime(2026, 8, 29, 1, tzinfo=UTC)
    proposal = _book_proposal()

    with pytest.raises(ValueError, match="book_id"):
        BookApproval("approval", "cycle", "other", 7, proposal, "pending", now, None, None)
    with pytest.raises(ValueError, match="config_revision"):
        BookApproval("approval", "cycle", proposal.book_id, 8, proposal, "pending", now, None, None)
    with pytest.raises(ValueError, match="UTC"):
        BookApproval(
            "approval",
            "cycle",
            proposal.book_id,
            7,
            proposal,
            "pending",
            now.replace(tzinfo=None),
            None,
            None,
        )
    with pytest.raises(ValueError, match="pending"):
        BookApproval("approval", "cycle", proposal.book_id, 7, proposal, "pending", now, now, None)
    with pytest.raises(ValueError, match="ready"):
        BookApproval("approval", "cycle", proposal.book_id, 7, _non_ready_proposal(), "pending", now, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_store_rejects_non_ready_proposal_before_create(tmp_path, database):
    from cryptotrader.hitl.store import BookApprovalStore

    store = BookApprovalStore(f"sqlite+aiosqlite:///{tmp_path / 'non-ready.db'}" if database else None)

    with pytest.raises(ValueError, match="ready"):
        await store.create(_non_ready_proposal(), cycle_id="cycle-non-ready")
    assert await store.list_pending() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_memory_and_sqlite_round_trip_the_exact_proposal_and_list_pending(tmp_path, database):
    from cryptotrader.hitl.store import BookApprovalStore

    store = BookApprovalStore(f"sqlite+aiosqlite:///{tmp_path / 'round-trip.db'}" if database else None)
    proposal = _book_proposal()

    approval = await store.create(
        proposal,
        cycle_id="cycle-memory",
        approval_id="approval-memory",
        created_at=datetime(2026, 8, 29, 1, tzinfo=UTC),
    )

    assert approval.proposal == proposal
    assert await store.get(approval.id) == approval
    assert await store.list_pending() == [approval]


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_status", ["pending", "approved"])
@pytest.mark.parametrize("database", [False, True])
async def test_revision_change_invalidates_unexecuted_approval(tmp_path, initial_status, database):
    from cryptotrader.hitl.store import ApprovalInvalidated, BookApprovalStore

    store = BookApprovalStore(f"sqlite+aiosqlite:///{tmp_path / 'revision.db'}" if database else None)
    approval = await store.create(_book_proposal(), cycle_id=f"cycle-{initial_status}")
    if initial_status == "approved":
        await store.approve(approval.id)

    with pytest.raises(ApprovalInvalidated):
        await store.claim_for_execution(approval.id, current_revision=8)

    invalidated = await store.get(approval.id)
    assert invalidated is not None
    assert invalidated.status == "invalidated"
    assert invalidated.decided_at is not None
    assert invalidated.claimed_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_pending_rejected_missing_and_claimed_states_raise_distinct_safe_errors(tmp_path, database):
    from cryptotrader.hitl.store import (
        ApprovalAlreadyClaimed,
        ApprovalNotApproved,
        ApprovalNotFound,
        ApprovalRejected,
        BookApprovalStore,
    )

    store = BookApprovalStore(f"sqlite+aiosqlite:///{tmp_path / 'states.db'}" if database else None)
    pending = await store.create(_book_proposal(), cycle_id="pending")
    rejected = await store.create(_book_proposal(), cycle_id="rejected")
    await store.reject(rejected.id)
    approved = await store.create(_book_proposal(), cycle_id="approved")
    await store.approve(approved.id)
    await store.claim_for_execution(approved.id, current_revision=7)

    cases = (
        (pending.id, ApprovalNotApproved),
        (rejected.id, ApprovalRejected),
        (approved.id, ApprovalAlreadyClaimed),
        ("missing", ApprovalNotFound),
    )
    for approval_id, error_type in cases:
        with pytest.raises(error_type) as captured:
            await store.claim_for_execution(approval_id, current_revision=7)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("database", [False, True])
async def test_approve_and_reject_only_accept_pending_approvals(tmp_path, database):
    from cryptotrader.hitl.store import ApprovalStateError, BookApprovalStore

    store = BookApprovalStore(f"sqlite+aiosqlite:///{tmp_path / 'decisions.db'}" if database else None)
    approved = await store.create(_book_proposal(), cycle_id="approved")
    rejected = await store.create(_book_proposal(), cycle_id="rejected")

    await store.approve(approved.id)
    await store.reject(rejected.id)

    with pytest.raises(ApprovalStateError):
        await store.reject(approved.id)
    with pytest.raises(ApprovalStateError):
        await store.approve(rejected.id)


@pytest.mark.asyncio
async def test_sqlite_concurrent_claim_executes_exact_saved_proposal_once(tmp_path):
    from cryptotrader.hitl.store import BookApprovalStore

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'approvals.db'}"
    creator = BookApprovalStore(database_url)
    approval = await creator.create(_book_proposal(), cycle_id="cycle-concurrent")
    await creator.approve(approval.id)

    first, second = await asyncio.gather(
        BookApprovalStore(database_url).claim_for_execution(approval.id, current_revision=7),
        BookApprovalStore(database_url).claim_for_execution(approval.id, current_revision=7),
        return_exceptions=True,
    )

    claims = [item for item in (first, second) if isinstance(item, BookExecutionProposal)]
    assert len(claims) == 1
    assert claims[0] == approval.proposal
    assert claims[0].weights == (Decimal("0.4"), Decimal("0.6"))
    persisted = await creator.get(approval.id)
    assert persisted is not None
    assert persisted.status == "executed"
    assert persisted.claimed_at is not None


@pytest.mark.asyncio
async def test_sqlite_approve_claim_race_returns_exact_approved_transition_snapshot(tmp_path):
    from cryptotrader.hitl.store import ApprovalNotApproved, BookApprovalStore

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'approve-claim-race.db'}"
    creator = BookApprovalStore(database_url)
    approval = await creator.create(_book_proposal(), cycle_id="cycle-approve-claim-race")

    approved, claimed = await asyncio.gather(
        BookApprovalStore(database_url).approve(approval.id),
        BookApprovalStore(database_url).claim_for_execution(approval.id, current_revision=7),
        return_exceptions=True,
    )

    assert not isinstance(approved, Exception)
    assert approved.status == "approved"
    assert isinstance(claimed, (BookExecutionProposal, ApprovalNotApproved))
    persisted = await creator.get(approval.id)
    assert persisted is not None
    assert persisted.status == ("executed" if isinstance(claimed, BookExecutionProposal) else "approved")


@pytest.mark.asyncio
async def test_memory_claim_revalidates_approved_proposal_before_execution():
    from cryptotrader.hitl.store import BookApprovalStore

    store = BookApprovalStore()
    approval = await store.create(_book_proposal(), cycle_id="cycle-memory-corrupt")
    await store.approve(approval.id)
    object.__setattr__(approval.proposal, "ready", False)

    with pytest.raises(ValueError, match="ready"):
        await store.claim_for_execution(approval.id, current_revision=7)
    assert store.records[0].status == "approved"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["approve", "reject", "claim"])
async def test_sqlite_state_transition_rolls_back_when_returned_envelope_is_corrupt(tmp_path, operation):
    from cryptotrader.hitl.store import BookApprovalStore

    path = tmp_path / f"rollback-{operation}.db"
    store = BookApprovalStore(f"sqlite+aiosqlite:///{path}")
    approval = await store.create(_book_proposal(), cycle_id="cycle-rollback")
    if operation == "claim":
        await store.approve(approval.id)
    with sqlite3.connect(path) as connection:
        envelope = json.loads(
            connection.execute(
                "SELECT proposal_json FROM book_approvals WHERE approval_id = ?",
                (approval.id,),
            ).fetchone()[0]
        )
        envelope["proposal"]["book_id"] = "RAW_SECRET_PAYLOAD"
        connection.execute(
            "UPDATE book_approvals SET proposal_json = ? WHERE approval_id = ?",
            (json.dumps(envelope), approval.id),
        )

    async def transition():
        if operation == "claim":
            return await store.claim_for_execution(approval.id, current_revision=7)
        return await getattr(store, operation)(approval.id)

    with pytest.raises(ValueError, match="stored approval payload") as captured:
        await transition()
    with sqlite3.connect(path) as connection:
        persisted_status = connection.execute(
            "SELECT status FROM book_approvals WHERE approval_id = ?",
            (approval.id,),
        ).fetchone()[0]
    assert persisted_status == ("approved" if operation == "claim" else "pending")
    assert "RAW_SECRET_PAYLOAD" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_approve_returns_its_exact_transition_snapshot_without_post_commit_reread(tmp_path, monkeypatch):
    from cryptotrader.hitl.store import BookApprovalStore

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'approve-race.db'}"
    approver = BookApprovalStore(database_url)
    claimer = BookApprovalStore(database_url)
    approval = await approver.create(_book_proposal(), cycle_id="cycle-race")
    original_get = approver.get
    reread_called = False

    async def raced_get(approval_id):
        nonlocal reread_called
        reread_called = True
        await claimer.claim_for_execution(approval_id, current_revision=7)
        return await original_get(approval_id)

    monkeypatch.setattr(approver, "get", raced_get)

    approved = await approver.approve(approval.id)

    assert approved.status == "approved"
    assert reread_called is False
    assert await claimer.claim_for_execution(approval.id, current_revision=7) == approval.proposal


@pytest.mark.asyncio
async def test_sqlite_schema_has_exact_book_approval_columns(tmp_path):
    from cryptotrader.hitl.store import BookApprovalStore

    path = tmp_path / "schema.db"
    store = BookApprovalStore(f"sqlite+aiosqlite:///{path}")
    await store.ensure_table()

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(book_approvals)")}

    assert columns == {
        "approval_id",
        "cycle_id",
        "book_id",
        "config_revision",
        "proposal_json",
        "status",
        "created_at",
        "decided_at",
        "claimed_at",
    }


@pytest.mark.asyncio
async def test_sqlite_proposal_json_is_a_strict_versioned_identity_envelope(tmp_path):
    from cryptotrader.hitl.store import BookApprovalStore

    path = tmp_path / "envelope.db"
    store = BookApprovalStore(f"sqlite+aiosqlite:///{path}")
    approval = await store.create(
        _book_proposal(),
        approval_id="approval-envelope",
        cycle_id="cycle-envelope",
    )

    with sqlite3.connect(path) as connection:
        envelope = json.loads(
            connection.execute(
                "SELECT proposal_json FROM book_approvals WHERE approval_id = ?",
                (approval.id,),
            ).fetchone()[0]
        )

    assert set(envelope) == {"version", "approval_id", "cycle_id", "book_id", "config_revision", "proposal"}
    assert envelope["version"] == 1
    assert envelope["approval_id"] == approval.approval_id
    assert envelope["cycle_id"] == approval.cycle_id
    assert envelope["book_id"] == approval.book_id
    assert envelope["config_revision"] == approval.config_revision


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "field_name", "invalid_value"),
    [
        ("envelope", "version", True),
        ("envelope", "approval_id", "other-approval"),
        ("envelope", "cycle_id", "other-cycle"),
        ("envelope", "book_id", "other-book"),
        ("envelope", "config_revision", 8),
        ("proposal", "book_id", "RAW_SECRET_PAYLOAD"),
        ("proposal", "config_revision", 8),
        ("row", "cycle_id", "other-cycle"),
        ("row", "book_id", "other-book"),
        ("row", "config_revision", 8),
    ],
)
async def test_corrupt_or_identity_mismatched_sqlite_payload_fails_closed(
    tmp_path,
    target,
    field_name,
    invalid_value,
):
    from cryptotrader.hitl.store import BookApprovalStore

    path = tmp_path / "corrupt.db"
    store = BookApprovalStore(f"sqlite+aiosqlite:///{path}")
    approval = await store.create(_book_proposal(), cycle_id="cycle-corrupt")

    with sqlite3.connect(path) as connection:
        proposal_json = json.loads(
            connection.execute(
                "SELECT proposal_json FROM book_approvals WHERE approval_id = ?",
                (approval.id,),
            ).fetchone()[0]
        )
        if target == "proposal":
            proposal_json["proposal"][field_name] = invalid_value
        elif target == "envelope":
            proposal_json[field_name] = invalid_value
        else:
            statements = {
                "cycle_id": "UPDATE book_approvals SET cycle_id = ? WHERE approval_id = ?",
                "book_id": "UPDATE book_approvals SET book_id = ? WHERE approval_id = ?",
                "config_revision": "UPDATE book_approvals SET config_revision = ? WHERE approval_id = ?",
            }
            connection.execute(statements[field_name], (invalid_value, approval.id))
        if target != "row":
            connection.execute(
                "UPDATE book_approvals SET proposal_json = ? WHERE approval_id = ?",
                (json.dumps(proposal_json), approval.id),
            )

    with pytest.raises(ValueError, match="stored approval payload") as captured:
        await store.get(approval.id)
    assert "RAW_SECRET_PAYLOAD" not in repr(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
