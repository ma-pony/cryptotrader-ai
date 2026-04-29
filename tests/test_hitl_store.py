"""Tests for HITL ApprovalStore — CRUD, CAS concurrency, expiry."""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta

import pytest

from cryptotrader._compat import UTC
from cryptotrader.hitl.store import ApprovalStore, _table_ready


@pytest.fixture(autouse=True)
def _clear_table_cache():
    _table_ready.clear()
    yield
    _table_ready.clear()


@pytest.fixture
def db_url():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        yield f"sqlite+aiosqlite:///{f.name}"


@pytest.mark.asyncio
async def test_create_and_get(db_url):
    expires = datetime.now(UTC) + timedelta(seconds=300)
    await ApprovalStore.create(
        db_url,
        approval_id="test-001",
        pair="BTC/USDT",
        expires_at=expires,
        trigger_reason="position_scale",
        verdict_snapshot=json.dumps({"action": "long", "position_scale": 0.8}),
        agent_analyses_snapshot=json.dumps([]),
        thread_id="thread-1",
    )
    record = await ApprovalStore.get(db_url, "test-001")
    assert record is not None
    assert record["approval_id"] == "test-001"
    assert record["pair"] == "BTC/USDT"
    assert record["status"] == "pending"
    assert record["trigger_reason"] == "position_scale"
    assert record["thread_id"] == "thread-1"


@pytest.mark.asyncio
async def test_list_pending(db_url):
    expires = datetime.now(UTC) + timedelta(seconds=300)
    for i in range(3):
        await ApprovalStore.create(
            db_url,
            approval_id=f"test-{i:03d}",
            pair="BTC/USDT",
            expires_at=expires,
            trigger_reason="position_scale",
            verdict_snapshot="{}",
            agent_analyses_snapshot="[]",
            thread_id=f"thread-{i}",
        )

    await ApprovalStore.decide(db_url, "test-001", status="approved", decision_by="web")

    pending = await ApprovalStore.list_pending(db_url)
    assert len(pending) == 2
    assert all(r["status"] == "pending" for r in pending)


@pytest.mark.asyncio
async def test_decide_approve(db_url):
    expires = datetime.now(UTC) + timedelta(seconds=300)
    await ApprovalStore.create(
        db_url,
        approval_id="test-approve",
        pair="ETH/USDT",
        expires_at=expires,
        trigger_reason="divergence",
        verdict_snapshot="{}",
        agent_analyses_snapshot="[]",
        thread_id="thread-x",
    )

    ok = await ApprovalStore.decide(db_url, "test-approve", status="approved", decision_by="web", comment="looks good")
    assert ok is True

    record = await ApprovalStore.get(db_url, "test-approve")
    assert record["status"] == "approved"
    assert record["decision_by"] == "web"
    assert record["decided_at"] is not None
    assert record["comment"] == "looks good"


@pytest.mark.asyncio
async def test_decide_concurrent_409(db_url):
    """Two concurrent decide() calls — exactly one succeeds."""
    expires = datetime.now(UTC) + timedelta(seconds=300)
    await ApprovalStore.create(
        db_url,
        approval_id="test-cas",
        pair="BTC/USDT",
        expires_at=expires,
        trigger_reason="position_scale",
        verdict_snapshot="{}",
        agent_analyses_snapshot="[]",
        thread_id="thread-cas",
    )

    results = await asyncio.gather(
        ApprovalStore.decide(db_url, "test-cas", status="approved", decision_by="web"),
        ApprovalStore.decide(db_url, "test-cas", status="rejected", decision_by="telegram"),
    )
    assert sorted(results) == [False, True]


@pytest.mark.asyncio
async def test_expire_stale(db_url):
    past = datetime.now(UTC) - timedelta(seconds=60)
    future = datetime.now(UTC) + timedelta(seconds=300)

    await ApprovalStore.create(
        db_url,
        approval_id="stale-1",
        pair="BTC/USDT",
        expires_at=past,
        trigger_reason="cold_start",
        verdict_snapshot="{}",
        agent_analyses_snapshot="[]",
        thread_id="t-1",
    )
    await ApprovalStore.create(
        db_url,
        approval_id="fresh-1",
        pair="ETH/USDT",
        expires_at=future,
        trigger_reason="position_scale",
        verdict_snapshot="{}",
        agent_analyses_snapshot="[]",
        thread_id="t-2",
    )

    count = await ApprovalStore.expire_stale(db_url)
    assert count == 1

    stale = await ApprovalStore.get(db_url, "stale-1")
    assert stale["status"] == "expired"
    assert stale["decision_by"] == "timeout"

    fresh = await ApprovalStore.get(db_url, "fresh-1")
    assert fresh["status"] == "pending"


@pytest.mark.asyncio
async def test_get_nonexistent(db_url):
    await ApprovalStore.ensure_table(db_url)
    record = await ApprovalStore.get(db_url, "nonexistent")
    assert record is None
