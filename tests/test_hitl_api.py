"""Tests for HITL API endpoints — pending, detail, respond."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

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


async def _seed(db_url: str, approval_id: str = "test-001", **kwargs):
    defaults = {
        "pair": "BTC/USDT",
        "expires_at": datetime.now(UTC) + timedelta(seconds=300),
        "trigger_reason": "position_scale",
        "verdict_snapshot": json.dumps(
            {"action": "long", "position_scale": 0.8, "confidence": 0.7, "reasoning": "test"}
        ),
        "agent_analyses_snapshot": json.dumps([{"agent": "tech", "direction": "bullish", "confidence": 0.8}]),
        "thread_id": "thread-1",
    }
    defaults.update(kwargs)
    await ApprovalStore.create(db_url, approval_id=approval_id, **defaults)


@pytest.mark.asyncio
async def test_get_pending_empty(db_url):
    from api.routes.hitl import list_pending

    with patch("api.routes.hitl._get_db_url", return_value=db_url):
        await ApprovalStore.ensure_table(db_url)
        result = await list_pending()
    assert result == []


@pytest.mark.asyncio
async def test_get_pending_returns_full_snapshot(db_url):
    from api.routes.hitl import list_pending

    await _seed(db_url)
    with patch("api.routes.hitl._get_db_url", return_value=db_url):
        result = await list_pending()
    assert len(result) == 1
    item = result[0]
    assert item.verdict_snapshot["action"] == "long"
    assert item.verdict_snapshot["position_scale"] == 0.8
    assert item.verdict_snapshot["confidence"] == 0.7
    assert item.verdict_snapshot["reasoning"] == "test"


@pytest.mark.asyncio
async def test_get_by_id_found(db_url):
    from api.routes.hitl import get_approval

    await _seed(db_url)
    with patch("api.routes.hitl._get_db_url", return_value=db_url):
        result = await get_approval("test-001")
    assert result.approval_id == "test-001"
    assert result.status == "pending"


@pytest.mark.asyncio
async def test_get_by_id_not_found(db_url):
    from fastapi import HTTPException

    from api.routes.hitl import get_approval

    await ApprovalStore.ensure_table(db_url)
    with (
        patch("api.routes.hitl._get_db_url", return_value=db_url),
        pytest.raises(HTTPException) as exc_info,
    ):
        await get_approval("nonexistent")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_respond_approve(db_url):
    from api.routes.hitl import HitlRespondIn, respond_approval

    await _seed(db_url)
    body = HitlRespondIn(decision="approve")
    with (
        patch("api.routes.hitl._get_db_url", return_value=db_url),
        patch("cryptotrader.hitl.notifier.notify_hitl_decision", new_callable=AsyncMock),
    ):
        result = await respond_approval("test-001", body)
    assert result.status == "approved"

    record = await ApprovalStore.get(db_url, "test-001")
    assert record["status"] == "approved"
    assert record["decision_by"] == "web"


@pytest.mark.asyncio
async def test_respond_reject(db_url):
    from api.routes.hitl import HitlRespondIn, respond_approval

    await _seed(db_url)
    body = HitlRespondIn(decision="reject", comment="too risky")
    with (
        patch("api.routes.hitl._get_db_url", return_value=db_url),
        patch("cryptotrader.hitl.notifier.notify_hitl_decision", new_callable=AsyncMock),
    ):
        result = await respond_approval("test-001", body)
    assert result.status == "rejected"

    record = await ApprovalStore.get(db_url, "test-001")
    assert record["status"] == "rejected"
    assert record["comment"] == "too risky"


@pytest.mark.asyncio
async def test_respond_conflict_409(db_url):
    from fastapi import HTTPException

    from api.routes.hitl import HitlRespondIn, respond_approval

    await _seed(db_url)
    body = HitlRespondIn(decision="approve")
    with (
        patch("api.routes.hitl._get_db_url", return_value=db_url),
        patch("cryptotrader.hitl.notifier.notify_hitl_decision", new_callable=AsyncMock),
    ):
        await respond_approval("test-001", body)
        with pytest.raises(HTTPException) as exc_info:
            await respond_approval("test-001", HitlRespondIn(decision="reject"))
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_respond_expired_409(db_url):
    from fastapi import HTTPException

    from api.routes.hitl import HitlRespondIn, respond_approval

    past = datetime.now(UTC) - timedelta(seconds=60)
    await _seed(db_url, expires_at=past)
    await ApprovalStore.expire_stale(db_url)

    body = HitlRespondIn(decision="approve")
    with (
        patch("api.routes.hitl._get_db_url", return_value=db_url),
        pytest.raises(HTTPException) as exc_info,
    ):
        await respond_approval("test-001", body)
    assert exc_info.value.status_code == 409
