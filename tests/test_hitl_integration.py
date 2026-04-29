"""Integration tests for HITL — full flow, timeout, concurrent, backtest bypass."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from cryptotrader._compat import UTC
from cryptotrader.config import HitlConfig
from cryptotrader.hitl.gate import _should_trigger, hitl_router
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


def _state(
    *,
    action: str = "long",
    position_scale: float = 0.8,
    confidence: float = 0.7,
    backtest_mode: bool = False,
    divergence_scores: list[float] | None = None,
) -> dict:
    return {
        "data": {
            "verdict": {
                "action": action,
                "position_scale": position_scale,
                "confidence": confidence,
                "reasoning": "integration test",
            },
            "analyses": {
                "tech": {"direction": "bullish", "confidence": 0.8},
                "chain": {"direction": "bullish", "confidence": 0.7},
            },
        },
        "metadata": {"backtest_mode": backtest_mode, "pair": "BTC/USDT"},
        "divergence_scores": divergence_scores or [],
        "hitl": {},
    }


@pytest.mark.asyncio
async def test_full_graph_hitl_disabled():
    """SC-001: hitl.enabled=False — gate passthrough, no DB writes."""
    from cryptotrader.hitl.gate import hitl_gate

    state = _state()
    with patch("cryptotrader.hitl.gate.load_config") as mock_cfg:
        mock_cfg.return_value.hitl = HitlConfig(enabled=False)
        result = await hitl_gate(state)

    assert result["hitl"]["skipped"] is True
    assert result["hitl"]["decision"] == "approve"
    assert hitl_router({**state, **result}) == "pass"


@pytest.mark.asyncio
async def test_backtest_mode_zero_io(db_url):
    """SC-007: backtest_mode=True — gate passthrough, no ApprovalStore writes."""
    from cryptotrader.hitl.gate import hitl_gate

    await ApprovalStore.ensure_table(db_url)
    state = _state(backtest_mode=True, position_scale=0.9)

    with patch("cryptotrader.hitl.gate.load_config") as mock_cfg:
        cfg = HitlConfig(enabled=True, min_position_scale=0.5)
        mock_cfg.return_value.hitl = cfg
        t0 = time.monotonic()
        result = await hitl_gate(state)
        elapsed = time.monotonic() - t0

    assert result["hitl"]["skipped"] is True
    assert elapsed < 0.01

    pending = await ApprovalStore.list_pending(db_url)
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_hitl_trigger_creates_approval_and_interrupts(db_url):
    """SC-002: hitl triggers, creates approval in DB, calls interrupt()."""
    from cryptotrader.hitl.gate import hitl_gate

    state = _state(position_scale=0.75)
    state["metadata"]["database_url"] = db_url
    state["metadata"]["thread_id"] = "test-thread-1"

    with (
        patch("cryptotrader.hitl.gate.load_config") as mock_cfg,
        patch("cryptotrader.hitl.notifier.notify_hitl_request", new_callable=AsyncMock),
        patch(
            "langgraph.types.interrupt",
            return_value={"decision": "approve", "decision_by": "web"},
        ) as mock_interrupt,
    ):
        mock_cfg.return_value.hitl = HitlConfig(enabled=True, min_position_scale=0.5)
        mock_cfg.return_value.infrastructure.database_url = db_url
        result = await hitl_gate(state)

    mock_interrupt.assert_called_once()
    assert result["hitl"]["decision"] == "approve"
    assert result["hitl"]["skipped"] is False
    assert result["hitl"]["trigger_reason"] == "position_scale"


@pytest.mark.asyncio
async def test_hitl_trigger_reject(db_url):
    """Rejected approval routes to 'rejected'."""
    from cryptotrader.hitl.gate import hitl_gate

    state = _state(position_scale=0.75)
    state["metadata"]["database_url"] = db_url
    state["metadata"]["thread_id"] = "test-thread-2"

    with (
        patch("cryptotrader.hitl.gate.load_config") as mock_cfg,
        patch("cryptotrader.hitl.notifier.notify_hitl_request", new_callable=AsyncMock),
        patch("langgraph.types.interrupt", return_value={"decision": "reject", "decision_by": "web"}),
    ):
        mock_cfg.return_value.hitl = HitlConfig(enabled=True, min_position_scale=0.5)
        mock_cfg.return_value.infrastructure.database_url = db_url
        result = await hitl_gate(state)

    assert result["hitl"]["decision"] == "reject"
    assert hitl_router({**state, **result}) == "rejected"


@pytest.mark.asyncio
async def test_concurrent_respond_one_wins(db_url):
    """SC-006: two concurrent decide() calls — exactly one succeeds."""
    expires = datetime.now(UTC) + timedelta(seconds=300)
    await ApprovalStore.create(
        db_url,
        approval_id="concurrent-test",
        pair="BTC/USDT",
        expires_at=expires,
        trigger_reason="position_scale",
        verdict_snapshot=json.dumps({"action": "long", "position_scale": 0.8}),
        agent_analyses_snapshot=json.dumps([{"agent": "tech", "direction": "bullish", "confidence": 0.8}]),
        thread_id="thread-concurrent",
    )

    results = await asyncio.gather(
        ApprovalStore.decide(db_url, "concurrent-test", status="approved", decision_by="web"),
        ApprovalStore.decide(db_url, "concurrent-test", status="rejected", decision_by="telegram"),
    )
    assert sorted(results) == [False, True]

    record = await ApprovalStore.get(db_url, "concurrent-test")
    assert record["status"] in ("approved", "rejected")


@pytest.mark.asyncio
async def test_pending_api_returns_full_snapshot(db_url):
    """SC-005: verdict_snapshot includes action, position_scale, confidence, reasoning."""
    verdict = {"action": "long", "position_scale": 0.8, "confidence": 0.7, "reasoning": "test reasoning"}
    analyses = [{"agent": "tech", "direction": "bullish", "confidence": 0.8}]

    await ApprovalStore.create(
        db_url,
        approval_id="snapshot-test",
        pair="BTC/USDT",
        expires_at=datetime.now(UTC) + timedelta(seconds=300),
        trigger_reason="position_scale",
        verdict_snapshot=json.dumps(verdict),
        agent_analyses_snapshot=json.dumps(analyses),
        thread_id="thread-snapshot",
    )

    from api.routes.hitl import list_pending

    with patch("api.routes.hitl._get_db_url", return_value=db_url):
        result = await list_pending()

    assert len(result) == 1
    item = result[0]
    assert item.verdict_snapshot["action"] == "long"
    assert item.verdict_snapshot["position_scale"] == 0.8
    assert item.verdict_snapshot["confidence"] == 0.7
    assert item.verdict_snapshot["reasoning"] == "test reasoning"
    assert len(item.agent_analyses_snapshot) == 1
    assert item.agent_analyses_snapshot[0]["agent"] == "tech"


@pytest.mark.asyncio
async def test_expire_stale_on_startup(db_url):
    """SC-003 variant: expired approvals are cleaned up."""
    past = datetime.now(UTC) - timedelta(seconds=60)
    await ApprovalStore.create(
        db_url,
        approval_id="stale-int-1",
        pair="BTC/USDT",
        expires_at=past,
        trigger_reason="cold_start",
        verdict_snapshot="{}",
        agent_analyses_snapshot="[]",
        thread_id="t-stale",
    )

    count = await ApprovalStore.expire_stale(db_url)
    assert count == 1

    record = await ApprovalStore.get(db_url, "stale-int-1")
    assert record["status"] == "expired"
    assert record["decision_by"] == "timeout"


@pytest.mark.asyncio
async def test_trigger_divergence_threshold():
    """Divergence above threshold triggers HITL."""
    cfg = HitlConfig(enabled=True, divergence_threshold=0.6, min_position_scale=0.99)
    state = _state(position_scale=0.3, divergence_scores=[0.3, 0.65])
    should, reason = _should_trigger(state, cfg)
    assert should is True
    assert reason == "divergence"


@pytest.mark.asyncio
async def test_hold_action_never_triggers():
    """Hold action never triggers regardless of other thresholds."""
    cfg = HitlConfig(enabled=True, min_position_scale=0.0)
    state = _state(action="hold", position_scale=1.0)
    should, _reason = _should_trigger(state, cfg)
    assert should is False
