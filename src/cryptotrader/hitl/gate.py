"""HITL gate node — optional human approval before risk gate."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import structlog

from cryptotrader._compat import UTC
from cryptotrader.config import HitlConfig, load_config
from cryptotrader.hitl.store import ApprovalStore
from cryptotrader.state import get_pair

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)


def _should_trigger(state: dict[str, Any], config: HitlConfig) -> tuple[bool, str]:
    """Evaluate whether HITL approval is required.

    Returns (should_trigger, trigger_reason).
    """
    if not config.enabled:
        return False, ""

    metadata = state.get("metadata") or {}
    if metadata.get("backtest_mode"):
        return False, ""

    data = state.get("data") or {}
    verdict = data.get("verdict") or {}
    action = verdict.get("action", "hold")
    if action == "hold":
        return False, ""

    position_scale = verdict.get("position_scale", 0.0)
    if position_scale >= config.min_position_scale:
        return True, "position_scale"

    divergence_scores = state.get("divergence_scores") or []
    if divergence_scores and divergence_scores[-1] >= config.divergence_threshold:
        return True, "divergence"

    return False, ""


async def _should_trigger_with_cold_start(state: dict[str, Any], config: HitlConfig, db_url: str) -> tuple[bool, str]:
    """Full trigger check including cold-start (requires DB query)."""
    should, reason = _should_trigger(state, config)
    if should:
        return should, reason

    if not config.enabled:
        return False, ""

    metadata = state.get("metadata") or {}
    if metadata.get("backtest_mode"):
        return False, ""

    data = state.get("data") or {}
    verdict = data.get("verdict") or {}
    if verdict.get("action", "hold") == "hold":
        return False, ""

    if db_url and config.cold_start_min_trades > 0:
        try:
            count = await ApprovalStore.get_completed_trades_count(db_url)
            if count < config.cold_start_min_trades:
                return True, "cold_start"
        except Exception:
            logger.warning("Cold-start check failed, skipping", exc_info=True)

    return False, ""


async def hitl_gate(state: dict[str, Any]) -> dict:
    """LangGraph node: conditionally pause for human approval."""
    config = load_config()
    hitl_config = config.hitl
    metadata = state.get("metadata") or {}
    db_url = metadata.get("database_url", "")

    should, reason = await _should_trigger_with_cold_start(state, hitl_config, db_url)

    if not should:
        return {"hitl": {"skipped": True, "decision": "approve"}}

    data = state.get("data") or {}
    verdict = data.get("verdict") or {}
    analyses = data.get("analyses") or {}
    try:
        pair = get_pair(state).canonical()
    except (KeyError, TypeError, ValueError):
        pair = "unknown"
    thread_id = metadata.get("thread_id", "")

    approval_id = str(uuid4())
    timeout_s = hitl_config.approval_timeout_seconds
    expires_at = datetime.now(UTC) + timedelta(seconds=timeout_s)

    if db_url:
        await ApprovalStore.create(
            db_url,
            approval_id=approval_id,
            pair=pair,
            expires_at=expires_at,
            trigger_reason=reason,
            verdict_snapshot=json.dumps(verdict),
            agent_analyses_snapshot=json.dumps(_summarize_analyses(analyses)),
            thread_id=thread_id,
        )

    _slog.info(
        "hitl_gate_triggered",
        approval_id=approval_id,
        pair=pair,
        trigger_reason=reason,
        position_scale=verdict.get("position_scale"),
        timeout_seconds=timeout_s,
    )

    from cryptotrader.hitl.notifier import notify_hitl_request

    await notify_hitl_request(approval_id, pair, reason, verdict, analyses, hitl_config)

    from langgraph.types import interrupt

    decision_data = interrupt({"approval_id": approval_id, "trigger_reason": reason})

    decision = decision_data if isinstance(decision_data, str) else decision_data.get("decision", "reject")

    if db_url and decision in ("approve", "reject"):
        await ApprovalStore.decide(
            db_url,
            approval_id,
            status="approved" if decision == "approve" else "rejected",
            decision_by=decision_data.get("decision_by", "web") if isinstance(decision_data, dict) else "web",
        )

    _slog.info(
        "hitl_gate_decided",
        approval_id=approval_id,
        pair=pair,
        decision=decision,
    )

    return {
        "hitl": {
            "approval_id": approval_id,
            "decision": decision,
            "trigger_reason": reason,
            "skipped": False,
        }
    }


def hitl_router(state: dict[str, Any]) -> str:
    """Route after hitl_gate: 'pass' to risk_gate, 'rejected' to record_rejection."""
    hitl = state.get("hitl") or {}
    decision = hitl.get("decision", "approve")
    if decision in ("approve", ""):
        return "pass"
    return "rejected"


def _summarize_analyses(analyses: dict) -> list[dict]:
    """Extract compact agent summaries for the approval card."""
    summaries = []
    for agent_type, result in analyses.items():
        if not isinstance(result, dict):
            continue
        summaries.append(
            {
                "agent": agent_type,
                "direction": result.get("direction", "neutral"),
                "confidence": result.get("confidence", 0.0),
                "reasoning": (result.get("reasoning") or "")[:200],
            }
        )
    return summaries


async def _timeout_reject(
    approval_id: str,
    timeout_s: int,
    db_url: str,
    compiled_graph: Any,
    thread_id: str,
) -> None:
    """Background task: auto-reject after timeout if still pending."""
    await asyncio.sleep(timeout_s)

    if not db_url:
        return

    record = await ApprovalStore.get(db_url, approval_id)
    if record is None or record["status"] != "pending":
        return

    ok = await ApprovalStore.decide(
        db_url,
        approval_id,
        status="expired",
        decision_by="timeout",
    )
    if not ok:
        return

    _slog.warning(
        "hitl_timeout_reject",
        approval_id=approval_id,
    )

    try:
        from langgraph.types import Command

        await compiled_graph.ainvoke(
            Command(resume={"decision": "expired", "decision_by": "timeout"}),
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception:
        logger.warning("Failed to resume graph after timeout", exc_info=True)
