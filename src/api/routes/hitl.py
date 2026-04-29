"""HITL approval API — pending list, detail, and respond endpoints."""

from __future__ import annotations

import json
import logging
import time
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cryptotrader.config import load_config
from cryptotrader.hitl.store import ApprovalStore

logger = logging.getLogger(__name__)
_slog = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/hitl", tags=["hitl"])


class ApprovalRequestOut(BaseModel):
    approval_id: str
    pair: str
    created_at: str | None
    expires_at: str | None
    trigger_reason: str
    verdict_snapshot: dict
    agent_analyses_snapshot: list[dict]
    status: Literal["pending", "approved", "rejected", "expired"]
    decision_by: str | None
    decided_at: str | None


class HitlRespondIn(BaseModel):
    decision: Literal["approve", "reject"]
    comment: str = ""


class HitlRespondOut(BaseModel):
    approval_id: str
    status: str
    message: str


def _parse_json_field(value: str | dict | list | None, fallback: object = None) -> object:
    if value is None:
        return fallback
    if isinstance(value, dict | list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _to_response(record: dict) -> ApprovalRequestOut:
    # Coerce snapshot fields to the model's expected shape:
    # - verdict_snapshot: dict (wrap if list slipped in)
    # - agent_analyses_snapshot: list[dict] (wrap if dict slipped in via
    #   manual DB seed / older migration). Production always emits a list.
    verdict = _parse_json_field(record.get("verdict_snapshot"), {})
    if isinstance(verdict, list):
        verdict = verdict[0] if verdict and isinstance(verdict[0], dict) else {}
    analyses = _parse_json_field(record.get("agent_analyses_snapshot"), [])
    if isinstance(analyses, dict):
        # Old/test shape: {agent_id: summary}. Reshape to list of dicts.
        analyses = [{"name": k, "summary": v} for k, v in analyses.items()]
    elif not isinstance(analyses, list):
        analyses = []

    return ApprovalRequestOut(
        approval_id=record["approval_id"],
        pair=record["pair"],
        created_at=record.get("created_at"),
        expires_at=record.get("expires_at"),
        trigger_reason=record["trigger_reason"],
        verdict_snapshot=verdict,
        agent_analyses_snapshot=analyses,
        status=record["status"],
        decision_by=record.get("decision_by"),
        decided_at=record.get("decided_at"),
    )


def _get_db_url() -> str:
    config = load_config()
    db_url = config.infrastructure.database_url
    if not db_url:
        raise HTTPException(status_code=503, detail="Database not configured")
    return db_url


@router.get("/pending")
async def list_pending() -> list[ApprovalRequestOut]:
    db_url = _get_db_url()
    records = await ApprovalStore.list_pending(db_url)
    return [_to_response(r) for r in records]


@router.get("/{approval_id}")
async def get_approval(approval_id: str) -> ApprovalRequestOut:
    db_url = _get_db_url()
    record = await ApprovalStore.get(db_url, approval_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    return _to_response(record)


@router.post("/{approval_id}/respond")
async def respond_approval(approval_id: str, body: HitlRespondIn) -> HitlRespondOut:
    db_url = _get_db_url()
    t0 = time.monotonic()

    record = await ApprovalStore.get(db_url, approval_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Approval request not found")

    ok = await ApprovalStore.decide(
        db_url,
        approval_id,
        status="approved" if body.decision == "approve" else "rejected",
        decision_by="web",
        comment=body.comment,
    )
    if not ok:
        raise HTTPException(
            status_code=409,
            detail="Approval already decided or expired",
        )

    latency = time.monotonic() - t0

    from cryptotrader.hitl.notifier import notify_hitl_decision

    await notify_hitl_decision(
        approval_id=approval_id,
        pair=record["pair"],
        trigger_reason=record["trigger_reason"],
        decision=body.decision,
        decision_by="web",
        latency_seconds=latency,
    )

    _slog.info(
        "hitl_respond",
        approval_id=approval_id,
        decision=body.decision,
        decision_by="web",
        latency_seconds=round(latency, 3),
    )

    return HitlRespondOut(
        approval_id=approval_id,
        status="approved" if body.decision == "approve" else "rejected",
        message=f"Decision '{body.decision}' recorded",
    )
