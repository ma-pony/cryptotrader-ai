"""冻结 TradePlan 的人工审批查询与周期恢复 API。"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from cryptotrader.cycle_serialization import trade_plan_payload
from cryptotrader.hitl.store import ApprovalRecord, ApprovalStateError

router = APIRouter(prefix="/api/hitl", tags=["hitl"])


class ApprovalRequestOut(BaseModel):
    approval_id: str
    cycle_id: str
    pair: str
    profile_revision: int
    trade_plan: dict
    status: Literal["pending", "approved", "rejected"]
    decision_by: str | None
    created_at: str
    decided_at: str | None


class HitlRespondIn(BaseModel):
    decision: Literal["approve", "reject"]


class HitlRespondOut(BaseModel):
    approval_id: str
    cycle_id: str
    status: str
    cycle_status: str


def _cycle(request: Request):
    cycle = getattr(request.app.state, "trading_cycle", None)
    if cycle is None:
        raise HTTPException(status_code=503, detail="Trading cycle is not initialized")
    return cycle


def _response(record: ApprovalRecord) -> ApprovalRequestOut:
    return ApprovalRequestOut(
        approval_id=record.approval_id,
        cycle_id=record.cycle_id,
        pair=record.pair,
        profile_revision=record.profile_revision,
        trade_plan=trade_plan_payload(record.plan),
        status=record.status,
        decision_by=record.decision_by,
        created_at=record.created_at.isoformat(),
        decided_at=record.decided_at.isoformat() if record.decided_at is not None else None,
    )


@router.get("/pending")
async def list_pending(request: Request) -> list[ApprovalRequestOut]:
    records = await _cycle(request).approvals.list_pending()
    return [_response(record) for record in records]


@router.get("/{approval_id}")
async def get_approval(approval_id: str, request: Request) -> ApprovalRequestOut:
    record = await _cycle(request).approvals.get(approval_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    return _response(record)


@router.post("/{approval_id}/respond")
async def respond_approval(
    approval_id: str,
    body: HitlRespondIn,
    request: Request,
) -> HitlRespondOut:
    cycle = _cycle(request)
    if await cycle.approvals.get(approval_id) is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    try:
        if body.decision == "approve":
            outcome = await cycle.resume_approved(approval_id, decision_by="web")
            approval_status = "approved"
        else:
            outcome = await cycle.reject_approval(approval_id, decision_by="web")
            approval_status = "rejected"
    except ApprovalStateError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return HitlRespondOut(
        approval_id=approval_id,
        cycle_id=outcome.cycle_id,
        status=approval_status,
        cycle_status=outcome.status,
    )
