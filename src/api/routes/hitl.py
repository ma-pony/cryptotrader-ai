"""资金池完整 proposal 的审批与同周期执行 API。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from cryptotrader.execution.codec import book_execution_proposal_payload
from cryptotrader.hitl.store import ApprovalStateError

if TYPE_CHECKING:
    from cryptotrader.hitl.models import BookApproval

router = APIRouter(prefix="/api/hitl", tags=["hitl"])


class ApprovalRequestOut(BaseModel):
    approval_id: str
    cycle_id: str
    book_id: str
    pair: str
    config_revision: int
    proposal: dict
    status: Literal["pending", "approved", "rejected", "invalidated", "executed"]
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
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.cycle is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not active")
    return runtime.cycle


def _response(record: BookApproval) -> ApprovalRequestOut:
    return ApprovalRequestOut(
        approval_id=record.approval_id,
        cycle_id=record.cycle_id,
        book_id=record.book_id,
        pair=record.proposal.pair.canonical(),
        config_revision=record.config_revision,
        proposal=book_execution_proposal_payload(record.proposal),
        status=record.status,
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
async def respond_approval(approval_id: str, body: HitlRespondIn, request: Request) -> HitlRespondOut:
    cycle = _cycle(request)
    try:
        if body.decision == "approve":
            await cycle.approvals.approve(approval_id)
            outcome = await cycle.execute_approved(approval_id)
            approval_status = "executed"
        else:
            outcome = await cycle.reject_approval(approval_id)
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
