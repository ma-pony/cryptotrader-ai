"""资金池完整 proposal 的审批与同周期执行 API。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from api.routes.response_dto import BookExecutionProposalOut, proposal_out
from cryptotrader.hitl.store import ApprovalStateError

if TYPE_CHECKING:
    from cryptotrader.hitl.models import BookApproval

router = APIRouter(prefix="/api/hitl", tags=["hitl"])


class ApprovalRequestOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    cycle_id: str
    book_id: str
    pair: str
    config_revision: int
    proposal: BookExecutionProposalOut
    status: Literal["pending", "approved", "rejected", "invalidated", "executed"]
    created_at: str
    decided_at: str | None


class HitlRespondIn(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    decision: Literal["approve", "reject"]


class HitlRespondOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    cycle_id: str
    approval_status: str
    cycle_status: str
    execution_status: str
    requires_attention: bool


def _cycle(request: Request):
    runtime = _runtime(request)
    return runtime.cycle


def _runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or runtime.cycle is None:
        raise HTTPException(status_code=503, detail="Trading runtime is not active")
    return runtime


def _response(record: BookApproval) -> ApprovalRequestOut:
    return ApprovalRequestOut(
        approval_id=record.approval_id,
        cycle_id=record.cycle_id,
        book_id=record.book_id,
        pair=record.proposal.pair.canonical(),
        config_revision=record.config_revision,
        proposal=proposal_out(record.proposal),
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
    try:
        async with _runtime(request).cycle_lease() as cycle:
            if body.decision == "approve":
                await cycle.approvals.approve(approval_id)
                outcome = await cycle.execute_approved(approval_id)
            else:
                outcome = await cycle.reject_approval(approval_id)
            final_approval = await cycle.approvals.get(approval_id)
    except ApprovalStateError:
        raise HTTPException(status_code=409, detail="Approval state conflict") from None
    except LookupError:
        raise HTTPException(status_code=404, detail="Approval request not found") from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Approval proposal is invalid") from error
    if final_approval is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    return HitlRespondOut(
        approval_id=approval_id,
        cycle_id=outcome.cycle_id,
        approval_status=final_approval.status,
        cycle_status=outcome.status,
        execution_status=outcome.execution_status,
        requires_attention=outcome.requires_attention,
    )
