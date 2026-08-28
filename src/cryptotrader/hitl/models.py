"""资金池级人工审批领域模型。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from cryptotrader.execution.models import BookExecutionProposal

BookApprovalStatus = Literal["pending", "approved", "rejected", "invalidated", "executed"]
_APPROVAL_STATUSES = frozenset({"pending", "approved", "rejected", "invalidated", "executed"})


def _require_utc(value: datetime | None, field_name: str) -> None:
    if value is None or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be UTC-aware")


@dataclass(frozen=True)
class BookApproval:
    """绑定一个资金池完整执行 proposal 与配置 revision 的不可变审批。"""

    approval_id: str
    cycle_id: str
    book_id: str
    config_revision: int
    proposal: BookExecutionProposal
    status: BookApprovalStatus
    created_at: datetime
    decided_at: datetime | None
    claimed_at: datetime | None

    def __post_init__(self) -> None:
        for field_name in ("approval_id", "cycle_id", "book_id"):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if type(self.config_revision) is not int or self.config_revision < 0:
            raise ValueError("config_revision must be a non-negative integer")
        if not isinstance(self.proposal, BookExecutionProposal):
            raise ValueError("proposal must be a BookExecutionProposal")
        if self.proposal.book_id != self.book_id:
            raise ValueError("book_id must match proposal")
        if self.proposal.config_revision != self.config_revision:
            raise ValueError("config_revision must match proposal")
        if self.status not in _APPROVAL_STATUSES:
            raise ValueError("unsupported approval status")
        self._validate_times()

    def _validate_times(self) -> None:
        _require_utc(self.created_at, "created_at")
        if self.decided_at is not None:
            _require_utc(self.decided_at, "decided_at")
            if self.decided_at < self.created_at:
                raise ValueError("decided_at must not precede created_at")
        if self.claimed_at is not None:
            _require_utc(self.claimed_at, "claimed_at")
            if self.claimed_at < self.created_at:
                raise ValueError("claimed_at must not precede created_at")
        if self.status == "pending" and (self.decided_at is not None or self.claimed_at is not None):
            raise ValueError("pending approval must not have decision or claim times")
        if self.status in {"approved", "rejected", "invalidated"} and (
            self.decided_at is None or self.claimed_at is not None
        ):
            raise ValueError(f"{self.status} approval requires only decided_at")
        if self.status == "executed":
            if self.decided_at is None or self.claimed_at is None:
                raise ValueError("executed approval requires decision and claim times")
            if self.claimed_at < self.decided_at:
                raise ValueError("claimed_at must not precede decided_at")

    @property
    def id(self) -> str:
        return self.approval_id
