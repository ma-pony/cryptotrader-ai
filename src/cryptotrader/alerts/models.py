"""Safe, strictly typed alert contracts. No raw venue/configuration payloads."""

from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from cryptotrader.pair import Pair

AlertType = Literal[
    "approval_pending",
    "execution_failed",
    "protection_failed",
    "risk_adjusted",
    "component_failed",
    "connection_failed",
    "daily_summary",
]
Resolution = Literal["open", "applied", "approval_processed", "expired", "recovered", "exit_completed", "informational"]


class BusinessAlertEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    event_key: str = Field(min_length=1, max_length=512)
    type: AlertType
    occurred_at: AwareDatetime
    capital_scope: Literal["simulated", "real"] | None = None
    decision_id: str | None = None
    book_id: str | None = None
    connection_id: str | None = None
    operation_id: str | None = None
    pair: str | None = None
    message: str = Field(min_length=1, max_length=512)

    @field_validator("pair")
    @classmethod
    def validate_pair(cls, value: str | None) -> str | None:
        return Pair.parse(value).canonical() if value is not None else None


class AlertOut(BusinessAlertEvent):
    id: str
    read_at: datetime | None
    resolution: Resolution
    resolved_at: datetime | None


class DeliveryOut(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    alert_id: str
    channel: Literal["webhook"]
    status: Literal["pending", "sending", "failed", "delivered"]
    attempts: int
    last_error: str | None
    last_attempt_at: datetime | None
    delivered_at: datetime | None
