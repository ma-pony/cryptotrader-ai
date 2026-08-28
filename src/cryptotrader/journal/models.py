"""一次 TradingCycle 的不可变审计记录。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from cryptotrader.decision.models import CycleStatus


@dataclass(frozen=True)
class TradingCycleRecord:
    cycle_id: str
    created_at: datetime
    pair: str
    status: CycleStatus
    profile_revision: int
    profile_snapshot: Mapping[str, Any]
    context_summary: Mapping[str, Any]
    component_signals: tuple[Mapping[str, Any], ...]
    component_error: Mapping[str, str] | None
    fused_signal: Mapping[str, Any] | None
    target_position: Mapping[str, Any] | None
    trade_plan: Mapping[str, Any] | None
    hitl_result: Mapping[str, Any] | None
    risk_result: Mapping[str, Any] | None
    execution_result: Mapping[str, Any] | None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.cycle_id:
            raise ValueError("cycle_id is required")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if not self.pair:
            raise ValueError("pair is required")
        if self.profile_revision < 1:
            raise ValueError("profile_revision must be positive")
        if self.profile_snapshot.get("revision") != self.profile_revision:
            raise ValueError("profile_snapshot revision must match profile_revision")
        available = self.context_summary.get("available")
        if available is True:
            required = {
                "pair",
                "as_of",
                "mode",
                "exchange_id",
                "market_type",
                "equity",
                "current_price",
                "atr",
                "current_position",
                "portfolio",
            }
        elif available is False:
            required = {"pair", "as_of", "mode", "exchange_id"}
        else:
            raise ValueError("context_summary must declare availability")
        missing = required - self.context_summary.keys()
        if missing:
            raise ValueError(f"context_summary missing fields: {sorted(missing)}")
