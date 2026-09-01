# ruff: noqa: RUF001 -- Chinese user-facing messages use Chinese punctuation.
"""Research inputs, independent of HTTP and live execution configuration."""

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cryptotrader.backtest.cache import _TF_MS
from cryptotrader.pair import Pair

RunStatus = Literal["queued", "running", "completed", "failed", "canceled", "interrupted"]
TERMINAL = {"completed", "failed", "canceled", "interrupted"}


class BacktestParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: str
    end: str
    pair: str
    interval: str | None = "1h"
    initial_equity: Decimal | None = Field(default=Decimal("10000"), ge=100, allow_inf_nan=False)
    fee_rate: Decimal | None = Field(default=Decimal("0.001"), ge=0, lt=1, allow_inf_nan=False)
    slippage_bps: Decimal | None = Field(default=Decimal("0"), ge=0, lt=10000, allow_inf_nan=False)
    funding_assumption: Literal["available_only", "disabled"] | None = "available_only"
    name: str | None = Field(default=None, max_length=120)
    snapshot_run_id: str | None = None

    @model_validator(mode="after")
    def validate_experiment(self):
        start, end = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if start >= end or end > datetime.now(UTC).date():
            raise ValueError("开始日期须早于结束日期，结束日期不能晚于今天")
        if self.interval is not None and self.interval not in _TF_MS:
            raise ValueError("不支持的回测周期")
        Pair.parse(self.pair)
        return self
