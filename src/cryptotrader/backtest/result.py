"""Backtest result dataclass and statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from cryptotrader.accounts.models import Fill, FundingEntry

if TYPE_CHECKING:
    from datetime import datetime

    from cryptotrader.journal.models import MultiVenueCycleRecord


@dataclass(frozen=True)
class EquityPoint:
    time: datetime
    equity: Decimal


@dataclass(frozen=True)
class ClosedTrade:
    pair: str
    opened_at: datetime
    closed_at: datetime
    side: str
    gross_pnl: Decimal
    fees: Decimal
    funding: Decimal
    net_pnl: Decimal
    fill_ids: tuple[str, ...]


def closed_round_trips(fills: list[Fill], funding_entries=()) -> list[ClosedTrade]:
    """A round ends only at flat/reversal; split a reversal fee by actual size."""
    states = {}
    completed = []
    events = sorted([*funding_entries, *fills], key=lambda item: item.occurred_at)
    for fill in events:
        key = (fill.connection_id, fill.instrument.venue_symbol)
        if isinstance(fill, FundingEntry):
            if key in states:
                states[key][5] += fill.amount.amount
            continue
        delta = fill.amount if fill.side == "buy" else -fill.amount
        state = states.get(key)
        fee, gross = fill.fee.amount, fill.realized_pnl.amount
        if fee is None or gross is None:
            raise ValueError("Paper replay requires known fee and gross P&L")
        if state is None:
            states[key] = [delta, fill.occurred_at, fee, Decimal("0"), [fill.venue_fill_id], Decimal("0")]
            continue
        amount, opened_at, fees, pnl, ids, funding = state
        next_amount = amount + delta
        ids = [*ids, fill.venue_fill_id]
        if amount * delta < 0 and abs(delta) >= abs(amount):
            closing_fee = fee * abs(amount) / abs(delta)
            completed.append(
                ClosedTrade(
                    str(fill.instrument.pair),
                    opened_at,
                    fill.occurred_at,
                    "long" if amount > 0 else "short",
                    pnl + gross,
                    fees + closing_fee,
                    funding,
                    pnl + gross + funding - fees - closing_fee,
                    tuple(ids),
                )
            )
            if next_amount:
                states[key] = [
                    next_amount,
                    fill.occurred_at,
                    fee - closing_fee,
                    Decimal("0"),
                    [fill.venue_fill_id],
                    Decimal("0"),
                ]
            else:
                states.pop(key)
        else:
            states[key] = [next_amount, opened_at, fees + fee, pnl + gross, ids, funding]
    return completed


@dataclass
class BacktestResult:
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float | None = None
    fills: list[Fill] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    fees: Decimal = Decimal("0")
    funding: Decimal = Decimal("0")
    funding_entries: list[FundingEntry] = field(default_factory=list)
    cost_assumptions: dict = field(default_factory=dict)
    unmodeled_costs: list[str] = field(default_factory=list)
    data_coverage: dict = field(default_factory=dict)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    cycle_records: list[MultiVenueCycleRecord] = field(default_factory=list)
    cycle_ids: list[str] = field(default_factory=list)
    config_revisions: list[int] = field(default_factory=list)
    llm_calls: int = 0
    llm_tokens: int = 0

    @property
    def fill_count(self) -> int:
        return len(self.fills)

    @property
    def closed_trade_count(self) -> int:
        return len(self.closed_trades)

    def summary(self) -> dict:
        result = {
            "total_return": f"{self.total_return:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "win_rate": f"{self.win_rate:.2%}" if self.win_rate is not None else None,
            "fill_count": self.fill_count,
            "closed_trade_count": self.closed_trade_count,
            "fees": str(self.fees),
            "funding": str(self.funding),
        }
        if self.llm_calls > 0:
            result["llm_calls"] = self.llm_calls
            result["llm_tokens"] = self.llm_tokens
        return result

    def to_json(self, path: str) -> None:
        from cryptotrader.accounts.store import payload
        from cryptotrader.cycle_serialization import json_value

        values = {key: payload(getattr(self, key)) for key in self.__dataclass_fields__ if key != "cycle_records"}
        with open(path, "w") as f:
            json.dump(
                json_value({"summary": self.summary(), **values}),
                f,
                indent=2,
                default=str,
            )
