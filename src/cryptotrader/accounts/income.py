"""Moving-average realized income; no equity-difference or implicit FX returns."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from cryptotrader.accounts.models import Money


@dataclass(frozen=True)
class IncomeSummary:
    realized_gross: tuple[Money, ...]
    fees: tuple[Money, ...]
    funding: tuple[Money, ...]
    unrealized: tuple[Money, ...]
    net_trading: tuple[Money, ...]
    completeness: tuple[str, ...]
    methodology: str
    unrealized_as_of: datetime | None


def sum_money(items):
    totals, reasons = {}, {}
    for item in items:
        totals.setdefault(item.currency, Decimal("0"))
        if item.amount is None:
            reasons[item.currency] = item.unavailable_reason
        else:
            totals[item.currency] += item.amount
    return tuple(
        Money(None, currency, reasons[currency]) if currency in reasons else Money(amount, currency)
        for currency, amount in sorted(totals.items())
    )


def _realized_and_fees(fills, start, from_inception):
    costs, realized, fees = {}, [], []
    for fill in fills:
        pair = fill.instrument.pair
        currency = fill.realized_pnl.currency
        key = (fill.instrument.venue_symbol, fill.instrument.market_type)
        quantity, average = costs.get(key, (Decimal("0"), Decimal("0")))
        delta = fill.amount if fill.side == "buy" else -fill.amount
        calculated = Decimal("0")
        if quantity * delta < 0:
            calculated = min(abs(quantity), abs(delta)) * (fill.price - average) * (1 if quantity > 0 else -1)
        new_quantity = quantity + delta
        if quantity * delta >= 0:
            average = (abs(quantity) * average + abs(delta) * fill.price) / abs(new_quantity)
        elif quantity * new_quantity < 0:
            average = fill.price
        costs[key] = (new_quantity, average)
        if fill.occurred_at < start:
            continue
        value = fill.realized_pnl
        if value.amount is None and fill.source != "local_calculation":
            if (
                from_inception
                and pair
                and fill.instrument.market_type in {"spot", "swap"}
                and (pair.settle or pair.quote) == pair.quote
            ):
                value = Money(calculated, pair.quote)
            else:
                value = Money(None, currency, "期初持仓成本或结算口径无法核对")
        realized.append(value)
        fees.append(fill.fee)
    return realized, fees


class IncomeService:
    def __init__(self, store):
        self.store = store

    async def summary(self, connection_id: str, start: datetime, end: datetime, *, symbol=None):
        if start > end or start.tzinfo is None or end.tzinfo is None:
            raise ValueError("income requires an ordered UTC-aware range")
        status = await self.store.status(connection_id)
        coverage = status["coverage"]
        gaps = []
        for kind in ("fills", "funding"):
            window = coverage.get(kind, {})
            if (
                not window.get("complete")
                or not window.get("coverage_start")
                or not window.get("coverage_end")
                or datetime.fromisoformat(window["coverage_start"]) > start
                or datetime.fromisoformat(window["coverage_end"]) < end
            ):
                gaps.append(f"{'成交' if kind == 'fills' else '资金费'}历史未覆盖查询区间")
        fills = await self.store.history(connection_id, end=end, symbol=symbol)
        funding = await self.store.history(connection_id, "funding", start=start, end=end, symbol=symbol)
        from_inception = coverage.get("fills", {}).get("from_inception", False)
        realized, fees = _realized_and_fees(fills, start, from_inception)
        snapshot = await self.store.latest(connection_id)
        default_currency = snapshot.equity.currency if snapshot else "UNKNOWN"
        gross = sum_money(realized) or (Money(Decimal("0"), default_currency),)
        expenses = sum_money(fees) or (Money(Decimal("0"), default_currency),)
        funding_values = sum_money(entry.amount for entry in funding) or (Money(Decimal("0"), default_currency),)
        if gaps:
            gross = tuple(Money(None, item.currency, "成交历史覆盖不完整") for item in gross)
            expenses = tuple(Money(None, item.currency, "手续费历史覆盖不完整") for item in expenses)
            funding_values = tuple(Money(None, item.currency, "资金费历史覆盖不完整") for item in funding_values)
        net = sum_money(
            (
                *gross,
                *(Money(-item.amount, item.currency) if item.amount is not None else item for item in expenses),
                *funding_values,
            )
        )
        unrealized = (
            sum_money(
                position.unrealized_pnl
                for position in snapshot.positions
                if symbol is None or symbol in (position.instrument.venue_symbol, str(position.instrument.pair))
            )
            if snapshot
            else ()
        )
        if not unrealized:
            unrealized = (Money(None, default_currency, "缺少当前估值"),)
        if any(item.amount is None for item in (*gross, *expenses, *funding_values)):
            gaps.append("部分已实现金额无法核对")
        return IncomeSummary(
            gross,
            expenses,
            funding_values,
            unrealized,
            net,
            tuple(gaps),
            "已实现收益采用平台明确结算收益，缺失时仅在期初成本可证明的同币种持仓上使用移动平均成本；手续费正支出、资金费正收入；净交易收益不含未实现盈亏，不做币种换算。",  # noqa: RUF001
            snapshot.observed_at if snapshot and any(item.amount is not None for item in unrealized) else None,
        )
