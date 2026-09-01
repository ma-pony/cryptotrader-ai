"""Durable pool-wide risk facts, with a high-water mark independent of config saves."""

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from cryptotrader.accounts.models import AccountSnapshot
from cryptotrader.accounts.store import Base, payload, snapshot_from_payload
from cryptotrader.db import get_async_session

ZERO = Decimal("0")


class BookRiskRow(Base):
    __tablename__ = "book_risk_states"
    book_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


def instrument_key(instrument, connection_id):
    pair = instrument.pair
    if pair is not None and pair.market_type == instrument.market_type:
        return pair.canonical()
    return f"{connection_id}:{instrument.market_type}:{instrument.venue_symbol}"


@dataclass(frozen=True)
class BookRiskState:
    book_id: str
    capital_scope: str
    valuation_currency: str
    observed_at: datetime
    equity: Decimal | None
    peak_equity: Decimal | None
    positions_by_instrument: dict[str, Decimal | None]
    pending_increase_notional: Decimal | None
    gross_notional: Decimal | None
    net_notional: Decimal | None
    used_margin: Decimal | None
    available_margin: Decimal | None
    completeness: tuple[str, ...]
    snapshots: tuple[AccountSnapshot, ...]

    @classmethod
    def from_snapshots(cls, book_id, snapshots, *, peak_equity=None):  # noqa: C901 - explicit independent fact reductions
        snapshots = tuple(snapshots)
        if not snapshots or len({item.connection_id for item in snapshots}) != len(snapshots):
            raise ValueError("risk requires unique complete book members")
        scopes = {item.capital_scope for item in snapshots}
        if len(scopes) != 1:
            raise ValueError("risk cannot combine real and simulated accounts")
        currencies = {item.equity.currency for item in snapshots}
        currency = next(iter(currencies)) if len(currencies) == 1 else "UNKNOWN"
        missing = [f"{s.connection_id}:{reason}" for s in snapshots for reason in s.completeness]

        def value(money, label):
            if money.currency != currency or money.amount is None:
                missing.append(f"{label}:valuation_unavailable:{money.currency}")
                return None
            return money.amount

        def total(values):
            return None if any(item is None for item in values) else sum(values, ZERO)

        equity = total([value(s.equity, f"{s.connection_id}:equity") for s in snapshots])
        used = total([value(s.used_margin, f"{s.connection_id}:used_margin") for s in snapshots])
        available = total([value(s.available_margin, f"{s.connection_id}:available_margin") for s in snapshots])
        positions, gross_values, pending = {}, [], []
        for account in snapshots:
            for position in account.positions:
                key = instrument_key(position.instrument, account.connection_id)
                amount = value(position.signed_notional, f"position:{key}")
                prior = positions.get(key, ZERO)
                positions[key] = None if amount is None or prior is None else prior + amount
                gross_values.append(None if amount is None else abs(amount))
            for order in account.orders:
                if order.status not in {"open", "partially_filled", "live", "New", "Untriggered"} or order.reduce_only:
                    continue
                pending.append(value(order.remaining_notional, f"order:{account.connection_id}:{order.venue_order_id}"))
        peak = peak_equity
        if equity is not None and equity > ZERO:
            peak = max(peak or ZERO, equity)
        return cls(
            book_id,
            next(iter(scopes)),
            currency,
            max(s.observed_at for s in snapshots),
            equity,
            peak,
            positions,
            total(pending),
            total(gross_values),
            total(list(positions.values())),
            used,
            available,
            tuple(dict.fromkeys(missing)),
            snapshots,
        )


def state_from_payload(value):
    decimal_fields = (
        "equity",
        "peak_equity",
        "pending_increase_notional",
        "gross_notional",
        "net_notional",
        "used_margin",
        "available_margin",
    )
    return BookRiskState(
        **(
            value
            | {
                **{key: None if value[key] is None else Decimal(value[key]) for key in decimal_fields},
                "observed_at": datetime.fromisoformat(value["observed_at"]),
                "positions_by_instrument": {
                    key: None if amount is None else Decimal(amount)
                    for key, amount in value["positions_by_instrument"].items()
                },
                "completeness": tuple(value["completeness"]),
                "snapshots": tuple(snapshot_from_payload(item) for item in value["snapshots"]),
            }
        )
    )


class BookRiskStateStore:
    def __init__(self, account_store):
        self.account_store = account_store

    async def get(self, book_id):
        await self.account_store.ensure_tables()
        async with await get_async_session(self.account_store.database_url) as session:
            row = await session.get(BookRiskRow, book_id)
            return state_from_payload(row.payload) if row else None

    async def update(self, book_id, snapshots):
        """Caller holds the book lease; persist only the exact account facts used by risk."""
        await self.account_store.ensure_tables()
        async with await get_async_session(self.account_store.database_url) as session, session.begin():
            row = await session.get(BookRiskRow, book_id, with_for_update=True)
            prior = state_from_payload(row.payload) if row else None
            state = BookRiskState.from_snapshots(book_id, snapshots, peak_equity=prior.peak_equity if prior else None)
            if prior and (
                prior.valuation_currency != state.valuation_currency
                or prior.capital_scope != state.capital_scope
                or "peak:valuation_basis_changed" in prior.completeness
            ):
                # A changed valuation basis cannot silently reset the high-water mark.
                state = replace(state, completeness=(*state.completeness, "peak:valuation_basis_changed"))
            if row is None:
                session.add(BookRiskRow(book_id=book_id, payload=payload(state)))
            else:
                row.payload = payload(state)
            return state
