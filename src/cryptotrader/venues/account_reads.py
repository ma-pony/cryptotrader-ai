"""Shared read-only normalization and bounded, persistable history cursors."""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

from cryptotrader.accounts.models import Instrument, Money
from cryptotrader.pair import Pair
from cryptotrader.venues.protocol import VenueOperationError

DAY_MS = 86_400_000


def milliseconds() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def timestamp(value) -> datetime:
    try:
        return datetime.fromtimestamp(int(value) / 1000, UTC)
    except (ValueError, TypeError, OverflowError):
        raise VenueOperationError("invalid account timestamp") from None


class AccountReadMixin:
    """Uses the session's credential-safe call boundary, never execution reads."""

    def _remaining_notional(self, instrument, amount, filled, price, *, explicit_value=None):
        currency = self._settlement(instrument) or "UNKNOWN"
        market = self._client.markets.get(instrument.pair.to_ccxt(), {}) if instrument.pair else {}
        if instrument.pair is None or market.get("inverse") or instrument.market_type == "option":
            return Money(None, currency, "remaining_order_valuation_unavailable")
        if explicit_value not in (None, ""):
            return self._money(explicit_value, currency, "remaining_order_valuation_unavailable")
        quote = self._optional_decimal(price)
        if quote is None or quote <= 0:
            return Money(None, currency, "remaining_order_price_unavailable")
        remaining = max(Decimal("0"), self._read_amount(amount, instrument) - self._read_amount(filled, instrument))
        return Money(remaining * quote, currency)

    async def list_instruments(self) -> tuple[Instrument, ...]:
        await self._ensure_markets()
        return tuple(self._instrument_from_market(market) for market in self._client.markets.values())

    def _instrument_from_market(self, market) -> Instrument:
        market_type = market.get("type") or next(
            (kind for kind in ("spot", "swap", "future", "option") if market.get(kind)), "unknown"
        )
        symbol = str(market.get("id") or market.get("symbol") or "UNKNOWN")
        pair = None
        if market_type != "option":
            with suppress(ValueError):
                pair = Pair.parse(market.get("symbol") or "")
        reason = None
        if pair is None:
            reason = "unmapped_venue_instrument:amount_in_venue_units"
        elif market_type not in self.capabilities.market_types or market.get("inverse"):
            reason = "execution_market_unsupported:amount_in_venue_units"
        elif market.get("active") is False:
            reason = "instrument_inactive"
        return Instrument(symbol, pair, market_type, reason is None, reason)

    def _read_instrument(self, symbol: str, category: str = "") -> Instrument:
        for market in self._client.markets.values():
            if symbol not in {market.get("id"), market.get("symbol")}:
                continue
            if category.lower() in {"spot", "margin"} and not market.get("spot"):
                continue
            if category.lower() in {"linear", "inverse", "swap", "futures", "option"} and market.get("spot"):
                continue
            instrument = self._instrument_from_market(market)
            if category.lower() == "margin":
                return replace(instrument, market_type="margin", tradable=False, reason="margin_execution_unsupported")
            return instrument
        return Instrument(
            str(symbol or "UNKNOWN"),
            None,
            category.lower() or "unknown",
            False,
            "unmapped_venue_instrument:amount_in_venue_units",
        )

    def _read_amount(self, value, instrument: Instrument) -> Decimal:
        amount = self._decimal(value, "account amount")
        pair = instrument.pair
        if pair is not None and pair.market_type != "spot":
            market = self._client.markets.get(pair.to_ccxt(), {})
            if market.get("linear"):
                amount *= self._decimal(market.get("contractSize"), "account contract size")
        return amount

    def _optional_decimal(self, value) -> Decimal | None:
        return None if value in (None, "") else self._decimal(value, "account number")

    def _money(self, value, currency: str | None, reason: str) -> Money:
        if not currency:
            return Money(None, "UNKNOWN", f"{reason}:currency_unavailable")
        amount = self._optional_decimal(value)
        return Money(amount, currency, reason if amount is None else None)

    @staticmethod
    def _settlement(instrument: Instrument) -> str | None:
        return (instrument.pair.settle or instrument.pair.quote) if instrument.pair else None

    @staticmethod
    def _signed_money(money: Money, sign: Decimal) -> Money:
        return replace(money, amount=abs(money.amount) * sign) if money.amount is not None else money

    def _history_state(self, cursor: str | None, kind: str, count: int, retention_days: int) -> dict:
        now = milliseconds()
        if cursor is None:
            state = {
                "connection_id": self.connection_id,
                "kind": kind,
                "begin": now - 7 * DAY_MS,
                "end": now,
                "scope": 0,
                "after": None,
                "complete": False,
            }
        else:
            try:
                state = json.loads(cursor)
                if (
                    state["connection_id"] != self.connection_id
                    or state["kind"] != kind
                    or type(state["begin"]) is not int
                    or type(state["end"]) is not int
                    or not 0 <= state["begin"] <= state["end"] <= now
                    or state["end"] - state["begin"] > 7 * DAY_MS
                    or type(state["scope"]) is not int
                    or not 0 <= state["scope"] < count
                    or type(state["complete"]) is not bool
                    or (state["after"] is not None and type(state["after"]) is not str)
                ):
                    raise ValueError
                if state["complete"]:
                    state.update(
                        begin=state["end"], end=min(now, state["end"] + 7 * DAY_MS), scope=0, after=None, complete=False
                    )
            except (ValueError, KeyError, TypeError):
                raise VenueOperationError("invalid account history cursor") from None
        if state["begin"] < now - retention_days * DAY_MS:
            raise VenueOperationError("account history gap exceeds platform retention", code="history_gap")
        return state

    @staticmethod
    def _history_page(page_type, items, state: dict, next_token: str | None, count: int):
        following = dict(state)
        if next_token:
            if next_token == state["after"]:
                raise VenueOperationError("account history cursor did not advance")
            following["after"] = next_token
        elif state["scope"] + 1 < count:
            following.update(scope=state["scope"] + 1, after=None)
        else:
            following.update(complete=True, after=None)
        return page_type(
            tuple(items),
            json.dumps(following, separators=(",", ":")),
            following["complete"],
            timestamp(state["begin"]),
            timestamp(state["end"]),
        )

    @staticmethod
    def _completeness(equity, used, available, positions) -> tuple[str, ...]:
        missing = [
            f"{key}:{money.unavailable_reason}"
            for key, money in (("equity", equity), ("used_margin", used), ("available_margin", available))
            if money.amount is None
        ]
        for position in positions:
            symbol = position.instrument.venue_symbol
            if position.available_amount is None:
                missing.append(f"position:{symbol}:available_amount_unavailable")
            for key, money in (("notional", position.signed_notional), ("unrealized_pnl", position.unrealized_pnl)):
                if money.amount is None:
                    missing.append(f"position:{symbol}:{key}:{money.unavailable_reason}")
        return tuple(missing)
