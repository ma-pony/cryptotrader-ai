"""Deterministic in-memory Paper implementation of the venue contracts."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from cryptotrader.accounts.models import (
    AccountOrder,
    AccountPosition,
    AccountSnapshot,
    Fill,
    FillPage,
    FundingEntry,
    FundingPage,
    Instrument,
    Money,
)
from cryptotrader.configuration.parameters import PaperParameters
from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
from cryptotrader.venues.ccxt_base import VenueOperationError
from cryptotrader.venues.models import (
    ACCOUNT_READS,
    EXIT_OPERATIONS,
    BacktestCostModel,
    ConnectionPosition,
    NormalizedOrder,
    OpenVenueState,
    OrderIntent,
    ProtectionSpec,
    ProtectionState,
    VenueCapabilities,
    VenueQuote,
)

if TYPE_CHECKING:
    from cryptotrader.pair import Pair
    from cryptotrader.runtime_config.secrets import CredentialPayload
    from cryptotrader.venues.models import VenueConnection

_SUPPORTED_MARKET_TYPES = frozenset({"spot", "swap"})


@dataclass
class _PaperPosition:
    signed_amount: Decimal = Decimal("0")
    entry_price: Decimal | None = None


@dataclass
class _PaperAccount:
    initial_equity: Decimal
    leverage: int
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    balances: dict[str, Decimal] = field(init=False)
    quotes: dict[Pair, VenueQuote] = field(default_factory=dict)
    quote_times: dict[Pair, datetime] = field(default_factory=dict)
    positions: dict[Pair, _PaperPosition] = field(default_factory=dict)
    spot_entry_prices: dict[Pair, Decimal] = field(default_factory=dict)
    orders: dict[str, NormalizedOrder] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    funding: list[FundingEntry] = field(default_factory=list)
    protections: dict[Pair, ProtectionState] = field(default_factory=dict)
    triggered_protections: dict[Pair, ProtectionState] = field(default_factory=dict)
    protection_pairs: dict[str, Pair] = field(default_factory=dict)
    pair_locks: dict[Pair, asyncio.Lock] = field(default_factory=dict)
    order_sequence: int = 0
    protection_sequence: int = 0
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.balances = {"USDT": self.initial_equity}

    def lock_for(self, pair: Pair) -> asyncio.Lock:
        return self.state_lock


class PaperVenueSession:
    """One session over connection-local deterministic Paper account state."""

    def __init__(
        self,
        connection: VenueConnection,
        account: _PaperAccount,
        capabilities: VenueCapabilities,
        store=None,
        *,
        clock=None,
        cost_model=None,
    ) -> None:
        self.connection_id = connection.id
        self.connection = connection
        self._account = account
        self._capabilities = capabilities
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._store = store
        self._clock = clock or (lambda: datetime.now(UTC))
        self.cost_model = cost_model or BacktestCostModel(Decimal("0"), Decimal("0"), False)

    @property
    def history_from_inception(self) -> bool:
        return True

    @asynccontextmanager
    async def _mutation(self, pair):
        async with self._account.lock_for(pair):
            previous = _paper_payload(self._account) if self._store else None
            try:
                yield
                if self._store:
                    await self._store.save_paper(self.connection_id, _paper_payload(self._account))
            except BaseException:
                if previous is not None:
                    restored = _paper_from_payload(previous)
                    for key in previous:
                        setattr(self._account, key, getattr(restored, key))
                raise

    @property
    def capabilities(self) -> VenueCapabilities:
        return self._capabilities

    async def check_connection(self) -> None:
        """Validate the deterministic local account without requiring market data."""
        self._require_open()
        if (
            not self._account.initial_equity.is_finite()
            or self._account.initial_equity <= 0
            or not self._account.balances
            or any(not balance.is_finite() for balance in self._account.balances.values())
        ):
            raise VenueOperationError(f"{self.connection_id}: invalid Paper account")

    @staticmethod
    def _instrument(pair: Pair) -> Instrument:
        return Instrument(pair.to_ccxt(), pair, pair.market_type, True)

    async def list_instruments(self) -> tuple[Instrument, ...]:
        self._require_open()
        return tuple(self._instrument(pair) for pair in self._account.quotes)

    async def fetch_account(self) -> AccountSnapshot:
        """A synchronous copy between awaits: never advance fills or bankruptcy."""
        async with self._account.state_lock:
            return self._account_snapshot()

    def _account_snapshot(self) -> AccountSnapshot:
        self._require_open()
        now = self._clock()
        positions = []
        for pair in (*self._account.positions, *self._account.spot_entry_prices):
            position = self._position_dto_locked(pair)
            currency = pair.settle or pair.quote
            pnl = (
                Money((self._quote_locked(pair).last - position.entry_price) * position.signed_amount, currency)
                if position.entry_price is not None
                else Money(None, currency, "entry_price_unavailable")
            )
            positions.append(
                AccountPosition(
                    self._instrument(pair),
                    position.signed_amount,
                    abs(position.signed_amount),
                    Money(position.signed_notional, currency),
                    position.entry_price,
                    pnl,
                )
            )
        orders = [
            AccountOrder(
                self.connection_id,
                order.id,
                self._instrument(order.pair),
                order.side,
                order.order_type,
                order.amount,
                order.filled_amount,
                order.average_price,
                order.status,
                order.reduce_only,
                False,
                order.client_order_id,
                now,
                Money(
                    max(Decimal("0"), order.amount - order.filled_amount) * self._quote_locked(order.pair).last,
                    order.pair.settle or order.pair.quote,
                ),
            )
            for order in self._account.orders.values()
            if order.status == "open"
        ]
        for protection in self._account.protections.values():
            orders.extend(
                AccountOrder(
                    self.connection_id,
                    order_id,
                    self._instrument(protection.pair),
                    "sell" if protection.position_side == "long" else "buy",
                    "oco",
                    protection.amount,
                    Decimal("0"),
                    None,
                    "open",
                    True,
                    True,
                    None,
                    now,
                    Money(
                        protection.amount * self._quote_locked(protection.pair).last,
                        protection.pair.settle or protection.pair.quote,
                    ),
                )
                for order_id in protection.protection_ids
            )
        missing = [
            f"valuation:{asset}:USDT_quote_unavailable"
            for asset, amount in self._account.balances.items()
            if asset != "USDT"
            and amount
            and not any(
                pair.market_type == "spot" and pair.base == asset and pair.quote == "USDT"
                for pair in self._account.quotes
            )
        ]
        missing.extend(
            f"valuation:{pair}:non_USDT_settlement"
            for pair in self._account.positions
            if (pair.settle or pair.quote) != "USDT"
        )
        if missing:
            equity = Money(None, "USDT", ";".join(missing))
            used = available = Money(None, "USDT", "account_valuation_incomplete")
        else:
            equity = Money(self._equity_locked(), "USDT")
            margin = sum(
                (
                    abs(position.signed_amount * self._quote_locked(pair).last) / self._account.leverage
                    for pair, position in self._account.positions.items()
                ),
                Decimal("0"),
            )
            used = Money(margin, "USDT")
            available = Money(equity.amount - margin, "USDT")
        return AccountSnapshot(
            self.connection_id,
            now,
            "simulated",
            equity,
            tuple(Money(amount, asset) for asset, amount in self._account.balances.items()),
            tuple(positions),
            tuple(orders),
            used,
            available,
            tuple(missing),
            tuple(
                f"保存报价 {pair}：{self._account.quote_times[pair].isoformat()}，不是本次同步的新行情"  # noqa: RUF001
                for pair in self._account.quotes
                if pair in self._account.quote_times
            ),
        )

    async def fetch_fills(self, cursor: str | None) -> FillPage:
        async with self._account.state_lock:
            return self._fill_page(cursor)

    def _fill_page(self, cursor: str | None) -> FillPage:
        self._require_open()
        start, end, checkpoint = self._history_window(cursor, "fills")
        return FillPage(
            tuple(fill for fill in self._account.fills if start <= fill.occurred_at <= end),
            checkpoint,
            True,
            start,
            end,
        )

    async def fetch_funding(self, cursor: str | None) -> FundingPage:
        self._require_open()
        start, end, checkpoint = self._history_window(cursor, "funding")
        return FundingPage(
            tuple(item for item in self._account.funding if start <= item.occurred_at <= end),
            checkpoint,
            True,
            start,
            end,
        )

    async def apply_funding(self, pair: Pair, *, settlement_id: str, rate: Decimal, mark_price: Decimal) -> None:
        """Apply one proven settlement at the injected clock to the held position."""
        self._require_open()
        self._require_supported_pair(pair)
        if not self.cost_model.funding_enabled or pair.market_type == "spot":
            return
        if not rate.is_finite() or not mark_price.is_finite() or mark_price <= 0:
            raise ValueError("invalid funding settlement")
        async with self._mutation(pair):
            if any(item.venue_entry_id == settlement_id for item in self._account.funding):
                return
            position = self._account.positions.get(pair, _PaperPosition())
            amount = -position.signed_amount * mark_price * rate
            currency = pair.settle or pair.quote
            self._account.balances[currency] = self._account.balances.get(currency, Decimal("0")) + amount
            self._account.funding.append(
                FundingEntry(
                    self.connection_id,
                    settlement_id,
                    self._instrument(pair),
                    Money(amount, currency),
                    self._clock(),
                )
            )

    def _history_window(self, cursor: str | None, kind: str) -> tuple[datetime, datetime, str]:
        now = self._clock()
        start = self._account.started_at
        if cursor is not None:
            try:
                state = json.loads(cursor)
                previous_start = datetime.fromisoformat(state["begin"])
                previous_end = datetime.fromisoformat(state["end"])
                if (
                    state["connection_id"] != self.connection_id
                    or state["kind"] != kind
                    or state["complete"] is not True
                    or previous_start.utcoffset() != timedelta(0)
                    or previous_end.utcoffset() != timedelta(0)
                    or not self._account.started_at <= previous_start <= previous_end <= now
                    or previous_end - previous_start > timedelta(days=7)
                ):
                    raise ValueError
                start = previous_end
            except (ValueError, TypeError, KeyError):
                raise VenueOperationError("invalid Paper history cursor") from None
        end = min(now, start + timedelta(days=7))
        checkpoint = json.dumps(
            {
                "connection_id": self.connection_id,
                "kind": kind,
                "begin": start.isoformat(),
                "end": end.isoformat(),
                "complete": True,
            },
            separators=(",", ":"),
        )
        return start, end, checkpoint

    async def set_quote(self, pair: Pair, price: Decimal) -> VenueQuote:
        """Install the latest deterministic Paper quote for one pair."""
        self._require_open()
        self._require_supported_pair(pair)
        if not isinstance(price, Decimal) or not price.is_finite() or price <= 0:
            raise ValueError("Paper quote price must be a positive finite Decimal")
        async with self._mutation(pair):
            quote = VenueQuote(pair, price, price, price)
            self._account.quotes[pair] = quote
            self._account.quote_times[pair] = self._clock()
            self._apply_bankruptcy_locked()
            return quote

    async def fetch_quote(self, pair: Pair) -> VenueQuote:
        self._require_open()
        self._require_supported_pair(pair)
        async with self._account.lock_for(pair):
            return self._quote_locked(pair)

    async def advance_bar(self, pair: Pair, bar) -> None:
        """Advance an already closed bar; only protection existing at entry participates.

        Stop wins an ambiguous bar. A stop gap fills at the worse opening price.
        The caller generates signals only after this operation has finished.
        """
        self._require_open()
        self._require_supported_pair(pair)
        async with self._mutation(pair):
            self._account.quotes[pair] = VenueQuote(pair, bar.open, bar.open, bar.open)
            protection = self._account.protections.get(pair)
            trigger = None
            if protection is not None:
                stop, take = protection.stop_loss, protection.take_profit
                if protection.position_side == "long":
                    if stop is not None and bar.low <= stop:
                        trigger = min(bar.open, stop)
                    elif take is not None and bar.high >= take:
                        trigger = take
                elif stop is not None and bar.high >= stop:
                    trigger = max(bar.open, stop)
                elif take is not None and bar.low <= take:
                    trigger = take
            if trigger is not None:
                triggered = self._trigger_protection_locked(pair, trigger_price=trigger)
                if triggered is not None:
                    self._account.triggered_protections[pair] = triggered
            self._account.quotes[pair] = VenueQuote(pair, bar.close, bar.close, bar.close)
            self._account.quote_times[pair] = self._clock()
            self._apply_bankruptcy_locked()

    async def normalize_amount(self, pair: Pair, base_amount: Decimal) -> Decimal:
        self._require_open()
        self._require_supported_pair(pair)
        if not isinstance(base_amount, Decimal) or not base_amount.is_finite() or base_amount <= 0:
            raise ValueError("Paper base amount must be a positive finite Decimal")
        return base_amount

    async def minimum_amount(self, pair: Pair, reference_price: Decimal, minimum_quote_notional: Decimal) -> Decimal:
        self._require_open()
        self._require_supported_pair(pair)
        if not isinstance(reference_price, Decimal) or reference_price <= 0:
            raise ValueError("Paper reference price must be positive")
        if not isinstance(minimum_quote_notional, Decimal) or minimum_quote_notional <= 0:
            raise ValueError("Paper minimum quote notional must be positive")
        return minimum_quote_notional / reference_price

    async def place_order(self, intent: OrderIntent) -> NormalizedOrder:
        self._require_open()
        self._require_supported_pair(intent.pair)
        async with self._mutation(intent.pair):
            quote = self._quote_locked(intent.pair)
            self._apply_bankruptcy_locked()
            if intent.order_type == "market":
                fill_price = quote.last
            elif intent.order_type == "limit" and intent.price is not None:
                fill_price = intent.price
            else:
                raise VenueOperationError(f"{self.connection_id}: unsupported Paper order type")
            return self._place_order_locked(intent, fill_price)

    async def cancel_order(self, order_id: str, pair: Pair) -> None:
        self._require_open()
        self._require_supported_pair(pair)
        async with self._mutation(pair):
            order = self._account.orders.get(order_id)
            if order is None or order.pair != pair:
                return
            if order.status == "open":
                self._account.orders[order_id] = replace(order, status="canceled")

    async def find_order(
        self, pair: Pair, *, order_id: str | None = None, client_order_id: str | None = None
    ) -> NormalizedOrder | None:
        self._require_open()
        self._require_supported_pair(pair)
        if bool(order_id) == bool(client_order_id):
            raise VenueOperationError(f"{self.connection_id}: provide exactly one order identifier")
        async with self._account.lock_for(pair):
            for order in self._account.orders.values():
                if order.pair != pair:
                    continue
                if order_id == order.id or client_order_id == order.client_order_id:
                    return order
        return None

    async def fetch_portfolio(self, pair: Pair) -> ConnectionPortfolioSnapshot:
        self._require_open()
        self._require_supported_pair(pair)
        async with self._mutation(pair):
            self._quote_locked(pair)
            bankrupt = self._apply_bankruptcy_locked()
            if not bankrupt:
                triggered = self._trigger_protection_locked(pair)
                if triggered is not None:
                    self._account.triggered_protections[pair] = triggered
            return self._portfolio_locked(pair)

    async def replace_protection(self, spec: ProtectionSpec) -> ProtectionState:
        self._require_open()
        self._require_supported_pair(spec.pair)
        async with self._mutation(spec.pair):
            quote = self._quote_locked(spec.pair)
            self._apply_bankruptcy_locked()
            position = self._position_dto_locked(spec.pair)
            expected_side = "long" if position.signed_amount > 0 else "short" if position.signed_amount < 0 else None
            if expected_side != spec.position_side or spec.amount > abs(position.signed_amount):
                raise VenueOperationError(f"{self.connection_id}: protection does not match the open position")
            try:
                spec.validate_geometry(quote.last)
            except ValueError:
                raise VenueOperationError(f"{self.connection_id}: invalid protection price geometry") from None

            self._account.protection_sequence += 1
            protection_id = f"{self.connection_id}-paper-protection-{self._account.protection_sequence}"
            prior = self._account.protections.get(spec.pair)
            if prior is not None:
                for prior_id in prior.protection_ids:
                    self._account.protection_pairs.pop(prior_id, None)
            protection = ProtectionState(
                (protection_id,),
                spec.pair,
                spec.position_side,
                spec.amount,
                spec.stop_loss,
                spec.take_profit,
                True,
                False,
                (protection_id,),
            )
            self._account.protections[spec.pair] = protection
            self._account.protection_pairs[protection_id] = spec.pair
            return protection

    async def normalize_protection(self, spec: ProtectionSpec) -> ProtectionSpec:
        self._require_open()
        self._require_supported_pair(spec.pair)
        return spec

    async def cancel_protection(self, protection_ids: tuple[str, ...]) -> None:
        self._require_open()
        pairs = {
            pair
            for protection_id in protection_ids
            if (pair := self._account.protection_pairs.get(protection_id)) is not None
        }
        for pair in sorted(pairs, key=lambda item: item.canonical()):
            async with self._mutation(pair):
                current = self._account.protections.get(pair)
                if current is None or not set(current.protection_ids).intersection(protection_ids):
                    continue
                self._account.protections.pop(pair, None)
                for protection_id in current.protection_ids:
                    self._account.protection_pairs.pop(protection_id, None)

    async def list_open_state(self, pair: Pair) -> OpenVenueState:
        self._require_open()
        self._require_supported_pair(pair)
        async with self._mutation(pair):
            self._quote_locked(pair)
            bankrupt = self._apply_bankruptcy_locked()
            triggered = None if bankrupt else self._account.triggered_protections.pop(pair, None)
            if triggered is None and not bankrupt:
                triggered = self._trigger_protection_locked(pair)
            position = self._position_dto_locked(pair)
            open_orders = tuple(
                order for order in self._account.orders.values() if order.pair == pair and order.status == "open"
            )
            if triggered is not None:
                protections = (triggered,)
            else:
                current = self._account.protections.get(pair)
                protections = (current,) if current is not None else ()
            return OpenVenueState(position, open_orders, protections)

    async def close(self) -> None:
        async with self._close_lock:
            self._closed = True

    def _place_order_locked(
        self, intent: OrderIntent, fill_price: Decimal, *, reserved_order_id: str | None = None
    ) -> NormalizedOrder:
        sign = Decimal("1") if intent.side == "buy" else Decimal("-1")
        fill_price *= Decimal("1") + sign * self.cost_model.slippage_bps / Decimal("10000")
        fee = abs(intent.amount * fill_price) * self.cost_model.fee_rate
        if reserved_order_id is None:
            self._account.order_sequence += 1
            order_id = f"{self.connection_id}-paper-order-{self._account.order_sequence}"
        else:
            # The execution ledger binds this actual ID when protection is created.
            order_id = reserved_order_id
        prior = self._position_dto_locked(intent.pair)
        delta = intent.amount if intent.side == "buy" else -intent.amount
        realized = Decimal("0")
        if prior.signed_amount * delta < 0:
            realized = (
                (fill_price - prior.entry_price)
                * min(abs(prior.signed_amount), abs(delta))
                * (1 if prior.signed_amount > 0 else -1)
                if prior.entry_price is not None
                else None
            )
        if intent.pair.market_type == "spot":
            accepted = self._fill_spot_locked(intent, fill_price, fee)
        else:
            accepted = self._fill_derivative_locked(intent, fill_price, fee)
        order = NormalizedOrder(
            order_id,
            intent.pair,
            intent.side,
            intent.order_type,
            intent.amount,
            intent.amount if accepted else Decimal("0"),
            fill_price if accepted else None,
            "filled" if accepted else "rejected",
            intent.reduce_only,
            intent.client_order_id,
        )
        self._account.orders[order_id] = order
        if accepted:
            currency = intent.pair.settle or intent.pair.quote
            self._account.balances[currency] = self._account.balances.get(currency, Decimal("0")) - fee
            self._account.fills.append(
                Fill(
                    self.connection_id,
                    f"{order_id}-fill",
                    order_id,
                    self._instrument(intent.pair),
                    intent.side,
                    intent.amount,
                    fill_price,
                    self._clock(),
                    Money(fee, currency),
                    Money(
                        realized,
                        currency,
                        "closing_cost_unavailable_in_settlement_currency" if realized is None else None,
                    ),
                    "local_calculation",
                    intent.client_order_id,
                )
            )
        return order

    def _fill_spot_locked(self, intent: OrderIntent, fill_price: Decimal, fee: Decimal) -> bool:
        quote_asset = intent.pair.quote
        base_asset = intent.pair.base
        quote_balance = self._account.balances.get(quote_asset, Decimal("0"))
        base_balance = self._account.balances.get(base_asset, Decimal("0"))
        cost = intent.amount * fill_price
        if intent.side == "buy":
            if intent.reduce_only or quote_balance < cost + fee:
                return False
            prior_cost = base_balance * self._account.spot_entry_prices.get(intent.pair, fill_price)
            next_amount = base_balance + intent.amount
            self._account.balances[quote_asset] = quote_balance - cost
            self._account.balances[base_asset] = next_amount
            self._account.spot_entry_prices[intent.pair] = (prior_cost + cost) / next_amount
            return True
        if base_balance < intent.amount:
            return False
        next_amount = base_balance - intent.amount
        self._account.balances[quote_asset] = quote_balance + cost
        self._account.balances[base_asset] = next_amount
        if next_amount == 0:
            self._account.spot_entry_prices.pop(intent.pair, None)
        return True

    def _fill_derivative_locked(self, intent: OrderIntent, fill_price: Decimal, fee: Decimal) -> bool:
        current = self._account.positions.get(intent.pair, _PaperPosition())
        delta = intent.amount if intent.side == "buy" else -intent.amount
        if intent.reduce_only and (
            current.signed_amount == 0 or current.signed_amount * delta >= 0 or abs(delta) > abs(current.signed_amount)
        ):
            return False
        next_amount = current.signed_amount + delta
        opens_opposite_leg = current.signed_amount * next_amount < 0
        realized = Decimal("0")
        if current.signed_amount * delta < 0 and current.entry_price is not None:
            closed_amount = min(abs(current.signed_amount), abs(delta))
            direction = Decimal("1") if current.signed_amount > 0 else Decimal("-1")
            realized = (fill_price - current.entry_price) * closed_amount * direction
        if opens_opposite_leg or abs(next_amount) > abs(current.signed_amount):
            mark_price = self._quote_locked(intent.pair).last
            margin_price = mark_price if opens_opposite_leg else fill_price
            required_margin = self._required_margin_locked(intent.pair, next_amount, margin_price)
            available_equity = self._equity_locked() - fee
            if opens_opposite_leg:
                current_unrealized = (
                    (mark_price - current.entry_price) * current.signed_amount
                    if current.entry_price is not None
                    else Decimal("0")
                )
                next_unrealized = (mark_price - fill_price) * next_amount
                available_equity = available_equity - current_unrealized + realized + next_unrealized
            if required_margin > available_equity:
                return False

        settlement = intent.pair.settle or intent.pair.quote
        self._account.balances[settlement] = self._account.balances.get(settlement, Decimal("0")) + realized

        if next_amount == 0:
            self._account.positions.pop(intent.pair, None)
            return True
        if current.signed_amount == 0 or current.signed_amount * next_amount < 0:
            entry_price = fill_price
        elif current.signed_amount * delta > 0:
            entry_price = (
                abs(current.signed_amount) * (current.entry_price or fill_price) + abs(delta) * fill_price
            ) / abs(next_amount)
        else:
            entry_price = current.entry_price
        self._account.positions[intent.pair] = _PaperPosition(next_amount, entry_price)
        return True

    def _required_margin_locked(self, changed_pair: Pair, next_amount: Decimal, fill_price: Decimal) -> Decimal:
        total = Decimal("0")
        pairs = set(self._account.positions) | {changed_pair}
        for pair in pairs:
            amount = next_amount if pair == changed_pair else self._account.positions[pair].signed_amount
            if amount == 0:
                continue
            price = fill_price if pair == changed_pair else self._quote_locked(pair).last
            total += abs(amount * price) / Decimal(self._account.leverage)
        return total

    def _portfolio_locked(self, pair: Pair) -> ConnectionPortfolioSnapshot:
        return ConnectionPortfolioSnapshot(
            self.connection_id,
            self._equity_locked(),
            {asset: amount for asset, amount in self._account.balances.items() if amount != 0},
            self._position_dto_locked(pair),
        )

    def _position_dto_locked(self, pair: Pair) -> ConnectionPosition:
        quote = self._quote_locked(pair)
        if pair.market_type == "spot":
            signed_amount = self._account.balances.get(pair.base, Decimal("0"))
            entry_price = self._account.spot_entry_prices.get(pair) if signed_amount != 0 else None
        else:
            position = self._account.positions.get(pair, _PaperPosition())
            signed_amount = position.signed_amount
            entry_price = position.entry_price
        return ConnectionPosition(pair, signed_amount, signed_amount * quote.last, entry_price)

    def _equity_locked(self) -> Decimal:
        equity = self._account.balances.get("USDT", Decimal("0"))
        for asset, amount in self._account.balances.items():
            if asset == "USDT" or amount == 0:
                continue
            pair = next(
                (
                    candidate
                    for candidate in self._account.quotes
                    if candidate.market_type == "spot" and candidate.base == asset and candidate.quote == "USDT"
                ),
                None,
            )
            if pair is not None:
                equity += amount * self._account.quotes[pair].last
        for pair, position in self._account.positions.items():
            if position.entry_price is not None:
                equity += (self._quote_locked(pair).last - position.entry_price) * position.signed_amount
        return equity

    def _apply_bankruptcy_locked(self) -> bool:
        if self._equity_locked() > 0:
            return False
        self._account.balances = {"USDT": Decimal("0")}
        self._account.positions.clear()
        self._account.spot_entry_prices.clear()
        self._account.protections.clear()
        self._account.triggered_protections.clear()
        self._account.protection_pairs.clear()
        return True

    def _trigger_protection_locked(self, pair: Pair, *, trigger_price: Decimal | None = None) -> ProtectionState | None:
        protection = self._account.protections.get(pair)
        if protection is None:
            return None
        quote = self._quote_locked(pair)
        trigger_price = trigger_price if trigger_price is not None else self._trigger_price(protection, quote.last)
        if trigger_price is None:
            return None
        intent = OrderIntent(
            pair,
            "sell" if protection.position_side == "long" else "buy",
            protection.amount,
            "market",
            None,
            True,
        )
        order = self._place_order_locked(intent, trigger_price, reserved_order_id=protection.actual_order_ids[0])
        if order.status != "filled":
            return None
        self._account.protections.pop(pair, None)
        for protection_id in protection.protection_ids:
            self._account.protection_pairs.pop(protection_id, None)
        return ProtectionState(
            protection.protection_ids,
            protection.pair,
            protection.position_side,
            protection.amount,
            protection.stop_loss,
            protection.take_profit,
            False,
            True,
            protection.actual_order_ids,
        )

    @staticmethod
    def _trigger_price(protection: ProtectionState, price: Decimal) -> Decimal | None:
        if protection.position_side == "long":
            if protection.stop_loss is not None and price <= protection.stop_loss:
                return protection.stop_loss
            if protection.take_profit is not None and price >= protection.take_profit:
                return protection.take_profit
            return None
        if protection.stop_loss is not None and price >= protection.stop_loss:
            return protection.stop_loss
        if protection.take_profit is not None and price <= protection.take_profit:
            return protection.take_profit
        return None

    def _quote_locked(self, pair: Pair) -> VenueQuote:
        try:
            return self._account.quotes[pair]
        except KeyError:
            raise VenueOperationError(f"{self.connection_id}: Paper quote is not set for {pair}") from None

    def _require_open(self) -> None:
        if self._closed:
            raise VenueOperationError(f"{self.connection_id}: Paper session is closed")

    def _require_supported_pair(self, pair: Pair) -> None:
        if pair.market_type not in _SUPPORTED_MARKET_TYPES:
            raise VenueOperationError(f"{self.connection_id}: unsupported Paper market type {pair.market_type}")


class PaperVenueAdapter:
    """Factory for credential-free Paper sessions backed by connection parameters."""

    adapter_id = "paper"

    def __init__(self, *, clock=None, cost_model=None) -> None:
        self._accounts: dict[str, _PaperAccount] = {}
        self.database_url = None
        self.account_store = None
        self.clock = clock or (lambda: datetime.now(UTC))
        self.cost_model = cost_model

    def capabilities(self, environment: str) -> VenueCapabilities:
        self._require_environment(environment)
        return VenueCapabilities(
            _SUPPORTED_MARKET_TYPES,
            native_protection=True,
            hedge_mode=False,
            reduce_only=True,
            supported_order_types=frozenset({"market", "limit"}),
            account_reads=ACCOUNT_READS,
            exit_operations=EXIT_OPERATIONS,
        )

    async def connect(
        self,
        connection: VenueConnection,
        credentials: CredentialPayload | None,
    ) -> PaperVenueSession:
        if connection.adapter_id != self.adapter_id:
            raise ValueError("Paper adapter requires connection adapter_id=paper")
        self._require_environment(connection.environment)
        if connection.credential_ref is not None or credentials is not None:
            raise ValueError("Paper connection does not accept credentials")
        initial_equity = self._initial_equity(connection)
        if self.account_store is None and self.database_url:
            from cryptotrader.accounts.store import AccountStore

            self.account_store = AccountStore(self.database_url)
        store = self.account_store
        if store is not None:
            self._accounts = store.paper_accounts
            async with store.paper_locks.setdefault(connection.id, asyncio.Lock()):
                if connection.id not in self._accounts:
                    saved = await store.load_paper(connection.id)
                    account = (
                        _paper_from_payload(saved)
                        if saved
                        else _PaperAccount(initial_equity, connection.leverage, self.clock())
                    )
                    if saved is None:
                        await store.save_paper(connection.id, _paper_payload(account))
                    self._accounts[connection.id] = account
        account = self._accounts.get(connection.id)
        if account is None:
            account = _PaperAccount(initial_equity, connection.leverage, self.clock())
            self._accounts[connection.id] = account
        elif account.initial_equity != initial_equity or account.leverage != connection.leverage:
            raise ValueError("Paper connection parameters changed for an existing session")
        return PaperVenueSession(
            connection,
            account,
            self.capabilities(connection.environment),
            store,
            clock=self.clock,
            cost_model=self.cost_model,
        )

    @staticmethod
    def _initial_equity(connection: VenueConnection) -> Decimal:
        value = PaperParameters.model_validate(dict(connection.parameters)).initial_equity
        try:
            initial_equity = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("Paper initial_equity must be Decimal-compatible") from None
        if not initial_equity.is_finite() or initial_equity <= 0:
            raise ValueError("Paper initial_equity must be finite and positive")
        return initial_equity

    @staticmethod
    def _require_environment(environment: str) -> None:
        if environment != "paper":
            raise ValueError(f"unsupported Paper environment: {environment}")


def _paper_payload(account):
    from cryptotrader.accounts.store import payload

    keys = (
        "initial_equity",
        "leverage",
        "started_at",
        "balances",
        "quotes",
        "quote_times",
        "positions",
        "spot_entry_prices",
        "orders",
        "fills",
        "funding",
        "protections",
        "triggered_protections",
        "protection_pairs",
        "order_sequence",
        "protection_sequence",
    )
    return {key: payload(getattr(account, key)) for key in keys}


def _paper_from_payload(value):
    from cryptotrader.accounts.store import fill_from_payload, funding_from_payload
    from cryptotrader.pair import Pair

    def number(item):
        return Decimal(item) if item is not None else None

    def normalized(item, cls, decimal_fields):
        result = item | {"pair": Pair.parse(item["pair"])}
        result.update({key: number(item[key]) for key in decimal_fields})
        if "protection_ids" in result:
            result["protection_ids"] = tuple(result["protection_ids"])
            result["actual_order_ids"] = tuple(result["actual_order_ids"])
        return cls(**result)

    account = _PaperAccount(
        Decimal(value["initial_equity"]), value["leverage"], datetime.fromisoformat(value["started_at"])
    )
    account.balances = {key: Decimal(amount) for key, amount in value["balances"].items()}
    account.quotes = {
        Pair.parse(key): normalized(item, VenueQuote, ("bid", "ask", "last")) for key, item in value["quotes"].items()
    }
    account.quote_times = {Pair.parse(key): datetime.fromisoformat(item) for key, item in value["quote_times"].items()}
    account.positions = {
        Pair.parse(key): _PaperPosition(Decimal(item["signed_amount"]), number(item["entry_price"]))
        for key, item in value["positions"].items()
    }
    account.spot_entry_prices = {Pair.parse(key): Decimal(item) for key, item in value["spot_entry_prices"].items()}
    account.orders = {
        key: normalized(item, NormalizedOrder, ("amount", "filled_amount", "average_price"))
        for key, item in value["orders"].items()
    }
    account.fills = [fill_from_payload(item) for item in value["fills"]]
    account.funding = [funding_from_payload(item) for item in value.get("funding", [])]
    for key in ("protections", "triggered_protections"):
        setattr(
            account,
            key,
            {
                Pair.parse(pair): normalized(item, ProtectionState, ("amount", "stop_loss", "take_profit"))
                for pair, item in value[key].items()
            },
        )
    account.protection_pairs = {key: Pair.parse(pair) for key, pair in value["protection_pairs"].items()}
    account.order_sequence = value["order_sequence"]
    account.protection_sequence = value["protection_sequence"]
    return account


def create_adapter() -> PaperVenueAdapter:
    return PaperVenueAdapter()
