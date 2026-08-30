"""Deterministic in-memory Paper implementation of the venue contracts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from cryptotrader.configuration.catalog import PluginConfiguration, configured_factory
from cryptotrader.configuration.fields import LocalizedText
from cryptotrader.configuration.parameters import PaperParameters
from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
from cryptotrader.venues.ccxt_base import VenueOperationError
from cryptotrader.venues.models import (
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
    balances: dict[str, Decimal] = field(init=False)
    quotes: dict[Pair, VenueQuote] = field(default_factory=dict)
    positions: dict[Pair, _PaperPosition] = field(default_factory=dict)
    spot_entry_prices: dict[Pair, Decimal] = field(default_factory=dict)
    orders: dict[str, NormalizedOrder] = field(default_factory=dict)
    protections: dict[Pair, ProtectionState] = field(default_factory=dict)
    triggered_protections: dict[Pair, ProtectionState] = field(default_factory=dict)
    protection_pairs: dict[str, Pair] = field(default_factory=dict)
    pair_locks: dict[Pair, asyncio.Lock] = field(default_factory=dict)
    order_sequence: int = 0
    protection_sequence: int = 0

    def __post_init__(self) -> None:
        self.balances = {"USDT": self.initial_equity}

    def lock_for(self, pair: Pair) -> asyncio.Lock:
        return self.pair_locks.setdefault(pair, asyncio.Lock())


class PaperVenueSession:
    """One session over connection-local deterministic Paper account state."""

    def __init__(
        self,
        connection: VenueConnection,
        account: _PaperAccount,
        capabilities: VenueCapabilities,
    ) -> None:
        self.connection_id = connection.id
        self.connection = connection
        self._account = account
        self._capabilities = capabilities
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def capabilities(self) -> VenueCapabilities:
        return self._capabilities

    async def set_quote(self, pair: Pair, price: Decimal) -> VenueQuote:
        """Install the latest deterministic Paper quote for one pair."""
        self._require_open()
        self._require_supported_pair(pair)
        if not isinstance(price, Decimal) or not price.is_finite() or price <= 0:
            raise ValueError("Paper quote price must be a positive finite Decimal")
        async with self._account.lock_for(pair):
            quote = VenueQuote(pair, price, price, price)
            self._account.quotes[pair] = quote
            self._apply_bankruptcy_locked()
            return quote

    async def fetch_quote(self, pair: Pair) -> VenueQuote:
        self._require_open()
        self._require_supported_pair(pair)
        async with self._account.lock_for(pair):
            return self._quote_locked(pair)

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
        async with self._account.lock_for(intent.pair):
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
        async with self._account.lock_for(pair):
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
        async with self._account.lock_for(pair):
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
        async with self._account.lock_for(spec.pair):
            quote = self._quote_locked(spec.pair)
            self._apply_bankruptcy_locked()
            position = self._account.positions.get(spec.pair, _PaperPosition())
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
            async with self._account.lock_for(pair):
                current = self._account.protections.get(pair)
                if current is None or not set(current.protection_ids).intersection(protection_ids):
                    continue
                self._account.protections.pop(pair, None)
                for protection_id in current.protection_ids:
                    self._account.protection_pairs.pop(protection_id, None)

    async def list_open_state(self, pair: Pair) -> OpenVenueState:
        self._require_open()
        self._require_supported_pair(pair)
        async with self._account.lock_for(pair):
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

    def _place_order_locked(self, intent: OrderIntent, fill_price: Decimal) -> NormalizedOrder:
        self._account.order_sequence += 1
        order_id = f"{self.connection_id}-paper-order-{self._account.order_sequence}"
        if intent.pair.market_type == "spot":
            accepted = self._fill_spot_locked(intent, fill_price)
        else:
            accepted = self._fill_derivative_locked(intent, fill_price)
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
        return order

    def _fill_spot_locked(self, intent: OrderIntent, fill_price: Decimal) -> bool:
        quote_asset = intent.pair.quote
        base_asset = intent.pair.base
        quote_balance = self._account.balances.get(quote_asset, Decimal("0"))
        base_balance = self._account.balances.get(base_asset, Decimal("0"))
        cost = intent.amount * fill_price
        if intent.side == "buy":
            if intent.reduce_only or quote_balance < cost:
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

    def _fill_derivative_locked(self, intent: OrderIntent, fill_price: Decimal) -> bool:
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
            available_equity = self._equity_locked()
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

    def _trigger_protection_locked(self, pair: Pair) -> ProtectionState | None:
        protection = self._account.protections.get(pair)
        if protection is None:
            return None
        quote = self._quote_locked(pair)
        trigger_price = self._trigger_price(protection, quote.last)
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
        order = self._place_order_locked(intent, trigger_price)
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

    def __init__(self) -> None:
        self._accounts: dict[str, _PaperAccount] = {}

    def capabilities(self, environment: str) -> VenueCapabilities:
        self._require_environment(environment)
        return VenueCapabilities(
            _SUPPORTED_MARKET_TYPES,
            native_protection=True,
            hedge_mode=False,
            reduce_only=True,
            supported_order_types=frozenset({"market", "limit"}),
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
        account = self._accounts.get(connection.id)
        if account is None:
            account = _PaperAccount(initial_equity, connection.leverage)
            self._accounts[connection.id] = account
        elif account.initial_equity != initial_equity or account.leverage != connection.leverage:
            raise ValueError("Paper connection parameters changed for an existing session")
        return PaperVenueSession(connection, account, self.capabilities(connection.environment))

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


@configured_factory(
    PluginConfiguration(
        id="paper",
        label=LocalizedText(zh_CN="模拟交易", en_US="Paper trading"),
        description=LocalizedText(
            zh_CN="在独立的模拟账户中验证交易策略; 不会向交易所提交订单。",
            en_US="Validates strategies in an isolated simulated account without submitting exchange orders.",
        ),
        parameter_model=PaperParameters,
        environments=("paper",),
    )
)
def create_adapter() -> PaperVenueAdapter:
    return PaperVenueAdapter()
