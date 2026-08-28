"""Platform-neutral CCXT session mechanics."""

from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
from cryptotrader.venues.models import (
    ConnectionPosition,
    NormalizedOrder,
    OpenVenueState,
    OrderIntent,
    ProtectionState,
    VenueQuote,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cryptotrader.pair import Pair
    from cryptotrader.venues.models import ProtectionSpec, VenueConnection


class VenueOperationError(RuntimeError):
    """A credential-safe failure at the normalized venue boundary."""


def create_async_client(exchange_id: str, config: dict[str, Any], client_factory: Callable | None = None) -> Any:
    """Build an async CCXT client without importing legacy execution code."""
    if client_factory is not None:
        return client_factory(config)
    try:
        import ccxt.async_support as ccxt_async
    except ImportError:
        raise ImportError("ccxt.async_support is required for CCXT venue adapters") from None
    return getattr(ccxt_async, exchange_id)(config)


class CcxtVenueBase:
    """Common precision, normalization, metadata, and lifecycle behavior."""

    def __init__(self, connection: VenueConnection, client: Any) -> None:
        self.connection_id = connection.id
        self.connection = connection
        self._client = client
        self._markets_loaded = False
        self._markets_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False

    @property
    def client(self) -> Any:
        """Expose the configured client for diagnostics; methods never return raw payloads."""
        return self._client

    async def _ensure_markets(self) -> None:
        if self._markets_loaded:
            return
        async with self._markets_lock:
            if not self._markets_loaded:
                await self._call("load markets", self._client.load_markets)
                self._markets_loaded = True

    async def _call(self, operation: str, method: Callable, *args: Any) -> Any:
        try:
            return await method(*args)
        except VenueOperationError:
            raise
        except Exception:
            raise VenueOperationError(f"{self.connection_id}: {operation} failed") from None

    @staticmethod
    def _decimal(value: Any, field: str, *, default: Decimal | None = None) -> Decimal:
        if value is None or value == "":
            if default is not None:
                return default
            raise VenueOperationError(f"invalid {field}")
        try:
            result = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            raise VenueOperationError(f"invalid {field}") from None
        if not result.is_finite():
            raise VenueOperationError(f"invalid {field}")
        return result

    async def _market(self, pair: Pair) -> dict[str, Any]:
        await self._ensure_markets()
        market = (getattr(self._client, "markets", None) or {}).get(pair.to_ccxt())
        if not isinstance(market, dict):
            raise VenueOperationError(f"{self.connection_id}: unknown market {pair}")
        return market

    async def _contract_size(self, pair: Pair) -> Decimal:
        if pair.market_type == "spot":
            return Decimal("1")
        market = await self._market(pair)
        size = self._decimal(market.get("contractSize"), "contract size")
        if size <= 0:
            raise VenueOperationError(f"{self.connection_id}: invalid contract size for {pair}")
        return size

    async def _amount_to_venue(self, pair: Pair, base_amount: Decimal) -> Decimal:
        contract_size = await self._contract_size(pair)
        raw_amount = base_amount / contract_size
        precise = self._client.amount_to_precision(pair.to_ccxt(), str(raw_amount))
        result = self._decimal(precise, "amount precision")
        if result <= 0:
            raise VenueOperationError(f"{self.connection_id}: amount rounds to zero for {pair}")
        return result

    async def _price_to_venue(self, pair: Pair, price: Decimal) -> Decimal:
        await self._market(pair)
        precise = self._client.price_to_precision(pair.to_ccxt(), str(price))
        result = self._decimal(precise, "price precision")
        if result <= 0:
            raise VenueOperationError(f"{self.connection_id}: price rounds to zero for {pair}")
        return result

    async def fetch_balances(self) -> Mapping[str, Decimal]:
        raw = await self._call("fetch balance", self._client.fetch_balance)
        totals = raw.get("total") if isinstance(raw, dict) else None
        if not isinstance(totals, dict):
            raise VenueOperationError(f"{self.connection_id}: balance response has no totals")
        normalized = {
            str(asset): amount
            for asset, value in totals.items()
            if (amount := self._decimal(value, f"{asset} balance", default=Decimal("0"))) != 0
        }
        return MappingProxyType(normalized)

    async def fetch_position(self, pair: Pair) -> ConnectionPosition:
        if pair.market_type == "spot":
            balances = await self.fetch_balances()
            amount = balances.get(pair.base, Decimal("0"))
            return ConnectionPosition(pair, amount, Decimal("0"), None)

        await self._market(pair)
        raw_positions = await self._call("fetch positions", self._client.fetch_positions, [pair.to_ccxt()])
        signed_amount = Decimal("0")
        signed_notional = Decimal("0")
        weighted_entry = Decimal("0")
        absolute_amount = Decimal("0")
        contract_size = await self._contract_size(pair)
        for raw in raw_positions or ():
            if not isinstance(raw, dict) or raw.get("symbol") != pair.to_ccxt():
                continue
            contracts = self._decimal(raw.get("contracts"), "position contracts", default=Decimal("0"))
            amount = contracts * contract_size
            side = str(raw.get("side") or "long").lower()
            sign = Decimal("-1") if side == "short" else Decimal("1")
            signed = amount * sign
            entry = self._decimal(raw.get("entryPrice"), "entry price", default=Decimal("0"))
            raw_notional = self._decimal(raw.get("notional"), "position notional", default=amount * entry)
            signed_amount += signed
            signed_notional += abs(raw_notional) * sign
            weighted_entry += entry * abs(amount)
            absolute_amount += abs(amount)
        entry_price = weighted_entry / absolute_amount if absolute_amount else None
        return ConnectionPosition(pair, signed_amount, signed_notional, entry_price)

    async def fetch_portfolio(self, pair: Pair) -> ConnectionPortfolioSnapshot:
        balances, position = await asyncio.gather(self.fetch_balances(), self.fetch_position(pair))
        equity_asset = pair.settle or pair.quote
        return ConnectionPortfolioSnapshot(
            self.connection_id,
            balances.get(equity_asset, Decimal("0")),
            balances,
            position,
        )

    async def fetch_quote(self, pair: Pair) -> VenueQuote:
        await self._market(pair)
        raw = await self._call("fetch quote", self._client.fetch_ticker, pair.to_ccxt())
        return VenueQuote(
            pair,
            self._decimal(raw.get("bid"), "bid"),
            self._decimal(raw.get("ask"), "ask"),
            self._decimal(raw.get("last"), "last"),
        )

    async def place_order(self, intent: OrderIntent) -> NormalizedOrder:
        await self._market(intent.pair)
        venue_amount = await self._amount_to_venue(intent.pair, intent.amount)
        venue_price = await self._price_to_venue(intent.pair, intent.price) if intent.price is not None else None
        params = await self._order_params(intent)
        raw = await self._call(
            "place order",
            self._client.create_order,
            intent.pair.to_ccxt(),
            intent.order_type,
            intent.side,
            float(venue_amount),
            float(venue_price) if venue_price is not None else None,
            params,
        )
        return await self._normalize_order(raw, intent.pair, fallback_reduce_only=intent.reduce_only)

    async def _normalize_order(
        self,
        raw: dict[str, Any],
        pair: Pair,
        *,
        fallback_reduce_only: bool = False,
    ) -> NormalizedOrder:
        contract_size = await self._contract_size(pair)
        amount = self._decimal(raw.get("amount"), "order amount") * contract_size
        filled = self._decimal(raw.get("filled"), "filled amount", default=Decimal("0")) * contract_size
        average_raw = raw.get("average")
        average = self._decimal(average_raw, "average price") if average_raw not in (None, "") else None
        info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
        reduce_only = self._boolean(raw.get("reduceOnly", info.get("reduceOnly", fallback_reduce_only)))
        return NormalizedOrder(
            str(raw.get("id") or ""),
            pair,
            str(raw.get("side") or ""),
            str(raw.get("type") or ""),
            amount,
            filled,
            average,
            str(raw.get("status") or "unknown"),
            reduce_only,
        )

    async def list_open_state(self, pair: Pair) -> OpenVenueState:
        await self._market(pair)
        position, raw_orders, protections = await asyncio.gather(
            self.fetch_position(pair),
            self._call("fetch open orders", self._client.fetch_open_orders, pair.to_ccxt()),
            self._fetch_protections(pair),
        )
        orders = []
        for raw in raw_orders or ():
            if not isinstance(raw, dict):
                raise VenueOperationError(f"{self.connection_id}: invalid open order")
            orders.append(await self._normalize_order(raw, pair))
        return OpenVenueState(position, tuple(orders), tuple(protections))

    @staticmethod
    def _boolean(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == "true"
        return value == 1

    async def _order_params(self, intent: OrderIntent) -> dict[str, Any]:
        raise NotImplementedError

    async def _fetch_protections(self, pair: Pair) -> tuple[ProtectionState, ...]:
        raise NotImplementedError

    async def replace_protection(self, spec: ProtectionSpec) -> ProtectionState:
        raise NotImplementedError

    async def cancel_protection(self, protection_ids: tuple[str, ...]) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            await self._call("close session", self._client.close)
