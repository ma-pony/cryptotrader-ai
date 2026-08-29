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
    VenueCapabilities,
    VenueQuote,
)
from cryptotrader.venues.protocol import VenueOperationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cryptotrader.pair import Pair
    from cryptotrader.venues.models import ProtectionSpec, VenueConnection


def create_async_client(adapter_id: str, config: dict[str, Any], client_factory: Callable | None = None) -> Any:
    """Build an async CCXT client without importing legacy execution code."""
    if client_factory is not None:
        return client_factory(config)
    try:
        import ccxt.async_support as ccxt_async
    except ImportError:
        raise ImportError("ccxt.async_support is required for CCXT venue adapters") from None
    return getattr(ccxt_async, adapter_id)(config)


class CcxtVenueBase:
    """Common precision, normalization, metadata, and lifecycle behavior."""

    def __init__(self, connection: VenueConnection, client: Any, capabilities: VenueCapabilities) -> None:
        self.connection_id = connection.id
        self.connection = connection
        self._capabilities = capabilities
        self._client = client
        self._markets_loaded = False
        self._markets_lock = asyncio.Lock()
        self._configuration_lock = asyncio.Lock()
        self._configured_markets: set[str] = set()
        self._close_lock = asyncio.Lock()
        self._closed = False

    @property
    def capabilities(self) -> VenueCapabilities:
        return self._capabilities

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

    def _sync(self, operation: str, method: Callable, *args: Any) -> Any:
        try:
            return method(*args)
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
        markets = self._sync("read market metadata", lambda: getattr(self._client, "markets", None))
        market = markets.get(pair.to_ccxt()) if isinstance(markets, dict) else None
        if not isinstance(market, dict):
            raise VenueOperationError(f"{self.connection_id}: unknown market {pair}")
        if not isinstance(market.get("id"), str) or not market["id"]:
            raise VenueOperationError(f"{self.connection_id}: invalid market metadata for {pair}")
        if pair.market_type == "swap" and (market.get("inverse") is True or market.get("linear") is not True):
            raise VenueOperationError(f"{self.connection_id}: inverse contracts are unsupported for {pair}")
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
        precise = self._sync(
            "normalize amount",
            self._client.amount_to_precision,
            pair.to_ccxt(),
            str(raw_amount),
        )
        result = self._decimal(precise, "amount precision")
        if result <= 0:
            raise VenueOperationError(f"{self.connection_id}: amount rounds to zero for {pair}")
        return result

    async def normalize_amount(self, pair: Pair, base_amount: Decimal) -> Decimal:
        """Normalize platform-neutral base units without exposing contract units."""
        if not isinstance(base_amount, Decimal) or not base_amount.is_finite() or base_amount <= 0:
            raise VenueOperationError(f"{self.connection_id}: base amount must be a positive finite Decimal")
        await self._market(pair)
        contract_size = await self._contract_size(pair)
        venue_amount = await self._amount_to_venue(pair, base_amount)
        normalized = venue_amount * contract_size
        if not normalized.is_finite() or normalized <= 0:
            raise VenueOperationError(f"{self.connection_id}: amount rounds to zero for {pair}")
        if normalized > base_amount:
            raise VenueOperationError(f"{self.connection_id}: unsafe amount normalization for {pair}")
        return normalized

    async def _price_to_venue(self, pair: Pair, price: Decimal) -> Decimal:
        await self._market(pair)
        precise = self._sync(
            "normalize price",
            self._client.price_to_precision,
            pair.to_ccxt(),
            str(price),
        )
        result = self._decimal(precise, "price precision")
        if result <= 0:
            raise VenueOperationError(f"{self.connection_id}: price rounds to zero for {pair}")
        return result

    async def fetch_balances(self) -> Mapping[str, Decimal]:
        raw = await self._call("fetch balance", self._client.fetch_balance)
        return self._normalize_balances(raw)

    def _normalize_balances(self, raw: Any) -> Mapping[str, Decimal]:
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
            balances, quote = await asyncio.gather(self.fetch_balances(), self.fetch_quote(pair))
            amount = balances.get(pair.base, Decimal("0"))
            return self._sync(
                "normalize spot position",
                ConnectionPosition,
                pair,
                amount,
                amount * quote.last,
                None,
            )

        await self._market(pair)
        raw_positions = await self._call("fetch positions", self._client.fetch_positions, [pair.to_ccxt()])
        if not isinstance(raw_positions, (list, tuple)):
            raise VenueOperationError(f"{self.connection_id}: invalid position response")
        signed_amount = Decimal("0")
        signed_notional = Decimal("0")
        weighted_entry = Decimal("0")
        absolute_amount = Decimal("0")
        long_amount = Decimal("0")
        short_amount = Decimal("0")
        contract_size = await self._contract_size(pair)
        for raw in raw_positions:
            if not isinstance(raw, dict):
                raise VenueOperationError(f"{self.connection_id}: invalid position response")
            if raw.get("symbol") != pair.to_ccxt():
                continue
            contracts = self._decimal(raw.get("contracts"), "position contracts", default=Decimal("0"))
            amount = contracts * contract_size
            side = str(raw.get("side") or "long").lower()
            sign = Decimal("-1") if side == "short" else Decimal("1")
            signed = amount * sign
            if amount != 0:
                if sign > 0:
                    long_amount += abs(amount)
                else:
                    short_amount += abs(amount)
            entry = self._decimal(raw.get("entryPrice"), "entry price", default=Decimal("0"))
            raw_notional = self._decimal(raw.get("notional"), "position notional", default=amount * entry)
            signed_amount += signed
            signed_notional += abs(raw_notional) * sign
            weighted_entry += entry * abs(amount)
            absolute_amount += abs(amount)
        if long_amount != 0 and short_amount != 0:
            raise VenueOperationError(f"{self.connection_id}: simultaneous hedge legs are unsupported")
        entry_price = weighted_entry / absolute_amount if absolute_amount else None
        return self._sync(
            "normalize position",
            ConnectionPosition,
            pair,
            signed_amount,
            signed_notional,
            entry_price,
        )

    async def fetch_portfolio(self, pair: Pair) -> ConnectionPortfolioSnapshot:
        raw_balance = await self._call("fetch balance", self._client.fetch_balance)
        balances = self._normalize_balances(raw_balance)
        if pair.market_type == "spot":
            quote = await self.fetch_quote(pair)
            amount = balances.get(pair.base, Decimal("0"))
            position = self._sync(
                "normalize spot position",
                ConnectionPosition,
                pair,
                amount,
                amount * quote.last,
                None,
            )
        else:
            position = await self.fetch_position(pair)
        return self._sync(
            "normalize portfolio",
            ConnectionPortfolioSnapshot,
            self.connection_id,
            self._account_equity(raw_balance),
            balances,
            position,
        )

    async def fetch_quote(self, pair: Pair) -> VenueQuote:
        await self._market(pair)
        raw = await self._call("fetch quote", self._client.fetch_ticker, pair.to_ccxt())
        if not isinstance(raw, dict):
            raise VenueOperationError(f"{self.connection_id}: invalid quote response")
        return self._sync(
            "normalize quote",
            VenueQuote,
            pair,
            self._decimal(raw.get("bid"), "bid"),
            self._decimal(raw.get("ask"), "ask"),
            self._decimal(raw.get("last"), "last"),
        )

    async def place_order(self, intent: OrderIntent) -> NormalizedOrder:
        await self._market(intent.pair)
        if intent.pair.market_type == "swap":
            await self._ensure_market_configured(intent.pair)
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
        try:
            if not isinstance(raw, dict):
                raise ValueError
            contract_size = await self._contract_size(pair)
            amount = self._decimal(raw.get("amount"), "order amount") * contract_size
            filled = self._decimal(raw.get("filled"), "filled amount", default=Decimal("0")) * contract_size
            average_raw = raw.get("average")
            average = self._decimal(average_raw, "average price") if average_raw not in (None, "") else None
            info = raw.get("info") if isinstance(raw.get("info"), dict) else {}
            reduce_only = fallback_reduce_only or self._boolean(raw.get("reduceOnly", info.get("reduceOnly", False)))
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
        except VenueOperationError:
            raise
        except Exception:
            raise VenueOperationError(f"{self.connection_id}: invalid order response") from None

    async def _ensure_market_configured(self, pair: Pair) -> None:
        symbol = pair.to_ccxt()
        if symbol in self._configured_markets:
            return
        async with self._configuration_lock:
            if symbol not in self._configured_markets:
                market = await self._market(pair)
                await self._configure_market(pair, market)
                self._configured_markets.add(symbol)

    async def list_open_state(self, pair: Pair) -> OpenVenueState:
        await self._market(pair)
        position, raw_orders, protections = await asyncio.gather(
            self.fetch_position(pair),
            self._call("fetch open orders", self._client.fetch_open_orders, pair.to_ccxt()),
            self._fetch_protections(pair),
        )
        if not isinstance(raw_orders, (list, tuple)):
            raise VenueOperationError(f"{self.connection_id}: invalid open orders response")
        orders = []
        for raw in raw_orders:
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

    async def _configure_market(self, pair: Pair, market: dict[str, Any]) -> None:
        raise NotImplementedError

    def _account_equity(self, raw_balance: Any) -> Decimal:
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
            await self._call("close session", self._client.close)
            self._closed = True
