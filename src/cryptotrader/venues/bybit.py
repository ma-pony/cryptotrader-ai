"""Bybit testnet, demo, and live venue adapter."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from cryptotrader.venues.ccxt_base import CcxtVenueBase, VenueOperationError, create_async_client
from cryptotrader.venues.models import ProtectionState, VenueCapabilities

if TYPE_CHECKING:
    from collections.abc import Callable
    from decimal import Decimal

    from cryptotrader.pair import Pair
    from cryptotrader.runtime_config.secrets import CredentialPayload
    from cryptotrader.venues.models import OrderIntent, ProtectionSpec, VenueConnection


class BybitVenueSession(CcxtVenueBase):
    """One isolated Bybit unified-account session."""

    def __init__(self, connection: VenueConnection, client: Any, capabilities: VenueCapabilities) -> None:
        super().__init__(connection, client, capabilities)
        self._protection_pairs: dict[str, tuple[Pair, int]] = {}
        self._position_mode_lock = asyncio.Lock()
        self._hedged_markets: dict[str, bool] = {}

    @staticmethod
    def _integer(value: Any, field: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            raise VenueOperationError(f"invalid {field}") from None

    async def _position_mode_is_hedged(self, pair: Pair) -> bool:
        symbol = pair.to_ccxt()
        if symbol in self._hedged_markets:
            return self._hedged_markets[symbol]
        async with self._position_mode_lock:
            if symbol in self._hedged_markets:
                return self._hedged_markets[symbol]
            rows = await self._call("resolve Bybit position mode", self._client.fetch_positions, [symbol])
            if not isinstance(rows, (list, tuple)):
                raise VenueOperationError(f"{self.connection_id}: invalid Bybit position mode response")
            indices: set[int] = set()
            for row in rows:
                if not isinstance(row, dict) or row.get("symbol") != symbol:
                    continue
                info = row.get("info") if isinstance(row.get("info"), dict) else None
                if info is None or "positionIdx" not in info:
                    raise VenueOperationError(f"{self.connection_id}: invalid Bybit position mode response")
                index = self._integer(info["positionIdx"], "Bybit position index")
                if index not in {0, 1, 2}:
                    raise VenueOperationError(f"{self.connection_id}: invalid Bybit position mode response")
                indices.add(index)
            if indices == {0}:
                hedged = False
            elif indices and indices <= {1, 2}:
                hedged = True
            else:
                raise VenueOperationError(f"{self.connection_id}: ambiguous Bybit position mode response")
            self._hedged_markets[symbol] = hedged
            return hedged

    async def _position_index(self, pair: Pair, *, side: str, reduce_only: bool) -> int:
        await self._market(pair)
        if not await self._position_mode_is_hedged(pair):
            return 0
        if reduce_only:
            return 1 if side == "sell" else 2
        return 1 if side == "buy" else 2

    async def _order_params(self, intent: OrderIntent) -> dict[str, Any]:
        if intent.pair.market_type == "spot":
            return {"orderLinkId": intent.client_order_id} if intent.client_order_id is not None else {}
        params: dict[str, Any] = {
            "positionIdx": await self._position_index(
                intent.pair,
                side=intent.side,
                reduce_only=intent.reduce_only,
            ),
        }
        if intent.reduce_only:
            params["reduceOnly"] = True
        if intent.client_order_id is not None:
            params["orderLinkId"] = intent.client_order_id
        return params

    async def _configure_market(self, pair: Pair, market: dict[str, Any]) -> None:
        await self._call(
            "configure Bybit margin mode",
            self._client.set_margin_mode,
            self.connection.margin_mode,
            None,
            {},
        )
        await self._call(
            "configure Bybit leverage",
            self._client.set_leverage,
            self.connection.leverage,
            pair.to_ccxt(),
            {"category": "linear"},
        )

    def _account_equity(self, raw_balance: Any) -> Decimal:
        info = raw_balance.get("info") if isinstance(raw_balance, dict) else None
        result = info.get("result") if isinstance(info, dict) else None
        accounts = result.get("list") if isinstance(result, dict) else None
        account = accounts[0] if isinstance(accounts, list) and accounts and isinstance(accounts[0], dict) else None
        if account is None:
            raise VenueOperationError(f"{self.connection_id}: invalid Bybit equity response")
        return self._decimal(account.get("totalEquity"), "Bybit total equity")

    async def replace_protection(self, spec: ProtectionSpec) -> ProtectionState:
        market = await self._market(spec.pair)
        if spec.pair.market_type != "swap" or not market.get("linear"):
            raise VenueOperationError(f"{self.connection_id}: Bybit native protection requires linear swap")
        venue_amount = await self._amount_to_venue(spec.pair, spec.amount)
        stop_loss = await self._price_to_venue(spec.pair, spec.stop_loss) if spec.stop_loss is not None else None
        take_profit = await self._price_to_venue(spec.pair, spec.take_profit) if spec.take_profit is not None else None
        position_index = await self._position_index(
            spec.pair,
            side="buy" if spec.position_side == "long" else "sell",
            reduce_only=False,
        )
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": str(market["id"]),
            "tpslMode": "Full",
            "positionIdx": position_index,
            "slTriggerBy": "LastPrice",
            "tpTriggerBy": "LastPrice",
            "slOrderType": "Market",
            "tpOrderType": "Market",
            "stopLoss": format(stop_loss, "f") if stop_loss is not None else "0",
            "takeProfit": format(take_profit, "f") if take_profit is not None else "0",
        }
        response = await self._call(
            "create Bybit protection",
            self._client.private_post_v5_position_trading_stop,
            params,
        )
        if not isinstance(response, dict) or self._integer(response.get("retCode", -1), "Bybit return code") != 0:
            raise VenueOperationError(f"{self.connection_id}: Bybit protection rejected")
        expected_id = self._protection_id(str(market["id"]), position_index)
        normalized_amount = venue_amount * await self._contract_size(spec.pair)
        state = await self.list_open_state(spec.pair)
        for protection in state.protections:
            if (
                expected_id in protection.protection_ids
                and protection.position_side == spec.position_side
                and protection.amount == normalized_amount
                and protection.stop_loss == stop_loss
                and protection.take_profit == take_profit
            ):
                return protection
        raise VenueOperationError(f"{self.connection_id}: protection not confirmed by Bybit query")

    async def _fetch_protections(self, pair: Pair) -> tuple[ProtectionState, ...]:
        market = await self._market(pair)
        rows = await self._call("list Bybit protections", self._client.fetch_positions, [pair.to_ccxt()])
        if not isinstance(rows, (list, tuple)):
            raise VenueOperationError(f"{self.connection_id}: invalid Bybit protection response")
        contract_size = await self._contract_size(pair)
        protections: list[ProtectionState] = []
        for row in rows or ():
            if not isinstance(row, dict):
                raise VenueOperationError(f"{self.connection_id}: invalid Bybit protection response")
            if row.get("symbol") != pair.to_ccxt():
                continue
            info = row.get("info") if isinstance(row.get("info"), dict) else {}
            raw_stop = info.get("stopLoss")
            raw_take = info.get("takeProfit")
            stop_loss = self._decimal(raw_stop, "Bybit stop loss") if raw_stop not in (None, "", "0") else None
            take_profit = self._decimal(raw_take, "Bybit take profit") if raw_take not in (None, "", "0") else None
            if stop_loss is None and take_profit is None:
                continue
            if "positionIdx" not in info:
                raise VenueOperationError(f"{self.connection_id}: invalid Bybit protection response")
            position_index = self._integer(info["positionIdx"], "Bybit position index")
            protection_id = self._protection_id(str(market["id"]), position_index)
            self._protection_pairs[protection_id] = (pair, position_index)
            amount = self._decimal(row.get("contracts"), "Bybit protection size") * contract_size
            protections.append(
                self._sync(
                    "normalize Bybit protection",
                    ProtectionState,
                    (protection_id,),
                    pair,
                    str(row.get("side") or ("long" if position_index == 1 else "short")).lower(),
                    amount,
                    stop_loss,
                    take_profit,
                    True,
                    False,
                )
            )
        return tuple(protections)

    async def cancel_protection(self, protection_ids: tuple[str, ...]) -> None:
        for protection_id in protection_ids:
            target = self._protection_pairs.get(protection_id)
            if target is None:
                continue
            pair, position_index = target
            market = await self._market(pair)
            response = await self._call(
                "cancel Bybit protection",
                self._client.private_post_v5_position_trading_stop,
                {
                    "category": "linear",
                    "symbol": str(market["id"]),
                    "tpslMode": "Full",
                    "positionIdx": position_index,
                    "stopLoss": "0",
                    "takeProfit": "0",
                },
            )
            if not isinstance(response, dict) or self._integer(response.get("retCode", -1), "Bybit return code") != 0:
                raise VenueOperationError(f"{self.connection_id}: cancel Bybit protection rejected")

    @staticmethod
    def _protection_id(market_id: str, position_index: int) -> str:
        return f"bybit-position:{market_id}:{position_index}"


class BybitVenueAdapter:
    adapter_id = "bybit"
    _ENVIRONMENTS = frozenset({"testnet", "demo", "live"})

    def __init__(self, *, client_factory: Callable | None = None) -> None:
        self._client_factory = client_factory

    def capabilities(self, environment: str) -> VenueCapabilities:
        self._require_environment(environment)
        return VenueCapabilities(
            frozenset({"spot", "swap"}),
            native_protection=True,
            hedge_mode=True,
            reduce_only=True,
            supported_order_types=frozenset({"market", "limit"}),
        )

    async def connect(
        self,
        connection: VenueConnection,
        credentials: CredentialPayload | None,
    ) -> BybitVenueSession:
        self._require_connection(connection)
        if credentials is None or not credentials.api_key or not credentials.secret:
            raise ValueError("Bybit connection requires API key and secret")
        config = {
            "apiKey": credentials.api_key,
            "secret": credentials.secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "fetchMarkets": ["spot", "linear"],
            },
        }
        client = create_async_client("bybit", config, self._client_factory)
        if connection.environment == "testnet":
            client.set_sandbox_mode(True)
        elif connection.environment == "demo":
            client.enable_demo_trading(True)
        return BybitVenueSession(connection, client, self.capabilities(connection.environment))

    def _require_connection(self, connection: VenueConnection) -> None:
        if connection.adapter_id != self.adapter_id:
            raise ValueError("Bybit adapter requires connection adapter_id=bybit")
        self._require_environment(connection.environment)

    def _require_environment(self, environment: str) -> None:
        if environment not in self._ENVIRONMENTS:
            raise ValueError(f"unsupported Bybit environment: {environment}")


def create_adapter() -> BybitVenueAdapter:
    return BybitVenueAdapter()
