"""OKX demo and live venue adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cryptotrader.venues.ccxt_base import CcxtVenueBase, VenueOperationError, create_async_client
from cryptotrader.venues.models import ProtectionState, VenueCapabilities

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.pair import Pair
    from cryptotrader.runtime_config.secrets import CredentialPayload
    from cryptotrader.venues.models import OrderIntent, ProtectionSpec, VenueConnection


class OkxVenueSession(CcxtVenueBase):
    """One isolated OKX account session."""

    def __init__(self, connection: VenueConnection, client: Any) -> None:
        super().__init__(connection, client)
        self._protection_pairs: dict[str, Pair] = {}

    async def _order_params(self, intent: OrderIntent) -> dict[str, Any]:
        if intent.pair.market_type == "spot":
            return {}
        position = await self.fetch_position(intent.pair)
        if position.signed_amount > 0:
            position_side = "long"
        elif position.signed_amount < 0:
            position_side = "short"
        else:
            position_side = "long" if intent.side == "buy" else "short"
        params: dict[str, Any] = {
            "tdMode": self.connection.margin_mode,
            "posSide": position_side,
        }
        if intent.reduce_only:
            params["reduceOnly"] = True
        return params

    async def replace_protection(self, spec: ProtectionSpec) -> ProtectionState:
        market = await self._market(spec.pair)
        if spec.pair.market_type != "swap":
            raise VenueOperationError(f"{self.connection_id}: OKX native protection requires swap market")
        old_protection_ids = tuple(
            protection_id
            for protection in await self._fetch_protections(spec.pair)
            for protection_id in protection.protection_ids
        )
        venue_amount = await self._amount_to_venue(spec.pair, spec.amount)
        stop_loss = await self._price_to_venue(spec.pair, spec.stop_loss) if spec.stop_loss is not None else None
        take_profit = await self._price_to_venue(spec.pair, spec.take_profit) if spec.take_profit is not None else None
        params: dict[str, Any] = {
            "instId": str(market["id"]),
            "tdMode": self.connection.margin_mode,
            "side": "sell" if spec.position_side == "long" else "buy",
            "posSide": spec.position_side,
            "ordType": "oco",
            "sz": format(venue_amount, "f"),
            "reduceOnly": "true",
        }
        if stop_loss is not None:
            params.update(
                {
                    "slTriggerPx": format(stop_loss, "f"),
                    "slTriggerPxType": "last",
                    "slOrdPx": "-1",
                }
            )
        if take_profit is not None:
            params.update(
                {
                    "tpTriggerPx": format(take_profit, "f"),
                    "tpTriggerPxType": "last",
                    "tpOrdPx": "-1",
                }
            )
        response = await self._call("create OKX protection", self._client.private_post_trade_order_algo, params)
        if not isinstance(response, dict) or str(response.get("code")) != "0":
            raise VenueOperationError(f"{self.connection_id}: OKX protection rejected")
        data = response.get("data") or ()
        leg = data[0] if data and isinstance(data[0], dict) else {}
        if str(leg.get("sCode")) != "0" or not leg.get("algoId"):
            raise VenueOperationError(f"{self.connection_id}: OKX protection leg rejected")
        protection_id = str(leg["algoId"])
        state = await self.list_open_state(spec.pair)
        for protection in state.protections:
            if protection_id in protection.protection_ids:
                await self.cancel_protection(tuple(old_id for old_id in old_protection_ids if old_id != protection_id))
                return protection
        raise VenueOperationError(f"{self.connection_id}: protection not confirmed by OKX query")

    async def _fetch_protections(self, pair: Pair) -> tuple[ProtectionState, ...]:
        market = await self._market(pair)
        response = await self._call(
            "list OKX protections",
            self._client.private_get_trade_orders_algo_pending,
            {"ordType": "oco", "instId": str(market["id"])},
        )
        if not isinstance(response, dict) or str(response.get("code")) != "0":
            raise VenueOperationError(f"{self.connection_id}: OKX protection query rejected")
        contract_size = await self._contract_size(pair)
        protections: list[ProtectionState] = []
        for row in response.get("data") or ():
            if not isinstance(row, dict) or not row.get("algoId"):
                continue
            protection_id = str(row["algoId"])
            self._protection_pairs[protection_id] = pair
            amount = self._decimal(row.get("sz"), "OKX protection size") * contract_size
            raw_stop = row.get("slTriggerPx")
            raw_take = row.get("tpTriggerPx")
            stop_loss = self._decimal(raw_stop, "OKX stop loss") if raw_stop not in (None, "", "0") else None
            take_profit = self._decimal(raw_take, "OKX take profit") if raw_take not in (None, "", "0") else None
            if stop_loss is None and take_profit is None:
                continue
            state = str(row.get("state") or "effective").lower()
            protections.append(
                ProtectionState(
                    (protection_id,),
                    pair,
                    str(row.get("posSide") or "long").lower(),
                    amount,
                    stop_loss,
                    take_profit,
                    state in {"effective", "live"},
                    state in {"triggered", "filled"},
                )
            )
        return tuple(protections)

    async def cancel_protection(self, protection_ids: tuple[str, ...]) -> None:
        grouped: dict[Pair, list[str]] = {}
        for protection_id in protection_ids:
            pair = self._protection_pairs.get(protection_id)
            if pair is None:
                continue
            grouped.setdefault(pair, []).append(protection_id)
        for pair, ids in grouped.items():
            market = await self._market(pair)
            params = [{"algoId": protection_id, "instId": str(market["id"])} for protection_id in ids]
            try:
                response = await self._client.private_post_trade_cancel_algos(params)
            except Exception as error:
                message = str(error).lower()
                if "51400" in message or "51401" in message or "not exist" in message:
                    continue
                raise VenueOperationError(f"{self.connection_id}: cancel OKX protection failed") from None
            if not isinstance(response, dict) or str(response.get("code")) != "0":
                raise VenueOperationError(f"{self.connection_id}: cancel OKX protection rejected")
            for leg in response.get("data") or ():
                if str(leg.get("sCode")) not in {"0", "51400", "51401"}:
                    raise VenueOperationError(f"{self.connection_id}: cancel OKX protection leg rejected")


class OkxVenueAdapter:
    adapter_id = "okx"
    _ENVIRONMENTS = frozenset({"demo", "live"})

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
    ) -> OkxVenueSession:
        self._require_connection(connection)
        if credentials is None or not credentials.api_key or not credentials.secret or not credentials.passphrase:
            raise ValueError("OKX connection requires API key, secret, and passphrase")
        config = {
            "apiKey": credentials.api_key,
            "secret": credentials.secret,
            "password": credentials.passphrase,
            "enableRateLimit": True,
            "options": {"fetchMarkets": ["spot", "swap"]},
        }
        client = create_async_client("okx", config, self._client_factory)
        if connection.environment == "demo":
            client.set_sandbox_mode(True)
        return OkxVenueSession(connection, client)

    def _require_connection(self, connection: VenueConnection) -> None:
        if connection.adapter_id != self.adapter_id:
            raise ValueError("OKX adapter requires connection adapter_id=okx")
        self._require_environment(connection.environment)

    def _require_environment(self, environment: str) -> None:
        if environment not in self._ENVIRONMENTS:
            raise ValueError(f"unsupported OKX environment: {environment}")


def create_adapter() -> OkxVenueAdapter:
    return OkxVenueAdapter()
