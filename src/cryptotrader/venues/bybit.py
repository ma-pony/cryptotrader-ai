"""Bybit testnet, demo, and live venue adapter."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from cryptotrader.accounts.models import (
    AccountOrder,
    AccountPosition,
    AccountSnapshot,
    Fill,
    FillPage,
    FundingEntry,
    FundingPage,
)
from cryptotrader.configuration.catalog import require_environment
from cryptotrader.venues.account_reads import timestamp
from cryptotrader.venues.ccxt_base import CcxtVenueBase, VenueOperationError, create_async_client
from cryptotrader.venues.models import ACCOUNT_READS, EXIT_OPERATIONS, ProtectionState, VenueCapabilities

if TYPE_CHECKING:
    from collections.abc import Callable

    from cryptotrader.pair import Pair
    from cryptotrader.runtime_config.secrets import CredentialPayload
    from cryptotrader.venues.models import OrderIntent, ProtectionSpec, VenueConnection


class BybitVenueSession(CcxtVenueBase):
    """One isolated Bybit unified-account session."""

    def __init__(self, connection: VenueConnection, client: Any, capabilities: VenueCapabilities) -> None:
        super().__init__(connection, client, capabilities)
        self._protection_pairs: dict[str, tuple[Pair, int]] = {}
        self._account_protection_orders: dict[str, Pair] = {}
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

    async def _account_page(self, method, params):
        response = await self._call("read Bybit account", method, params)
        if not isinstance(response, dict) or str(response.get("retCode")) != "0":
            raise VenueOperationError("Bybit account read rejected")
        result = response.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("list"), list):
            raise VenueOperationError("invalid Bybit account response")
        return result["list"], result.get("nextPageCursor") or None

    async def _all_account_rows(self, method, params):
        rows, cursor, seen = [], None, set()
        while True:
            page, following = await self._account_page(method, {**params, **({"cursor": cursor} if cursor else {})})
            rows.extend(page)
            if not following:
                return rows
            if following in seen:
                raise VenueOperationError("Bybit account cursor did not advance")
            seen.add(following)
            cursor = following

    async def fetch_account(self) -> AccountSnapshot:
        await self._ensure_markets()
        raw = await self._call("fetch Bybit balance", self._client.fetch_balance)
        accounts = raw.get("info", {}).get("result", {}).get("list", [])
        account = accounts[0] if accounts else {}
        equity = self._money(account.get("totalEquity"), "USD", "total_equity_unavailable")
        used = self._money(account.get("totalInitialMargin"), "USD", "initial_margin_unavailable")
        available = self._money(account.get("totalAvailableBalance"), "USD", "available_margin_unavailable")
        balances = tuple(
            self._money(value, asset, "balance_unavailable") for asset, value in raw.get("total", {}).items()
        )
        scopes = (
            {"category": "linear", "settleCoin": "USDT"},
            {"category": "linear", "settleCoin": "USDC"},
            {"category": "inverse"},
            {"category": "option"},
        )
        positions = []
        for scope in scopes:
            rows = await self._all_account_rows(self._client.private_get_v5_position_list, {**scope, "limit": 200})
            for row in rows:
                instrument = self._read_instrument(row.get("symbol"), scope["category"])
                amount = self._read_amount(row.get("size"), instrument)
                if amount == 0:
                    continue
                sign = Decimal("-1") if row.get("side") == "Sell" else Decimal("1")
                currency = self._settlement(instrument) or scope.get("settleCoin")
                positions.append(
                    AccountPosition(
                        instrument,
                        abs(amount) * sign,
                        None,
                        self._signed_money(
                            self._money(row.get("positionValue"), currency, "current_notional_unavailable"), sign
                        ),
                        self._optional_decimal(row.get("avgPrice")),
                        self._money(row.get("unrealisedPnl"), currency, "unrealized_pnl_unavailable"),
                    )
                )
        orders, now = [], datetime.now(UTC)
        for scope in ({"category": "spot"}, *scopes):
            rows = await self._all_account_rows(
                self._client.private_get_v5_order_realtime, {**scope, "openOnly": 0, "limit": 50}
            )
            for row in rows:
                instrument = self._read_instrument(row.get("symbol"), scope["category"])
                protection = row.get("stopOrderType") in {
                    "StopLoss",
                    "TakeProfit",
                    "PartialStopLoss",
                    "PartialTakeProfit",
                    "TrailingStop",
                    "tpslOrder",
                    "OcoOrder",
                    "BidirectionalTpslOrder",
                } or (
                    row.get("stopOrderType") == "Stop"
                    and (self._boolean(row.get("reduceOnly")) or self._boolean(row.get("closeOnTrigger")))
                )
                orders.append(
                    AccountOrder(
                        self.connection_id,
                        str(row["orderId"]),
                        instrument,
                        row.get("side", "unknown").lower(),
                        row.get("orderType", "unknown").lower(),
                        self._read_amount(row["qty"], instrument),
                        self._read_amount(row["cumExecQty"], instrument),
                        self._optional_decimal(row.get("avgPrice")),
                        "open"
                        if row.get("orderStatus") in {"New", "PartiallyFilled", "Untriggered"}
                        else row.get("orderStatus", "unknown"),
                        self._boolean(row.get("reduceOnly")),
                        protection,
                        row.get("orderLinkId") or None,
                        now,
                        self._remaining_notional(
                            instrument,
                            row["qty"],
                            row["cumExecQty"],
                            row.get("price"),
                            explicit_value=row.get("leavesValue"),
                        ),
                    )
                )
                if protection and instrument.tradable:
                    self._account_protection_orders[str(row["orderId"])] = instrument.pair
        return AccountSnapshot(
            self.connection_id,
            now,
            require_environment(self.connection.adapter_id, self.connection.environment).capital_scope,
            equity,
            balances,
            tuple(positions),
            tuple(orders),
            used,
            available,
            self._completeness(equity, used, available, positions),
        )

    async def fetch_fills(self, cursor: str | None) -> FillPage:
        await self._ensure_markets()
        categories = ("spot", "linear", "inverse", "option")
        state = self._history_state(cursor, "fills", len(categories), 730)
        category = categories[state["scope"]]
        params = {"category": category, "startTime": state["begin"], "endTime": state["end"], "limit": 100}
        if state["after"]:
            params["cursor"] = state["after"]
        # Option execution/list defaults to BTC. Account trade ledger covers all
        # option bases without inventing a scan of every possible instrument.
        if category == "option":
            params.update(type="TRADE", accountType="UNIFIED", limit=50)
            rows, token = await self._account_page(self._client.private_get_v5_account_transaction_log, params)
        else:
            params["execType"] = "Trade"
            rows, token = await self._account_page(self._client.private_get_v5_execution_list, params)
        fills = []
        for row in rows:
            ledger = category == "option"
            if ledger and (row.get("type") != "TRADE" or not row.get("tradeId")):
                continue
            instrument = self._read_instrument(row.get("symbol"), category)
            fills.append(
                Fill(
                    self.connection_id,
                    str(row["tradeId"] if ledger else row["execId"]),
                    str(row["orderId"]),
                    instrument,
                    row["side"].lower(),
                    self._read_amount(row["qty"] if ledger else row["execQty"], instrument),
                    self._decimal(row["tradePrice"] if ledger else row["execPrice"], "fill price"),
                    timestamp(row["transactionTime"] if ledger else row["execTime"]),
                    self._money(
                        row.get("fee") if ledger else row.get("execFee"),
                        row.get("currency") if ledger else row.get("feeCurrency"),
                        "fill_fee_unavailable",
                    ),
                    self._money(
                        None,
                        row.get("currency") if ledger else self._settlement(instrument),
                        "realized_pnl_not_provided",
                    ),
                    "platform",
                    row.get("orderLinkId") or None,
                )
            )
        return self._history_page(FillPage, fills, state, token, len(categories))

    async def fetch_funding(self, cursor: str | None) -> FundingPage:
        await self._ensure_markets()
        state = self._history_state(cursor, "funding", 1, 730)
        params = {
            "accountType": "UNIFIED",
            "type": "SETTLEMENT",
            "startTime": state["begin"],
            "endTime": state["end"],
            "limit": 50,
        }
        if state["after"]:
            params["cursor"] = state["after"]
        rows, token = await self._account_page(self._client.private_get_v5_account_transaction_log, params)
        entries = tuple(
            FundingEntry(
                self.connection_id,
                str(row["id"]),
                self._read_instrument(row.get("symbol"), row.get("category", "")),
                self._money(row.get("funding"), row.get("currency"), "funding_amount_unavailable"),
                timestamp(row["transactionTime"]),
            )
            for row in rows
            if row.get("type") == "SETTLEMENT"
        )
        return self._history_page(FundingPage, entries, state, token, 1)

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
            account_pair = self._account_protection_orders.get(protection_id)
            if account_pair is not None:
                await self.cancel_order(protection_id, account_pair)
                self._account_protection_orders.pop(protection_id, None)
                continue
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
            account_reads=ACCOUNT_READS,
            exit_operations=EXIT_OPERATIONS,
            history_initial_days=7,
        )

    async def connect(
        self,
        connection: VenueConnection,
        credentials: CredentialPayload | None,
    ) -> BybitVenueSession:
        self._require_connection(connection)
        if credentials is None or not credentials.values.get("api_key") or not credentials.values.get("secret"):
            raise ValueError("Bybit connection requires API key and secret")
        config = {
            "apiKey": credentials.values["api_key"].get_secret_value(),
            "secret": credentials.values["secret"].get_secret_value(),
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
