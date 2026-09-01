"""Offline raw CCXT responses, shaped after the linked official v5 docs."""

from copy import deepcopy

from tests.fakes.ccxt_client import FakeCcxtClient


class AccountReadClient(FakeCcxtClient):
    def __init__(self, adapter_id, config):
        super().__init__(adapter_id, config)
        eth = deepcopy(self.markets["BTC/USDT:USDT"])
        eth.update(id="ETHUSDT" if adapter_id == "bybit" else "ETH-USDT-SWAP", symbol="ETH/USDT:USDT")
        self.markets["ETH/USDT:USDT"] = eth
        self.history_calls = []

    async def private_get_account_positions(self, params):
        self.calls.append(("account_positions", params))
        return {
            "code": "0",
            "data": [
                {
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "pos": "2",
                    "posSide": "long",
                    "availPos": "1",
                    "notionalUsd": "1000",
                    "avgPx": "49000",
                    "upl": "20",
                    "ccy": "USDT",
                },
                {
                    "instId": "ETH-USDT-SWAP",
                    "instType": "SWAP",
                    "pos": "3",
                    "posSide": "short",
                    "availPos": "3",
                    "notionalUsd": "90",
                    "avgPx": "3100",
                    "upl": "3",
                    "ccy": "USDT",
                },
                {
                    "instId": "UNKNOWN",
                    "instType": "SWAP",
                    "pos": "1",
                    "posSide": "long",
                    "availPos": "",
                    "notionalUsd": "",
                    "avgPx": "",
                    "upl": "",
                    "ccy": "",
                },
            ],
        }

    async def private_get_trade_orders_pending(self, params):
        self.calls.append(("account_orders", params))
        return {
            "code": "0",
            "data": [
                {
                    "ordId": "manual",
                    "instId": "UNKNOWN",
                    "instType": "SWAP",
                    "side": "sell",
                    "ordType": "limit",
                    "sz": "1",
                    "accFillSz": "0",
                    "avgPx": "",
                    "state": "live",
                    "reduceOnly": "false",
                    "clOrdId": "manual1",
                },
            ]
            if "after" not in params
            else [],
        }

    async def private_get_trade_orders_algo_pending(self, params):
        self.calls.append(("account_protections", params))
        return {
            "code": "0",
            "data": [
                {
                    "algoId": "protect1",
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "side": "sell",
                    "ordType": "oco",
                    "sz": "2",
                    "actualSz": "0",
                    "actualPx": "",
                    "state": "live",
                    "reduceOnly": "true",
                    "algoClOrdId": "owned1",
                    "posSide": "long",
                },
            ]
            if params["ordType"] == "oco" and "after" not in params
            else [],
        }

    async def private_get_trade_fills_history(self, params):
        self.history_calls.append(("fills", deepcopy(params)))
        if params["instType"] != "SWAP":
            return {"code": "0", "data": []}
        after = params.get("after")
        rows = (
            []
            if after == "100"
            else [
                {
                    "billId": "100" if after else "101",
                    "tradeId": "trade2" if after else "trade1",
                    "ordId": "order1",
                    "clOrdId": "owned1",
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "side": "sell",
                    "fillSz": "2",
                    "fillPx": "50000",
                    "fillTime": "1700000000000",
                    "ts": "1700000000001",
                    "fee": "-0.5",
                    "feeCcy": "USDT",
                    "fillPnl": "20",
                }
            ]
        )
        return {"code": "0", "data": rows}

    async def private_get_account_bills_archive(self, params):
        self.history_calls.append(("funding", deepcopy(params)))
        rows = (
            []
            if params.get("after")
            else [
                {
                    "billId": "fund1",
                    "instId": "BTC-USDT-SWAP",
                    "instType": "SWAP",
                    "type": "8",
                    "ccy": "USDT",
                    "balChg": "-2",
                    "posBalChg": "0",
                    "ts": "1700000000000",
                },
                {
                    "billId": "fund2",
                    "instId": "ETH-USD-SWAP",
                    "instType": "SWAP",
                    "type": "8",
                    "ccy": "ETH",
                    "balChg": "0",
                    "posBalChg": "0.1",
                    "ts": "1700000000000",
                },
            ]
        )
        return {"code": "0", "data": rows}

    async def private_get_v5_position_list(self, params):
        self.calls.append(("account_positions", deepcopy(params)))
        rows = []
        token = ""
        if params["category"] == "linear" and params["settleCoin"] == "USDT":
            if not params.get("cursor"):
                rows = [
                    {
                        "symbol": "BTCUSDT",
                        "size": "0.02",
                        "side": "Buy",
                        "avgPrice": "49000",
                        "positionValue": "1000",
                        "unrealisedPnl": "20",
                        "positionIdx": 1,
                    }
                ]
                token = "position-page-2"
            else:
                rows = [
                    {
                        "symbol": "ETHUSDT",
                        "size": "0.03",
                        "side": "Sell",
                        "avgPrice": "3100",
                        "positionValue": "90",
                        "unrealisedPnl": "3",
                        "positionIdx": 2,
                    }
                ]
        if params["category"] == "inverse":
            rows = [
                {
                    "symbol": "UNKNOWN",
                    "size": "1",
                    "side": "Buy",
                    "avgPrice": "",
                    "positionValue": "",
                    "unrealisedPnl": "",
                    "positionIdx": 0,
                }
            ]
        return {"retCode": 0, "result": {"list": rows, "nextPageCursor": token}}

    async def private_get_v5_order_realtime(self, params):
        self.calls.append(("account_orders", deepcopy(params)))
        rows = []
        token = ""
        if params["category"] == "linear" and params["settleCoin"] == "USDT":
            protection = bool(params.get("cursor"))
            token = "" if protection else "orders-page-2"
            rows = [
                {
                    "orderId": "protect1" if protection else "manual",
                    "symbol": "BTCUSDT" if protection else "UNKNOWN",
                    "side": "Sell",
                    "orderType": "Market" if protection else "Limit",
                    "qty": "0.02",
                    "cumExecQty": "0",
                    "avgPrice": "",
                    "orderStatus": "Untriggered" if protection else "New",
                    "reduceOnly": protection,
                    "stopOrderType": "StopLoss" if protection else "UNKNOWN",
                    "orderLinkId": "owned1" if protection else "manual1",
                }
            ]
        return {"retCode": 0, "result": {"list": rows, "nextPageCursor": token}}

    async def private_get_v5_execution_list(self, params):
        self.history_calls.append(("fills", deepcopy(params)))
        rows, token = [], ""
        if params["category"] == "linear":
            second = bool(params.get("cursor"))
            token = "" if second else "same-millisecond-next"
            rows = [
                {
                    "execId": "trade2" if second else "trade1",
                    "orderId": "order1",
                    "orderLinkId": "owned1",
                    "symbol": "BTCUSDT",
                    "side": "Sell",
                    "execQty": "0.02",
                    "execPrice": "50000",
                    "execTime": "1700000000000",
                    "execFee": "0.5",
                    "feeCurrency": "USDT",
                    "execType": "Trade",
                }
            ]
        return {"retCode": 0, "result": {"list": rows, "nextPageCursor": token}}

    async def private_get_v5_account_transaction_log(self, params):
        self.history_calls.append(("ledger", deepcopy(params)))
        if params.get("type") == "TRADE":
            rows = [
                {
                    "id": "option-entry",
                    "tradeId": "eth-option-fill",
                    "orderId": "option-order",
                    "symbol": "ETH-25DEC26-3000-C",
                    "category": "option",
                    "type": "TRADE",
                    "currency": "USDC",
                    "qty": "1",
                    "side": "Buy",
                    "tradePrice": "10",
                    "fee": "0.2",
                    "cashFlow": "0",
                    "transactionTime": "1700000000000",
                    "orderLinkId": "",
                }
            ]
        else:
            rows = [
                {
                    "id": "fund1",
                    "symbol": "BTCUSDT",
                    "category": "linear",
                    "type": "SETTLEMENT",
                    "funding": "-2",
                    "currency": "USDT",
                    "transactionTime": "1700000000000",
                },
                {
                    "id": "fund2",
                    "symbol": "ETHPERP",
                    "category": "linear",
                    "type": "SETTLEMENT",
                    "funding": "0.1",
                    "currency": "USDC",
                    "transactionTime": "1700000000000",
                },
            ]
        return {"retCode": 0, "result": {"list": rows, "nextPageCursor": ""}}


async def account_session(adapter_id):
    from cryptotrader.runtime_config.secrets import CredentialPayload
    from cryptotrader.venues.bybit import BybitVenueAdapter
    from cryptotrader.venues.okx import OkxVenueAdapter
    from tests.factories.runtime_config import connection

    client = AccountReadClient(adapter_id, {})
    adapter = (OkxVenueAdapter if adapter_id == "okx" else BybitVenueAdapter)(client_factory=lambda _: client)
    session = await adapter.connect(
        connection(adapter_id, "demo", adapter_id=adapter_id, credential_ref="fixture"),
        CredentialPayload(
            values={
                "api_key": "fixture",  # pragma: allowlist secret
                "secret": "fixture",  # pragma: allowlist secret
                "passphrase": "fixture",  # pragma: allowlist secret
            }
        ),
    )
    client.calls.clear()
    return session, client
