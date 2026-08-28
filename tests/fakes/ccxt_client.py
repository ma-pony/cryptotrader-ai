"""Deterministic CCXT doubles that mirror unified and venue-native fields."""

from __future__ import annotations

from copy import deepcopy
from decimal import ROUND_DOWN, Decimal
from typing import Any


class FakeCcxtClient:
    def __init__(self, exchange_id: str, config: dict[str, Any]) -> None:
        self.id = exchange_id
        self.config = deepcopy(config)
        self.headers: dict[str, str] = {}
        self.urls = {
            "api": dict.fromkeys(("spot", "futures", "v2", "public", "private"), "https://api.bybit.com"),
            "test": dict.fromkeys(("spot", "futures", "v2", "public", "private"), "https://api-testnet.bybit.com"),
            "demotrading": dict.fromkeys(("spot", "futures", "v2", "public", "private"), "https://api-demo.bybit.com"),
        }
        self.markets = {
            "BTC/USDT": {
                "id": "BTCUSDT" if exchange_id == "bybit" else "BTC-USDT",
                "symbol": "BTC/USDT",
                "spot": True,
                "swap": False,
                "contractSize": None,
                "precision": {"amount": "0.001", "price": "0.1"},
            },
            "BTC/USDT:USDT": {
                "id": "BTCUSDT" if exchange_id == "bybit" else "BTC-USDT-SWAP",
                "symbol": "BTC/USDT:USDT",
                "spot": False,
                "swap": True,
                "linear": True,
                "contractSize": "0.01",
                "precision": {"amount": "1", "price": "0.1"},
            },
        }
        self.calls: list[tuple[str, Any]] = []
        self.load_markets_calls = 0
        self.close_calls = 0
        self._next_order = 1
        self._next_algo = 1
        self._open_orders: list[dict[str, Any]] = [
            {
                "id": "open-1",
                "clientOrderId": None,
                "symbol": "BTC/USDT:USDT",
                "type": "limit",
                "side": "buy",
                "amount": "3",
                "filled": "1",
                "remaining": "2",
                "average": "49900",
                "price": "49900",
                "status": "open",
                "reduceOnly": False,
                "info": {"orderId": "open-1", "reduceOnly": False},
            }
        ]
        self._okx_algos: list[dict[str, Any]] = []
        self._bybit_stop_loss = "0"
        self._bybit_take_profit = "0"
        self.position_contracts = "2"
        self.position_index = 1
        self.position_side = "long"
        self.confirm_protection = True

    def set_sandbox_mode(self, enabled: bool) -> None:
        self.calls.append(("set_sandbox_mode", enabled))
        if self.id == "okx":
            if enabled:
                self.headers["x-simulated-trading"] = "1"
            else:
                self.headers.pop("x-simulated-trading", None)
        elif enabled:
            self.urls["api"] = deepcopy(self.urls["test"])

    def enable_demo_trading(self, enabled: bool) -> None:
        self.calls.append(("enable_demo_trading", enabled))
        if enabled:
            self.urls["api"] = deepcopy(self.urls["demotrading"])

    async def load_markets(self) -> dict[str, dict[str, Any]]:
        self.load_markets_calls += 1
        self.calls.append(("load_markets", None))
        return self.markets

    def market(self, symbol: str) -> dict[str, Any]:
        return self.markets[symbol]

    def amount_to_precision(self, symbol: str, amount: Any) -> str:
        return self._to_precision(symbol, amount, "amount")

    def price_to_precision(self, symbol: str, price: Any) -> str:
        return self._to_precision(symbol, price, "price")

    def _to_precision(self, symbol: str, value: Any, field: str) -> str:
        step = Decimal(self.markets[symbol]["precision"][field])
        normalized = Decimal(str(value)).quantize(step, rounding=ROUND_DOWN)
        return format(normalized, "f")

    async def fetch_balance(self) -> dict[str, Any]:
        self.calls.append(("fetch_balance", None))
        return {
            "total": {"USDT": "10000.50", "BTC": "0.25"},
            "free": {"USDT": "9000.25", "BTC": "0.20"},
            "used": {"USDT": "1000.25", "BTC": "0.05"},
            "info": {"accountType": "UNIFIED" if self.id == "bybit" else "18"},
        }

    async def fetch_positions(self, symbols: list[str] | None = None) -> list[dict[str, Any]]:
        self.calls.append(("fetch_positions", deepcopy(symbols)))
        symbol = (symbols or ["BTC/USDT:USDT"])[0]
        if symbol == "BTC/USDT":
            return []
        info = (
            {"instId": "BTC-USDT-SWAP", "posSide": "long", "pos": "2", "avgPx": "50000"}
            if self.id == "okx"
            else {
                "symbol": "BTCUSDT",
                "side": "Buy" if self.position_side == "long" else "Sell",
                "size": self.position_contracts,
                "avgPrice": "50000",
                "positionIdx": self.position_index,
                "stopLoss": self._bybit_stop_loss,
                "takeProfit": self._bybit_take_profit,
            }
        )
        return [
            {
                "symbol": symbol,
                "contracts": self.position_contracts,
                "contractSize": "0.01",
                "side": self.position_side,
                "entryPrice": "50000",
                "notional": "1000",
                "unrealizedPnl": "12.5",
                "info": info,
            }
        ]

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("fetch_ticker", symbol))
        return {
            "symbol": symbol,
            "bid": "49999.9",
            "ask": "50000.1",
            "last": "50000.0",
            "info": {"instId": self.markets[symbol]["id"]},
        }

    async def create_order(self, symbol, order_type, side, amount, price, params):
        payload = (symbol, order_type, side, amount, price, deepcopy(params))
        self.calls.append(("create_order", payload))
        order_id = f"order-{self._next_order}"
        self._next_order += 1
        return {
            "id": order_id,
            "clientOrderId": None,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": str(amount),
            "filled": str(amount),
            "remaining": "0",
            "average": str(price or "50000"),
            "price": str(price) if price is not None else None,
            "status": "closed",
            "reduceOnly": bool(params.get("reduceOnly")),
            "info": {"ordId": order_id, "reduceOnly": params.get("reduceOnly", False)},
        }

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        self.calls.append(("fetch_open_orders", symbol))
        return deepcopy([row for row in self._open_orders if symbol is None or row["symbol"] == symbol])

    async def private_post_trade_order_algo(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("private_post_trade_order_algo", deepcopy(params)))
        algo_id = f"algo-{self._next_algo}"
        self._next_algo += 1
        row = {
            "algoId": algo_id,
            "instId": params["instId"],
            "ordType": "oco",
            "posSide": params["posSide"],
            "sz": params["sz"],
            "slTriggerPx": params.get("slTriggerPx", ""),
            "tpTriggerPx": params.get("tpTriggerPx", ""),
            "state": "effective",
        }
        if self.confirm_protection:
            self._okx_algos.append(row)
        return {"code": "0", "msg": "", "data": [{"algoId": algo_id, "sCode": "0", "sMsg": ""}]}

    async def private_get_trade_orders_algo_pending(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("private_get_trade_orders_algo_pending", deepcopy(params)))
        rows = self._okx_algos
        if "instId" in params:
            rows = [row for row in rows if row["instId"] == params["instId"]]
        return {"code": "0", "msg": "", "data": deepcopy(rows)}

    async def private_post_trade_cancel_algos(self, params: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append(("private_post_trade_cancel_algos", deepcopy(params)))
        ids = {row["algoId"] for row in params}
        self._okx_algos = [row for row in self._okx_algos if row["algoId"] not in ids]
        return {"code": "0", "msg": "", "data": [{"algoId": item, "sCode": "0", "sMsg": ""} for item in ids]}

    async def private_post_v5_position_trading_stop(self, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("private_post_v5_position_trading_stop", deepcopy(params)))
        if self.confirm_protection:
            self._bybit_stop_loss = params.get("stopLoss", self._bybit_stop_loss)
            self._bybit_take_profit = params.get("takeProfit", self._bybit_take_profit)
        return {"retCode": 0, "retMsg": "OK", "result": {}, "retExtInfo": {}, "time": 1770000000000}

    async def close(self) -> None:
        self.close_calls += 1
        self.calls.append(("close", None))


class FakeCcxtFactory:
    def __init__(self, exchange_id: str) -> None:
        self.exchange_id = exchange_id
        self.configs: list[dict[str, Any]] = []
        self.clients: list[FakeCcxtClient] = []

    def __call__(self, config: dict[str, Any]) -> FakeCcxtClient:
        self.configs.append(deepcopy(config))
        client = FakeCcxtClient(self.exchange_id, config)
        self.clients.append(client)
        return client
