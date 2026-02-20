"""Backtest engine — steps through historical data and runs the full graph."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cryptotrader.backtest.cache import fetch_historical
from cryptotrader.backtest.result import BacktestResult
from cryptotrader.models import DataSnapshot, MarketData, OnchainData, NewsSentiment, MacroData

_TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


class BacktestEngine:
    def __init__(
        self, pair: str, start: str, end: str,
        interval: str = "4h", initial_capital: float = 10000,
        use_llm: bool = False,
    ):
        self.pair = pair
        self.start_ms = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000)
        self.end_ms = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000)
        self.interval = interval
        self.capital = initial_capital
        self.use_llm = use_llm

    async def run(self) -> BacktestResult:
        candles = await fetch_historical(self.pair, self.interval, self.start_ms, self.end_ms)
        if not candles:
            return BacktestResult()

        equity = self.capital
        position = 0.0
        entry_price = 0.0
        equity_curve = [equity]
        trades: list[dict] = []
        peak = equity

        from cryptotrader.graph import build_lite_graph, ArenaState

        graph = build_lite_graph() if self.use_llm else None

        step_ms = _TF_MS.get(self.interval, 3_600_000)
        lookback = 100

        for i in range(lookback, len(candles)):
            window = candles[max(0, i - lookback):i + 1]
            cur = candles[i]
            ts, o, h, l, c, v = cur[0], cur[1], cur[2], cur[3], cur[4], cur[5]

            if graph and self.use_llm:
                snapshot = self._build_snapshot(window, ts)
                result = await self._run_graph(graph, snapshot)
                action = result.get("data", {}).get("verdict", {}).get("action", "hold")
            else:
                action = self._simple_signal(window)

            # Execute trades
            if action == "long" and position <= 0:
                if position < 0:  # close short
                    pnl = (entry_price - c) * abs(position)
                    equity += pnl
                    trades.append({"side": "close_short", "price": c, "pnl": pnl, "ts": ts})
                size = equity * 0.1 / c
                position = size
                entry_price = c
                trades.append({"side": "buy", "price": c, "amount": size, "ts": ts})

            elif action == "short" and position >= 0:
                if position > 0:  # close long
                    pnl = (c - entry_price) * position
                    equity += pnl
                    trades.append({"side": "close_long", "price": c, "pnl": pnl, "ts": ts})
                size = equity * 0.1 / c
                position = -size
                entry_price = c
                trades.append({"side": "sell", "price": c, "amount": size, "ts": ts})

            # Mark to market
            if position > 0:
                mtm = equity + (c - entry_price) * position
            elif position < 0:
                mtm = equity + (entry_price - c) * abs(position)
            else:
                mtm = equity
            equity_curve.append(mtm)
            peak = max(peak, mtm)

        # Close any open position at end
        if position != 0:
            final_price = candles[-1][4]
            if position > 0:
                pnl = (final_price - entry_price) * position
            else:
                pnl = (entry_price - final_price) * abs(position)
            equity += pnl
            trades.append({"side": "close", "price": final_price, "pnl": pnl, "ts": candles[-1][0]})

        return self._compute_result(equity, equity_curve, trades)

    def _simple_signal(self, window: list[list]) -> str:
        closes = [c[4] for c in window]
        if len(closes) < 20:
            return "hold"
        sma20 = sum(closes[-20:]) / 20
        sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma20
        if closes[-1] > sma20 > sma50:
            return "long"
        elif closes[-1] < sma20 < sma50:
            return "short"
        return "hold"

    def _build_snapshot(self, window: list[list], ts: int) -> DataSnapshot:
        df = pd.DataFrame(window, columns=["timestamp", "open", "high", "low", "close", "volume"])
        cur = window[-1]
        return DataSnapshot(
            timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC),
            pair=self.pair,
            market=MarketData(
                pair=self.pair, ohlcv=df,
                ticker={"last": cur[4], "baseVolume": cur[5]},
                funding_rate=0.0, orderbook_imbalance=0.0,
                volatility=df["close"].pct_change().std() or 0.0,
            ),
            onchain=OnchainData(), news=NewsSentiment(), macro=MacroData(),
        )

    async def _run_graph(self, graph, snapshot: DataSnapshot) -> dict:
        initial: dict[str, Any] = {
            "messages": [], "data": {"snapshot": snapshot},
            "metadata": {
                "pair": self.pair, "engine": "paper",
                "models": {
                    "tech_agent": "openai/deepseek-chat",
                    "chain_agent": "openai/deepseek-chat",
                    "news_agent": "openai/deepseek-chat",
                    "macro_agent": "openai/deepseek-chat",
                },
                "debate_model": "openai/deepseek-chat",
            },
            "debate_round": 0, "max_debate_rounds": 2, "divergence_scores": [],
        }
        return await graph.ainvoke(initial)

    def _compute_result(self, equity: float, curve: list[float], trades: list[dict]) -> BacktestResult:
        total_return = (equity - self.capital) / self.capital
        # Sharpe
        if len(curve) > 1:
            returns = [(curve[i] - curve[i-1]) / curve[i-1] for i in range(1, len(curve)) if curve[i-1] > 0]
            if returns:
                avg = sum(returns) / len(returns)
                std = (sum((r - avg) ** 2 for r in returns) / len(returns)) ** 0.5
                sharpe = (avg / std * math.sqrt(252)) if std > 0 else 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        # Max drawdown
        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            peak = max(peak, v)
            dd = (v - peak) / peak if peak > 0 else 0.0
            max_dd = min(max_dd, dd)
        # Win rate
        pnl_trades = [t for t in trades if "pnl" in t]
        wins = sum(1 for t in pnl_trades if t["pnl"] > 0)
        win_rate = wins / len(pnl_trades) if pnl_trades else 0.0

        return BacktestResult(
            total_return=total_return, sharpe_ratio=sharpe,
            max_drawdown=max_dd, win_rate=win_rate,
            trades=trades, equity_curve=curve,
        )
