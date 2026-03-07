"""Backtest engine — steps through historical data and runs the full graph."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from cryptotrader.backtest.cache import fetch_historical
from cryptotrader.backtest.result import BacktestResult
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData

logger = logging.getLogger(__name__)

_TF_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class BacktestEngine:
    def __init__(
        self,
        pair: str,
        start: str,
        end: str,
        interval: str = "4h",
        initial_capital: float = 10000,
        use_llm: bool = True,
        slippage_bps: float = 5.0,
        fee_bps: float = 10.0,
        position_pct: float = 0.1,
    ):
        self.pair = pair
        self.start_ms = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000)
        self.end_ms = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000)
        self.interval = interval
        self.capital = initial_capital
        self.use_llm = use_llm
        self.slippage_bps = slippage_bps  # basis points
        self.fee_bps = fee_bps  # basis points
        self.position_pct = position_pct
        # Cache config once to avoid re-parsing TOML per candle
        self._config = None
        # LLM usage tracking
        self._llm_calls = 0
        self._llm_tokens = 0

    @property
    def _cached_config(self):
        if self._config is None:
            from cryptotrader.config import load_config

            self._config = load_config()
        return self._config

    def _apply_costs(self, price: float, side: str) -> float:
        """Apply slippage and fees to get realistic fill price."""
        slip = price * self.slippage_bps / 10000
        fee = price * self.fee_bps / 10000
        if side == "buy":
            return price + slip + fee
        return price - slip - fee

    def _execute_pending_action(
        self, pending_action: str, position: float, entry_price: float, equity: float, exec_price: float, ts: int
    ) -> tuple[float, float, float, list[dict]]:
        """Execute pending action and return updated position, entry_price, equity, and new trades."""
        trades = []
        if pending_action == "long" and position <= 0:
            if position < 0:  # close short
                fill = self._apply_costs(exec_price, "buy")
                pnl = (entry_price - fill) * abs(position)
                equity += pnl
                trades.append({"side": "close_short", "price": fill, "pnl": pnl, "ts": ts})
            fill = self._apply_costs(exec_price, "buy")
            size = equity * self.position_pct / fill
            position = size
            entry_price = fill
            trades.append({"side": "buy", "price": fill, "amount": size, "ts": ts})
        elif pending_action == "short" and position >= 0:
            if position > 0:  # close long
                fill = self._apply_costs(exec_price, "sell")
                pnl = (fill - entry_price) * position
                equity += pnl
                trades.append({"side": "close_long", "price": fill, "pnl": pnl, "ts": ts})
            fill = self._apply_costs(exec_price, "sell")
            size = equity * self.position_pct / fill
            position = -size
            entry_price = fill
            trades.append({"side": "sell", "price": fill, "amount": size, "ts": ts})
        return position, entry_price, equity, trades

    def _mark_to_market(self, position: float, equity: float, entry_price: float, current_price: float) -> float:
        """Calculate mark-to-market equity."""
        if position > 0:
            return equity + (current_price - entry_price) * position
        elif position < 0:
            return equity + (entry_price - current_price) * abs(position)
        return equity

    def _close_final_position(
        self, position: float, entry_price: float, equity: float, final_price: float, ts: int
    ) -> tuple[float, list[dict]]:
        """Close any open position at end of backtest."""
        trades = []
        if position != 0 and final_price and final_price > 0:
            if position > 0:
                fill = self._apply_costs(final_price, "sell")
                pnl = (fill - entry_price) * position
            else:
                fill = self._apply_costs(final_price, "buy")
                pnl = (entry_price - fill) * abs(position)
            equity += pnl
            trades.append({"side": "close", "price": fill, "pnl": pnl, "ts": ts})
        return equity, trades

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

        from cryptotrader.graph import build_lite_graph

        graph = build_lite_graph() if self.use_llm else None

        lookback = 100

        pending_action: str | None = None

        for i in range(lookback, len(candles)):
            window = candles[max(0, i - lookback) : i + 1]
            cur = candles[i]
            ts, o, c = cur[0], cur[1], cur[4]

            # Skip candles with invalid data
            if c is None or c <= 0:
                continue

            # Execute pending action from PREVIOUS bar's signal at current bar's open
            # This eliminates look-ahead bias: signal on bar[i-1], fill on bar[i] open
            if pending_action is not None:
                exec_price = o if (o is not None and o > 0) else c
                position, entry_price, equity, new_trades = self._execute_pending_action(
                    pending_action, position, entry_price, equity, exec_price, ts
                )
                trades.extend(new_trades)
                pending_action = None

            # Generate signal on current bar (will be executed on NEXT bar)
            if graph and self.use_llm:
                snapshot = self._build_snapshot(window, ts)
                result = await self._run_graph(graph, snapshot)
                action = result.get("data", {}).get("verdict", {}).get("action", "hold")
            else:
                action = self._simple_signal(window)

            if action != "hold":
                pending_action = action

            # Mark to market at close
            mtm = self._mark_to_market(position, equity, entry_price, c)
            equity_curve.append(mtm)
            peak = max(peak, mtm)

        # Close any open position at end
        equity, final_trades = self._close_final_position(position, entry_price, equity, candles[-1][4], candles[-1][0])
        trades.extend(final_trades)

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
                pair=self.pair,
                ohlcv=df,
                ticker={"last": cur[4], "baseVolume": cur[5]},
                funding_rate=0.0,
                orderbook_imbalance=0.0,
                volatility=float(vol) if not pd.isna(vol := df["close"].pct_change().std()) else 0.0,
            ),
            onchain=OnchainData(),
            news=NewsSentiment(),
            macro=MacroData(),
        )

    async def _run_graph(self, graph, snapshot: DataSnapshot) -> dict:
        import litellm

        config = self._cached_config

        # Track LLM usage via litellm callbacks
        if not getattr(litellm, "_backtest_hooks_installed", False):
            original_success = litellm.success_callback

            def _track_usage(_kwargs, completion_response, _start_time, _end_time):
                usage = getattr(completion_response, "usage", None)
                if usage:
                    litellm._backtest_token_count = getattr(litellm, "_backtest_token_count", 0) + (
                        usage.total_tokens or 0
                    )
                litellm._backtest_call_count = getattr(litellm, "_backtest_call_count", 0) + 1

            litellm.success_callback = [*original_success, _track_usage] if original_success else [_track_usage]
            litellm._backtest_hooks_installed = True

        initial: dict[str, Any] = {
            "messages": [],
            "data": {"snapshot": snapshot},
            "metadata": {
                "pair": self.pair,
                "engine": "paper",
                "models": {
                    "tech_agent": config.models.tech_agent,
                    "chain_agent": config.models.chain_agent,
                    "news_agent": config.models.news_agent,
                    "macro_agent": config.models.macro_agent,
                },
                "analysis_model": config.models.analysis,
                "debate_model": config.models.debate,
                "verdict_model": config.models.verdict,
                # Use rules verdict in backtest to avoid LLM cost per candle
                "llm_verdict": False,
            },
            "debate_round": 0,
            "max_debate_rounds": 2,
            "divergence_scores": [],
        }
        result = await graph.ainvoke(initial)

        self._llm_calls = getattr(litellm, "_backtest_call_count", 0)
        self._llm_tokens = getattr(litellm, "_backtest_token_count", 0)

        return result

    def _compute_result(self, equity: float, curve: list[float], trades: list[dict]) -> BacktestResult:
        total_return = (equity - self.capital) / self.capital
        # Sharpe
        if len(curve) > 1:
            returns = [(curve[i] - curve[i - 1]) / curve[i - 1] for i in range(1, len(curve)) if curve[i - 1] > 0]
            if returns:
                avg = sum(returns) / len(returns)
                std = (sum((r - avg) ** 2 for r in returns) / len(returns)) ** 0.5
                # Crypto trades 365 days/year; annualize based on interval
                periods_per_day = 86_400_000 / _TF_MS.get(self.interval, 3_600_000)
                annualization = math.sqrt(365 * periods_per_day)
                sharpe = (avg / std * annualization) if std > 0 else 0.0
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
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            trades=trades,
            equity_curve=curve,
            llm_calls=self._llm_calls,
            llm_tokens=self._llm_tokens,
        )
