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
        lookback: int = 60,
    ):
        self.pair = pair
        self.start = start
        self.end = end
        self.start_ms = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000)
        self.end_ms = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000)
        self.interval = interval
        self.capital = initial_capital
        self.use_llm = use_llm
        self.slippage_bps = slippage_bps  # basis points
        self.fee_bps = fee_bps  # basis points
        self.position_pct = position_pct
        self.lookback = lookback
        # Cache config once to avoid re-parsing TOML per candle
        self._config = None
        # LLM usage tracking
        self._llm_calls = 0
        self._llm_tokens = 0
        # Historical data caches (populated in run())
        self._fng: dict[str, int] = {}
        self._funding: dict[str, float] = {}
        self._btc_dom: dict[str, float] = {}
        self._fed_rate: dict[str, float] = {}
        self._dxy: dict[str, float] = {}
        self._fut_vol: dict[str, dict] = {}
        self._candles: list[list] = []
        # Extended data from unified store
        self._etf_flows: dict[str, dict] = {}
        self._stablecoin_supply: dict[str, float] = {}
        self._btc_hashrate: dict[str, float] = {}
        self._defi_tvl: dict[str, float] = {}
        self._vix: dict[str, float] = {}
        self._sp500: dict[str, float] = {}
        self._oi: dict[str, dict] = {}
        self._ls_ratio: dict[str, dict] = {}

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
        if position < 0:
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

    async def _fetch_historical_data(self) -> None:
        """Pre-fetch all historical data sources for the backtest period."""
        from cryptotrader.backtest.historical_data import (
            fetch_btc_dominance,
            fetch_fear_greed,
            fetch_fred_series,
            fetch_funding_rate,
            fetch_futures_volume,
        )

        self._candles = await fetch_historical(self.pair, self.interval, self.start_ms, self.end_ms)

        symbol = self.pair.split("/")[0]
        logger.info("Fetching historical macro data for %s...", self.pair)

        self._fng = await fetch_fear_greed(self.start, self.end)
        self._funding = await fetch_funding_rate(symbol, self.start, self.end)

        try:
            self._btc_dom = await fetch_btc_dominance(self.start, self.end)
        except Exception:
            logger.warning("BTC dominance fetch failed, using empty")

        try:
            self._fed_rate = await fetch_fred_series("DFF", self.start, self.end)
        except Exception:
            logger.warning("Fed rate fetch failed, using empty")

        try:
            self._dxy = await fetch_fred_series("DTWEXBGS", self.start, self.end)
        except Exception:
            logger.warning("DXY fetch failed, using empty")

        try:
            self._fut_vol = await fetch_futures_volume(symbol, self.start, self.end)
        except Exception:
            logger.warning("Futures volume fetch failed, using empty")

        # Load extended data from unified store (pre-synced via `arena sync`)
        self._load_extended_data()

        logger.info(
            "Historical data: %d candles, %d fng, %d funding, %d btc_dom, %d fed, %d dxy, %d fut_vol",
            len(self._candles),
            len(self._fng),
            len(self._funding),
            len(self._btc_dom),
            len(self._fed_rate),
            len(self._dxy),
            len(self._fut_vol),
        )
        logger.info(
            "Extended data: %d etf_flows, %d stablecoin, %d hashrate, %d tvl, %d vix, %d sp500, %d oi, %d ls_ratio",
            len(self._etf_flows),
            len(self._stablecoin_supply),
            len(self._btc_hashrate),
            len(self._defi_tvl),
            len(self._vix),
            len(self._sp500),
            len(self._oi),
            len(self._ls_ratio),
        )

    @staticmethod
    def _extract_numeric(data, key: str | None = None) -> float:
        """Extract a float value from store data (dict with key, or scalar)."""
        if isinstance(data, dict):
            return float(data.get(key, 0)) if key else 0.0
        if isinstance(data, int | float):
            return float(data)
        return 0.0

    @staticmethod
    def _load_dict_range(source: str, start: str, end: str) -> dict:
        """Load dict-valued records from store, filtering non-dict entries."""
        from cryptotrader.data.store import get_range

        return {date: data for date, data in get_range(source, start, end).items() if isinstance(data, dict)}

    def _load_extended_data(self) -> None:
        """Load pre-synced data from unified SQLite store into memory caches."""
        from cryptotrader.data.store import get_range

        start, end = self.start, self.end

        self._etf_flows = self._load_dict_range("sosovalue_etf", start, end)
        self._oi = self._load_dict_range("binance_oi_BTC", start, end)
        self._ls_ratio = self._load_dict_range("binance_ls_ratio_BTC", start, end)

        for date, data in get_range("stablecoin_total_supply", start, end).items():
            self._stablecoin_supply[date] = self._extract_numeric(data, "total_supply")

        for date, data in get_range("defillama_tvl", start, end).items():
            self._defi_tvl[date] = self._extract_numeric(data, "tvl")

        for source, cache in [
            ("btc_hashrate", self._btc_hashrate),
            ("fred_VIXCLS", self._vix),
            ("fred_SP500", self._sp500),
        ]:
            for date, data in get_range(source, start, end).items():
                cache[date] = self._extract_numeric(data)

    async def run(self) -> BacktestResult:
        await self._fetch_historical_data()
        candles = self._candles
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

        lookback = self.lookback

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
                snapshot = self._build_snapshot(window, ts, i)
                result = await self._run_graph(graph, snapshot)
                verdict = result.get("data", {}).get("verdict", {})
                action = verdict.get("action", "hold")
                confidence = verdict.get("confidence", 0.5)
                # Dynamic position sizing based on AI confidence
                if confidence >= 0.8:
                    self.position_pct = 0.20
                elif confidence >= 0.6:
                    self.position_pct = 0.12
                else:
                    self.position_pct = 0.06
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
        if closes[-1] < sma20 < sma50:
            return "short"
        return "hold"

    def _build_snapshot(self, window: list[list], ts: int, candle_idx: int) -> DataSnapshot:
        from cryptotrader.backtest.historical_data import derive_news_sentiment

        df = pd.DataFrame(window, columns=["timestamp", "open", "high", "low", "close", "volume"])
        cur = window[-1]
        date_str = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d")

        # Historical funding rate
        fr_val = self._funding.get(date_str, 0.0)

        # Fear & Greed
        fng_val = self._fng.get(date_str, 50)

        # BTC dominance, Fed rate, DXY
        dom_val = self._btc_dom.get(date_str, 0.0)
        fed_val = self._fed_rate.get(date_str, 0.0)
        dxy_val = self._dxy.get(date_str, 0.0)

        # Futures volume
        fv = self._fut_vol.get(date_str, {})
        fut_volume = fv.get("volume", 0.0)
        # 20-day average volume ratio
        vol_20d = []
        for j in range(max(0, candle_idx - 20), candle_idx):
            d_j = datetime.fromtimestamp(self._candles[j][0] / 1000, UTC).strftime("%Y-%m-%d")
            vj = self._fut_vol.get(d_j, {}).get("volume", 0)
            if vj > 0:
                vol_20d.append(vj)
        avg_vol = sum(vol_20d) / len(vol_20d) if vol_20d else max(fut_volume, 1)
        vol_ratio = fut_volume / avg_vol if avg_vol > 0 else 1.0

        # News sentiment derived from price action
        sentiment, events = derive_news_sentiment(self._candles, candle_idx)

        # Extended data from unified store
        etf = self._etf_flows.get(date_str, {})
        oi_data = self._oi.get(date_str, {})
        ls_data = self._ls_ratio.get(date_str, {})
        oi_val = oi_data.get("openInterestValue", fut_volume) if oi_data else fut_volume
        defi_tvl = self._defi_tvl.get(date_str, 0.0)
        hashrate = self._btc_hashrate.get(date_str, 0.0)
        stablecoin = self._stablecoin_supply.get(date_str, 0.0)
        vix_val = self._vix.get(date_str, 0.0)
        sp500_val = self._sp500.get(date_str, 0.0)

        return DataSnapshot(
            timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC),
            pair=self.pair,
            market=MarketData(
                pair=self.pair,
                ohlcv=df,
                ticker={"last": cur[4], "baseVolume": cur[5]},
                funding_rate=fr_val,
                orderbook_imbalance=0.0,
                volatility=float(vol) if not pd.isna(vol := df["close"].pct_change().std()) else 0.0,
            ),
            onchain=OnchainData(
                open_interest=oi_val,
                liquidations_24h={
                    "volume_ratio": vol_ratio,
                    "futures_volume": fut_volume,
                    "long_short_ratio": ls_data.get("longShortRatio", 1.0) if ls_data else 1.0,
                },
                defi_tvl=defi_tvl,
                data_quality={
                    "has_oi": bool(oi_data),
                    "has_ls_ratio": bool(ls_data),
                    "has_etf": bool(etf),
                    "has_hashrate": hashrate > 0,
                    "has_stablecoin": stablecoin > 0,
                },
            ),
            news=NewsSentiment(
                sentiment_score=sentiment,
                key_events=events,
                headlines=[f"BTC at ${cur[4]:,.0f}, Fear&Greed={fng_val}"],
            ),
            macro=MacroData(
                fear_greed_index=fng_val,
                btc_dominance=dom_val,
                fed_rate=fed_val,
                dxy=dxy_val,
                etf_daily_net_inflow=etf.get("totalNetInflow", 0.0) if etf else 0.0,
                etf_total_net_assets=etf.get("totalNetAssets", 0.0) if etf else 0.0,
                etf_cum_net_inflow=etf.get("cumNetInflow", 0.0) if etf else 0.0,
                vix=vix_val,
                sp500=sp500_val,
                stablecoin_total_supply=stablecoin,
                btc_hashrate=hashrate,
            ),
        )

    async def _run_graph(self, graph, snapshot: DataSnapshot) -> dict:
        config = self._cached_config

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
        return await graph.ainvoke(initial)

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
