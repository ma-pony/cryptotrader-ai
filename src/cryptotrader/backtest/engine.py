"""Backtest engine — steps through historical data and runs the full graph."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

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
        initial_capital: float | None = None,
        use_llm: bool = True,
        slippage_bps: float | None = None,
        fee_bps: float | None = None,
        position_pct: float | None = None,
        lookback: int | None = None,
    ):
        from cryptotrader.config import load_config

        bt_cfg = load_config().backtest

        self.pair = pair
        self.start = start
        self.end = end
        self.start_ms = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp() * 1000)
        self.end_ms = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp() * 1000)
        self.interval = interval
        self.capital = initial_capital if initial_capital is not None else bt_cfg.initial_capital
        self.use_llm = use_llm
        self.slippage_bps = slippage_bps if slippage_bps is not None else bt_cfg.slippage_base * 10000
        self.fee_bps = fee_bps if fee_bps is not None else bt_cfg.fee_bps
        # Use risk.position.max_single_pct for consistency with live execution
        risk_cfg = load_config().risk.position
        self.position_pct = position_pct if position_pct is not None else risk_cfg.max_single_pct
        self.lookback = lookback if lookback is not None else bt_cfg.lookback
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

    def _close_position(
        self, position: float, entry_price: float, equity: float, exec_price: float, ts: int
    ) -> tuple[float, float, float, list[dict]]:
        """Close current position and return updated state."""
        trades = []
        if position > 0:
            fill = self._apply_costs(exec_price, "sell")
            pnl = (fill - entry_price) * position
            trades.append({"side": "ai_close_long", "price": fill, "pnl": pnl, "ts": ts})
        elif position < 0:
            fill = self._apply_costs(exec_price, "buy")
            pnl = (entry_price - fill) * abs(position)
            trades.append({"side": "ai_close_short", "price": fill, "pnl": pnl, "ts": ts})
        else:
            return 0.0, 0.0, equity, trades
        return 0.0, 0.0, equity + pnl, trades

    def _open_or_add_long(
        self, position: float, entry_price: float, equity: float, exec_price: float, ts: int, scale: float
    ) -> tuple[float, float, list[dict]]:
        """Open new long or add to existing long position."""
        fill = self._apply_costs(exec_price, "buy")
        target_size = equity * self.position_pct * scale / fill
        if target_size <= position + 1e-12:
            return position, entry_price, []
        delta = target_size - position
        new_entry = (entry_price * position + fill * delta) / target_size if position > 0 else fill
        return target_size, new_entry, [{"side": "buy", "price": fill, "amount": delta, "ts": ts}]

    def _open_or_add_short(
        self, position: float, entry_price: float, equity: float, exec_price: float, ts: int, scale: float
    ) -> tuple[float, float, list[dict]]:
        """Open new short or add to existing short position."""
        fill = self._apply_costs(exec_price, "sell")
        target_size = equity * self.position_pct * scale / fill
        abs_pos = abs(position)
        if target_size <= abs_pos + 1e-12:
            return position, entry_price, []
        delta = target_size - abs_pos
        new_entry = (entry_price * abs_pos + fill * delta) / target_size if position < 0 else fill
        return -target_size, new_entry, [{"side": "sell", "price": fill, "amount": delta, "ts": ts}]

    def _execute_pending_action(
        self,
        pending_action: str,
        position: float,
        entry_price: float,
        equity: float,
        exec_price: float,
        ts: int,
        position_scale: float = 1.0,
    ) -> tuple[float, float, float, list[dict]]:
        """Execute pending action and return updated position, entry_price, equity, and new trades.

        Supports: new entry, add to position (加仓), close, and reverse.
        """
        trades: list[dict] = []
        if pending_action == "close" and position != 0:
            position, entry_price, equity, trades = self._close_position(position, entry_price, equity, exec_price, ts)
        elif pending_action == "long":
            if position < 0:  # close short first
                position, entry_price, equity, close_trades = self._close_position(
                    position, entry_price, equity, exec_price, ts
                )
                trades.extend(close_trades)
            position, entry_price, open_trades = self._open_or_add_long(
                position, entry_price, equity, exec_price, ts, position_scale
            )
            trades.extend(open_trades)
        elif pending_action == "short":
            if position > 0:  # close long first
                position, entry_price, equity, close_trades = self._close_position(
                    position, entry_price, equity, exec_price, ts
                )
                trades.extend(close_trades)
            position, entry_price, open_trades = self._open_or_add_short(
                position, entry_price, equity, exec_price, ts, position_scale
            )
            trades.extend(open_trades)
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

        # Fetch extra lookback candles before start so that the first bar in the trading
        # range already has a full history window for SMA / snapshot construction.
        tf_ms = _TF_MS.get(self.interval, 3_600_000)
        lookback_ms = self.lookback * tf_ms
        self._candles = await fetch_historical(self.pair, self.interval, self.start_ms - lookback_ms, self.end_ms)

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

        symbol = self.pair.split("/")[0]
        self._etf_flows = self._load_dict_range("sosovalue_etf", start, end)
        self._oi = self._load_dict_range(f"binance_oi_{symbol}", start, end)
        self._ls_ratio = self._load_dict_range(f"binance_ls_ratio_{symbol}", start, end)

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
        decisions: list[dict] = []
        peak = equity

        from cryptotrader.graph import build_backtest_graph

        graph = build_backtest_graph() if self.use_llm else None

        lookback = self.lookback

        pending_action: str | None = None
        pending_scale: float = 1.0

        max_stop_loss_pct = self._cached_config.risk.max_stop_loss_pct

        for i in range(lookback, len(candles)):
            window = candles[max(0, i - lookback) : i + 1]
            cur = candles[i]
            ts, o, c = cur[0], cur[1], cur[4]

            # Skip candles with invalid data
            if c is None or c <= 0:
                continue

            # Per-bar decision record
            bar_decision: dict = {
                "ts": ts,
                "price": c,
                "open": o,
                "position_before": position,
                "entry_price": entry_price,
                "equity": equity,
            }

            stop_loss_triggered = False
            # Stop-loss check — mirror live graph's check_stop_loss node
            # Skip if AI already requested close (avoid redundant override + confusing logs)
            if position != 0 and entry_price > 0 and pending_action != "close":
                pnl_pct = (c - entry_price) / entry_price if position > 0 else (entry_price - c) / entry_price
                if pnl_pct < -max_stop_loss_pct:
                    pending_action = "close"
                    pending_scale = 1.0
                    stop_loss_triggered = True
                    logger.info(
                        "Backtest stop-loss: %.2f%% loss (threshold: %.2f%%)",
                        pnl_pct * 100,
                        -max_stop_loss_pct * 100,
                    )
            bar_decision["stop_loss_triggered"] = stop_loss_triggered

            # Execute pending action from PREVIOUS bar's signal at current bar's open
            # This eliminates look-ahead bias: signal on bar[i-1], fill on bar[i] open
            executed_action = None
            if pending_action is not None:
                executed_action = pending_action
                exec_price = o if (o is not None and o > 0) else c
                position, entry_price, equity, new_trades = self._execute_pending_action(
                    pending_action, position, entry_price, equity, exec_price, ts, pending_scale
                )
                trades.extend(new_trades)
                pending_action = None
                pending_scale = 1.0
            bar_decision["executed_action"] = executed_action

            # Generate signal on current bar (will be executed on NEXT bar)
            analyses = {}
            verdict = {}
            risk_gate = {}
            debate_skipped = False
            original_action = "hold"
            node_trace: list[dict] = []
            if graph and self.use_llm:
                snapshot = self._build_snapshot(window, ts, i)
                result = await self._run_graph(graph, snapshot, position, entry_price, equity, peak)
                node_trace = result.pop("_node_trace", [])
                data = result.get("data", {})
                verdict = data.get("verdict", {})
                analyses = data.get("analyses", {})
                risk_gate = data.get("risk_gate", {})
                debate_skipped = data.get("debate_skipped", False)
                original_action = verdict.get("action", "hold")
                action = original_action
                # Check risk gate — reject trade if risk gate failed
                if not risk_gate.get("passed", True) and action != "hold":
                    logger.info(
                        "Backtest risk gate rejected: %s — %s",
                        risk_gate.get("rejected_by", "unknown"),
                        risk_gate.get("reason", ""),
                    )
                    action = "hold"
            else:
                action = self._simple_signal(window)
                original_action = action

            if action != "hold":
                pending_action = action
                pending_scale = verdict.get("position_scale", 1.0)

            # Mark to market at close
            mtm = self._mark_to_market(position, equity, entry_price, c)
            equity_curve.append(mtm)
            peak = max(peak, mtm)

            # Record decision details
            bar_decision.update(
                {
                    "position_after": position,
                    "equity_after": mtm,
                    "analyses": {
                        k: {
                            "direction": v.get("direction", ""),
                            "confidence": v.get("confidence", 0),
                            "data_sufficiency": v.get("data_sufficiency", ""),
                        }
                        for k, v in analyses.items()
                    },
                    "debate_skipped": debate_skipped,
                    "verdict": {
                        "action": verdict.get("action", "hold"),
                        "confidence": verdict.get("confidence", 0),
                        "position_scale": verdict.get("position_scale", 0),
                        "reasoning": verdict.get("reasoning", ""),
                        "thesis": verdict.get("thesis", ""),
                    },
                    "risk_gate": {
                        "passed": risk_gate.get("passed", True),
                        "rejected_by": risk_gate.get("rejected_by", ""),
                        "reason": risk_gate.get("reason", ""),
                    },
                    "final_action": action
                    if action != "hold"
                    else ("hold" if original_action == "hold" else "rejected"),
                    "pending_action": pending_action,
                    "node_trace": [
                        {"node": t["node"], "summary": t["summary"], "duration_ms": t["duration_ms"]}
                        for t in node_trace
                    ],
                }
            )
            decisions.append(bar_decision)

        # Close any open position at end
        equity, final_trades = self._close_final_position(position, entry_price, equity, candles[-1][4], candles[-1][0])
        trades.extend(final_trades)

        return self._compute_result(equity, equity_curve, trades, decisions)

    def _simple_signal(self, window: list[list]) -> str:
        closes = [c[4] for c in window]
        sma_fast = self._cached_config.backtest.sma_fast
        sma_slow = self._cached_config.backtest.sma_slow
        if len(closes) < sma_fast:
            return "hold"
        fast_avg = sum(closes[-sma_fast:]) / sma_fast
        slow_avg = sum(closes[-sma_slow:]) / sma_slow if len(closes) >= sma_slow else fast_avg
        if closes[-1] > fast_avg > slow_avg:
            return "long"
        if closes[-1] < fast_avg < slow_avg:
            return "short"
        return "hold"

    def _build_snapshot(self, window: list[list], ts: int, candle_idx: int) -> DataSnapshot:
        from cryptotrader.backtest.historical_data import derive_news_events

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

        # Key events derived from price action (sentiment analysis delegated to LLM)
        events = derive_news_events(self._candles, candle_idx)

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

    async def _run_graph(
        self,
        graph,
        snapshot: DataSnapshot,
        position: float = 0.0,
        entry_price: float = 0.0,
        equity: float = 0.0,
        peak: float = 0.0,
    ) -> dict:
        from cryptotrader.state import build_initial_state

        # Build position context so verdict has position awareness
        # Mirror the format used by _build_position_from_portfolio() in live mode
        current_price = snapshot.market.ticker.get("last", 0)
        if position == 0:
            pos_ctx = {"side": "flat"}
        else:
            pos_ctx = {
                "side": "long" if position > 0 else "short",
                "entry_price": entry_price,
                "current_price": current_price,
                "amount": abs(position),
            }

        # Construct risk constraints from backtest state variables
        # (live mode queries PortfolioManager/Redis, but backtest has its own equity tracking)
        risk_cfg = self._cached_config.risk
        position_value = abs(position * entry_price) if position != 0 else 0.0
        exposure_pct = position_value / equity if equity > 0 else 0.0
        max_exp = risk_cfg.position.max_total_exposure_pct
        drawdown_current = (peak - equity) / peak if peak > 0 else 0.0
        backtest_constraints = {
            "max_position_pct": risk_cfg.position.max_single_pct,
            "max_drawdown_pct": risk_cfg.loss.max_drawdown_pct,
            "remaining_exposure_pct": max(0.0, max_exp - exposure_pct),
            "daily_loss_remaining_pct": risk_cfg.loss.max_daily_loss_pct,
            "drawdown_current": drawdown_current,
        }
        # Add market conditions if available
        summary = snapshot.market
        if hasattr(summary, "funding_rate") and summary.funding_rate is not None:
            backtest_constraints["funding_rate"] = summary.funding_rate
        if hasattr(summary, "volatility") and summary.volatility is not None:
            backtest_constraints["volatility"] = summary.volatility

        # Build portfolio dict for risk gate (mirrors what risk_check() builds in live mode)
        recent_closes = snapshot.market.ohlcv["close"].dropna().tolist() if snapshot.market.ohlcv is not None else []
        positions = {self.pair: {"amount": position, "avg_price": entry_price}} if position != 0 else {}
        portfolio = {
            "total_value": equity,
            "positions": positions,
            "daily_pnl": 0.0,
            "drawdown": drawdown_current,
            "returns_60d": [],
            "recent_prices": recent_closes[-60:],
            "funding_rate": snapshot.market.funding_rate or 0,
            "api_latency_ms": 100,
            "pair": self.pair,
        }

        initial = build_initial_state(
            self.pair,
            engine="paper",
            snapshot=snapshot,
            config=self._cached_config,
            extra_metadata={"llm_verdict": self.use_llm, "backtest_mode": True},
            extra_data={
                "position_context": pos_ctx,
                "backtest_constraints": backtest_constraints,
                "portfolio": portfolio,
            },
        )
        initial["max_debate_rounds"] = self._cached_config.debate.max_rounds

        from cryptotrader.tracing import add_timing_to_trace, run_graph_traced

        final_state, node_trace = await run_graph_traced(graph, initial)
        add_timing_to_trace(node_trace)
        # Attach trace to result for dashboard display
        final_state["_node_trace"] = node_trace
        return final_state

    def _compute_result(
        self, equity: float, curve: list[float], trades: list[dict], decisions: list[dict] | None = None
    ) -> BacktestResult:
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
            decisions=decisions or [],
            llm_calls=self._llm_calls,
            llm_tokens=self._llm_tokens,
        )
