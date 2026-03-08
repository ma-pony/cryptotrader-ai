"""Backtest with progress output. Uses lite graph (no debate)."""

import argparse
import asyncio
import math
import os
import sys
import time
from datetime import UTC, datetime

import pandas as pd

from cryptotrader.backtest.cache import fetch_historical
from cryptotrader.backtest.historical_data import (
    derive_news_sentiment,
    fetch_btc_dominance,
    fetch_fear_greed,
    fetch_fred_series,
    fetch_funding_rate,
)
from cryptotrader.graph import build_debate_graph, build_lite_graph
from cryptotrader.models import DataSnapshot, MacroData, MarketData, NewsSentiment, OnchainData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_p = argparse.ArgumentParser()
_p.add_argument("--pair", default="BTC/USDT")
_p.add_argument("--model", default="")
_p.add_argument("--start", default="2025-06-01")
_p.add_argument("--end", default="2025-12-31")
_p.add_argument(
    "--stop-loss", type=float, default=0.15, help="Catastrophic stop loss pct (default 15%, safety net only)"
)
_p.add_argument("--trailing-stop", type=float, default=0.0, help="Trailing stop pct (default 0 = disabled)")
_p.add_argument(
    "--reversal-days", type=int, default=3, help="Consecutive opposite-signal days to trigger exit (default 3)"
)
_p.add_argument(
    "--drawdown-pause", type=float, default=0.10, help="Account drawdown pct to pause trading (default 10%)"
)
_p.add_argument("--atr-sizing", action="store_true", help="Use ATR-based position sizing")
_p.add_argument("--version", default="v13", help="Version label for output")
_p.add_argument("--debate", action="store_true", help="Use bull/bear adversarial debate graph")
_p.add_argument("--debate-rounds", type=int, default=2, help="Number of debate rounds (default 2)")
_args = _p.parse_args()

PAIR = _args.pair
START = _args.start
END = _args.end
INTERVAL = "1d"
CAPITAL = 10000
LOOKBACK = 100
MODEL = _args.model
MIN_HOLD_DAYS = 5
ADX_THRESHOLD = 0
STOP_LOSS = _args.stop_loss
TRAILING_STOP = _args.trailing_stop
ATR_SIZING = _args.atr_sizing
REVERSAL_DAYS = _args.reversal_days
DRAWDOWN_PAUSE = _args.drawdown_pause


def calc_adx(candles: list[list], idx: int, period: int = 14) -> float:
    """Calculate ADX from OHLCV candles ending at idx."""
    if idx < period * 2:
        return 0.0
    window = candles[idx - period * 2 : idx + 1]
    plus_dm, minus_dm, tr_list = [], [], []
    for i in range(1, len(window)):
        h, low, pc = window[i][2], window[i][3], window[i - 1][4]
        up = h - window[i - 1][2]
        down = window[i - 1][3] - low
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)
        tr_list.append(max(h - low, abs(h - pc), abs(low - pc)))
    # Smoothed averages (Wilder's method)
    atr = sum(tr_list[:period]) / period
    plus_di_s = sum(plus_dm[:period]) / period
    minus_di_s = sum(minus_dm[:period]) / period
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_di_s = (plus_di_s * (period - 1) + plus_dm[i]) / period
        minus_di_s = (minus_di_s * (period - 1) + minus_dm[i]) / period
    if atr == 0:
        return 0.0
    plus_di = 100 * plus_di_s / atr
    minus_di = 100 * minus_di_s / atr
    return 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0


async def _fetch_all_data(pair: str, start: str, end: str, start_ms: int, end_ms: int, interval: str):
    """Fetch all historical data needed for backtest."""
    print("Fetching candles...", flush=True)
    candles = await fetch_historical(pair, interval, start_ms, end_ms)
    print(f"Got {len(candles)} candles, {len(candles) - LOOKBACK} tradeable steps", flush=True)

    print("Fetching Fear & Greed history...", flush=True)
    fng = await fetch_fear_greed(start, end)
    print(f"Got {len(fng)} days of Fear & Greed data", flush=True)

    print("Fetching funding rate history...", flush=True)
    funding = await fetch_funding_rate(pair.split("/")[0], start, end)
    print(f"Got {len(funding)} days of funding rate data", flush=True)

    print("Fetching BTC dominance history...", flush=True)
    try:
        btc_dom = await fetch_btc_dominance(start, end)
    except Exception as e:
        print(f"  Warning: BTC dominance fetch failed ({e}), using empty", flush=True)
        btc_dom = {}
    print(f"Got {len(btc_dom)} days of BTC dominance data", flush=True)

    print("Fetching Fed Rate history...", flush=True)
    try:
        fed_rate = await fetch_fred_series("DFF", start, end)
    except Exception as e:
        print(f"  Warning: Fed Rate fetch failed ({e}), using empty", flush=True)
        fed_rate = {}
    print(f"Got {len(fed_rate)} days of Fed Rate data", flush=True)

    print("Fetching DXY history...", flush=True)
    try:
        dxy = await fetch_fred_series("DTWEXBGS", start, end)
    except Exception as e:
        print(f"  Warning: DXY fetch failed ({e}), using empty", flush=True)
        dxy = {}
    print(f"Got {len(dxy)} days of DXY data", flush=True)

    print("Fetching futures volume history...", flush=True)
    from cryptotrader.backtest.historical_data import fetch_futures_volume

    fut_vol = await fetch_futures_volume(pair.split("/")[0], start, end)
    print(f"Got {len(fut_vol)} days of futures volume data", flush=True)

    # Load extended on-chain data from unified SQLite store
    from cryptotrader.data.store import get_range

    symbol = pair.split("/")[0]
    store_data = {}
    for source_key in [
        f"binance_oi_{symbol}",
        f"binance_ls_ratio_{symbol}",
        "defillama_tvl",
        "sosovalue_etf",
        "stablecoin_total_supply",
        f"news_headlines_{symbol.lower()}",
    ]:
        store_data[source_key] = {d: v for d, v in get_range(source_key, start, end).items() if isinstance(v, dict)}
    news_key = f"news_headlines_{symbol.lower()}"
    print(
        f"Store data: OI={len(store_data.get(f'binance_oi_{symbol}', {}))}, "
        f"LS={len(store_data.get(f'binance_ls_ratio_{symbol}', {}))}, "
        f"TVL={len(store_data.get('defillama_tvl', {}))}, "
        f"ETF={len(store_data.get('sosovalue_etf', {}))}, "
        f"News={len(store_data.get(news_key, {}))}",
        flush=True,
    )

    return candles, fng, funding, btc_dom, fed_rate, dxy, fut_vol, store_data


def _check_reversal_stop(position: float, reversal_count: int, entry_price: float, c: float, date: str):
    """Check if reversal stop should trigger."""
    if REVERSAL_DAYS > 0 and reversal_count >= REVERSAL_DAYS and position != 0:
        if position > 0:
            pnl = (c - entry_price) * position
            trade = {"side": "reversal_exit_long", "price": c, "pnl": pnl, "date": date}
        else:
            pnl = (entry_price - c) * abs(position)
            trade = {"side": "reversal_exit_short", "price": c, "pnl": pnl, "date": date}
        return True, pnl, trade
    return False, 0.0, None


def _check_catastrophic_stop(position: float, entry_price: float, c: float, date: str):
    """Check if catastrophic stop should trigger."""
    if position != 0:
        loss = (entry_price - c) / entry_price if position > 0 else (c - entry_price) / entry_price
        if loss >= STOP_LOSS:
            pnl = (c - entry_price) * position if position > 0 else (entry_price - c) * abs(position)
            trade = {"side": "catastrophic_stop", "price": c, "pnl": pnl, "date": date}
            return True, pnl, trade
    return False, 0.0, None


def _check_trailing_stop(position: float, entry_price: float, c: float, peak_price: float, date: str):
    """Check if trailing stop should trigger."""
    if TRAILING_STOP > 0 and position != 0:
        if position > 0:
            new_peak = max(peak_price, c)
            if c > entry_price and (new_peak - c) / new_peak >= TRAILING_STOP:
                pnl = (c - entry_price) * position
                trade = {"side": "trail_stop_long", "price": c, "pnl": pnl, "date": date}
                return True, pnl, trade, new_peak
            return False, 0.0, None, new_peak
        new_peak = min(peak_price, c) if peak_price > 0 else c
        if c < entry_price and (c - new_peak) / new_peak >= TRAILING_STOP:
            pnl = (entry_price - c) * abs(position)
            trade = {"side": "trail_stop_short", "price": c, "pnl": pnl, "date": date}
            return True, pnl, trade, new_peak
        return False, 0.0, None, new_peak
    return False, 0.0, None, peak_price


def _execute_trade(
    action: str, position: float, equity: float, c: float, entry_price: float, size_pct: float, date: str
):
    """Execute a trade based on action signal."""
    trades = []

    if action == "long" and position <= 0:
        if position < 0:  # Close short first
            pnl = (entry_price - c) * abs(position)
            equity += pnl
            trades.append({"side": "close_short", "price": c, "pnl": pnl, "date": date})
        # Open long
        size = equity * size_pct / c
        position = size
        entry_price = c
        trades.append({"side": "buy", "price": c, "amount": size, "date": date})
    elif action == "short" and position >= 0:
        if position > 0:  # Close long first
            pnl = (c - entry_price) * position
            equity += pnl
            trades.append({"side": "close_long", "price": c, "pnl": pnl, "date": date})
        # Open short
        size = equity * size_pct / c
        position = -size
        entry_price = c
        trades.append({"side": "sell", "price": c, "amount": size, "date": date})

    return position, entry_price, equity, trades


def _calculate_position_size(confidence: float, equity: float, c: float, window: list, atr_sizing: bool) -> float:
    """Calculate position size based on confidence and optionally ATR."""
    if confidence >= 0.8:
        size_pct = 0.15
    elif confidence >= 0.6:
        size_pct = 0.10
    else:
        size_pct = 0.05

    if atr_sizing and len(window) >= 15:
        atr_sum = sum(
            max(
                window[j][2] - window[j][3],
                abs(window[j][2] - window[j - 1][4]),
                abs(window[j][3] - window[j - 1][4]),
            )
            for j in range(len(window) - 14, len(window))
        )
        atr_val = atr_sum / 14
        if atr_val > 0:
            size_pct = min(0.15, max(0.03, (equity * 0.01 / atr_val) * c / equity))

    return size_pct


def _calculate_mtm(position: float, equity: float, entry_price: float, c: float) -> float:
    """Calculate mark-to-market equity."""
    if position > 0:
        return equity + (c - entry_price) * position
    if position < 0:
        return equity + (entry_price - c) * abs(position)
    return equity


def _build_news(sd, symbol, date_str, c, fng_val, fallback_sentiment, fallback_events):
    """Build NewsSentiment from store data or fallback to price-derived proxy."""
    news_data = sd.get(f"news_headlines_{symbol.lower()}", {}).get(date_str, {})
    if news_data and isinstance(news_data, dict) and news_data.get("headlines"):
        return NewsSentiment(
            headlines=news_data["headlines"][:10],
            sentiment_score=news_data.get("sentiment_score", 0.0),
            key_events=news_data.get("key_events", []),
        )
    return NewsSentiment(
        sentiment_score=fallback_sentiment,
        key_events=fallback_events,
        headlines=[f"BTC at ${c:,.0f}, Fear&Greed={fng_val}"],
    )


def _build_snapshot(candles, i, ts, c, v, window, fng, funding, btc_dom, fed_rate, dxy, fut_vol, store_data=None):
    """Build DataSnapshot for current candle."""
    df = pd.DataFrame(window, columns=["timestamp", "open", "high", "low", "close", "volume"])
    date_str = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d")
    fng_val = fng.get(date_str, 50)
    fr_val = funding.get(date_str, 0.0)
    dom_val = btc_dom.get(date_str, 0.0)
    fed_val = fed_rate.get(date_str, 0.0)
    dxy_val = dxy.get(date_str, 0.0)
    sentiment, events = derive_news_sentiment(candles, i)
    fv = fut_vol.get(date_str, {})
    fut_volume = fv.get("volume", 0.0)
    vol_20d = []
    for j in range(max(0, i - 20), i):
        d_j = datetime.fromtimestamp(candles[j][0] / 1000, UTC).strftime("%Y-%m-%d")
        vj = fut_vol.get(d_j, {}).get("volume", 0)
        if vj > 0:
            vol_20d.append(vj)
    avg_vol = sum(vol_20d) / len(vol_20d) if vol_20d else fut_volume or 1
    vol_ratio = fut_volume / avg_vol if avg_vol > 0 else 1.0

    # Enrich on-chain data from unified store
    symbol = PAIR.split("/")[0]
    sd = store_data or {}
    oi_data = sd.get(f"binance_oi_{symbol}", {}).get(date_str, {})
    ls_data = sd.get(f"binance_ls_ratio_{symbol}", {}).get(date_str, {})
    tvl_data = sd.get("defillama_tvl", {}).get(date_str, {})
    etf_data = sd.get("sosovalue_etf", {}).get(date_str, {})
    stablecoin_data = sd.get("stablecoin_total_supply", {}).get(date_str, {})

    oi_val = oi_data.get("openInterestValue", fut_volume) if oi_data else fut_volume
    defi_tvl = tvl_data.get("tvl", 0.0) if isinstance(tvl_data, dict) else 0.0
    ls_ratio = ls_data.get("longShortRatio", 1.0) if ls_data else 1.0
    etf_inflow = etf_data.get("dailyNetInflow", 0.0) if etf_data else 0.0
    etf_aum = etf_data.get("totalNetAssets", 0.0) if etf_data else 0.0
    stablecoin_supply = stablecoin_data.get("total_supply", 0.0) if isinstance(stablecoin_data, dict) else 0.0

    return DataSnapshot(
        timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC),
        pair=PAIR,
        market=MarketData(
            pair=PAIR,
            ohlcv=df,
            ticker={"last": c, "baseVolume": v},
            funding_rate=fr_val,
            orderbook_imbalance=0.0,
            volatility=df["close"].pct_change().std() or 0.0,
        ),
        onchain=OnchainData(
            open_interest=oi_val,
            liquidations_24h={
                "volume_ratio": vol_ratio,
                "futures_volume": fut_volume,
                "long_short_ratio": ls_ratio,
            },
            defi_tvl=defi_tvl,
            data_quality={
                "has_oi": bool(oi_data),
                "has_ls_ratio": bool(ls_data),
                "has_etf": bool(etf_data),
                "has_defi_tvl": defi_tvl > 0,
            },
        ),
        news=_build_news(sd, symbol, date_str, c, fng_val, sentiment, events),
        macro=MacroData(
            fear_greed_index=fng_val,
            btc_dominance=dom_val,
            fed_rate=fed_val,
            dxy=dxy_val,
            etf_daily_net_inflow=etf_inflow,
            etf_total_net_assets=etf_aum,
            stablecoin_total_supply=stablecoin_supply,
        ),
    ), fng_val


def _build_trend_context(candles, i, c):
    """Build trend summary from price history for verdict AI."""
    ctx = {"current_price": c}
    for days, label in [(7, "7d"), (14, "14d"), (30, "30d")]:
        if i >= days:
            past_close = candles[i - days][4]
            if past_close > 0:
                ctx[f"change_{label}"] = (c - past_close) / past_close
    # 30d high/low
    lookback_30 = candles[max(0, i - 30) : i + 1]
    if lookback_30:
        ctx["high_30d"] = max(bar[2] for bar in lookback_30)
        ctx["low_30d"] = min(bar[3] for bar in lookback_30)
    return ctx


def _build_position_context(position, entry_price, c, entry_step, current_step):
    """Build position state dict for verdict AI."""
    if position == 0:
        return {"side": "flat"}
    return {
        "side": "long" if position > 0 else "short",
        "entry_price": entry_price,
        "current_price": c,
        "days_held": current_step - entry_step,
    }


async def _get_action(
    graph, snapshot, candles, i, window, equity, c, t0, position=0.0, entry_price=0.0, entry_step=0, current_step=0
):
    """Get trading action from ADX filter or graph."""
    adx = calc_adx(candles, i)
    if adx < ADX_THRESHOLD:
        return "hold", 0.0, 0.05, time.time() - t0, adx

    position_context = _build_position_context(position, entry_price, c, entry_step, current_step)
    trend_context = _build_trend_context(candles, i, c)

    result = await graph.ainvoke(
        {
            "messages": [],
            "data": {
                "snapshot": snapshot,
                "position_context": position_context,
                "trend_context": trend_context,
            },
            "metadata": {
                "pair": PAIR,
                "engine": "paper",
                "models": dict.fromkeys(["tech_agent", "chain_agent", "news_agent", "macro_agent"], MODEL),
                "debate_model": MODEL,
                "verdict_model": MODEL,
                "llm_verdict": True,
                "debate_rounds": _args.debate_rounds,
            },
            "debate_round": 0,
            "max_debate_rounds": 0,
            "divergence_scores": [],
        }
    )
    verdict = result.get("data", {}).get("verdict", {})
    action = verdict.get("action", "hold")
    confidence = verdict.get("confidence", 0.5)
    size_pct = _calculate_position_size(confidence, equity, c, window, False)
    return action, confidence, size_pct, time.time() - t0, adx


def _process_stops(position, reversal_count, entry_price, c, date, peak_price, action):
    """Process all stop loss checks and return updated state."""
    new_reversal_count = reversal_count

    # Track signal reversal count
    if (position > 0 and action == "short") or (position < 0 and action == "long"):
        new_reversal_count += 1
    else:
        new_reversal_count = 0

    # Check reversal stop
    rev_stopped, rev_pnl, rev_trade = _check_reversal_stop(position, new_reversal_count, entry_price, c, date)
    if rev_stopped:
        return True, rev_pnl, rev_trade, 0, peak_price

    # Check catastrophic stop
    cat_stopped, cat_pnl, cat_trade = _check_catastrophic_stop(position, entry_price, c, date)
    if cat_stopped:
        return True, cat_pnl, cat_trade, 0, peak_price

    # Check trailing stop
    trail_stopped, trail_pnl, trail_trade, new_peak = _check_trailing_stop(position, entry_price, c, peak_price, date)
    if trail_stopped:
        return True, trail_pnl, trail_trade, 0, new_peak

    return False, 0.0, None, new_reversal_count, peak_price


def _should_execute_trade(action, position, stopped, step, paused_until, days_since_flip):
    """Check if trade should be executed based on all conditions."""
    if stopped or step <= paused_until:
        return False
    can_flip = days_since_flip >= MIN_HOLD_DAYS
    return (action == "long" and position <= 0 and (position == 0 or can_flip)) or (
        action == "short" and position >= 0 and (position == 0 or can_flip)
    )


def _check_drawdown_pause(equity, position, entry_price, c, peak_equity, step):
    """Check if trading should be paused due to drawdown."""
    mtm_now = equity + (
        (c - entry_price) * position if position > 0 else (entry_price - c) * abs(position) if position < 0 else 0
    )
    new_peak = max(peak_equity, mtm_now)
    if DRAWDOWN_PAUSE > 0 and (new_peak - mtm_now) / new_peak >= DRAWDOWN_PAUSE:
        return new_peak, step + 1
    return new_peak, -1


def _get_position_str(position: float) -> str:
    """Get position state string for display."""
    if position > 0:
        return "LONG"
    if position < 0:
        return "SHORT"
    return "FLAT"


def _print_summary(equity, equity_curve, trades, total_time, total_steps):
    """Print backtest performance summary."""
    total_return = (equity - CAPITAL) / CAPITAL
    # Sharpe
    rets = [
        (equity_curve[j] - equity_curve[j - 1]) / equity_curve[j - 1]
        for j in range(1, len(equity_curve))
        if equity_curve[j - 1] > 0
    ]
    avg = sum(rets) / len(rets) if rets else 0
    std = (sum((r - avg) ** 2 for r in rets) / len(rets)) ** 0.5 if rets else 1
    sharpe = avg / std * math.sqrt(252) if std > 0 else 0
    # Max DD
    peak = equity_curve[0]
    max_dd = 0
    for ev in equity_curve:
        peak = max(peak, ev)
        max_dd = min(max_dd, (ev - peak) / peak if peak > 0 else 0)
    # Win rate
    pnl_trades = [t for t in trades if "pnl" in t]
    wins = sum(1 for t in pnl_trades if t["pnl"] > 0)
    wr = wins / len(pnl_trades) if pnl_trades else 0

    print(f"\n{'=' * 50}", flush=True)
    graph_label = "debate" if _args.debate else "lite"
    model_label = MODEL or "config-default"
    print(f"LLM Strategy ({model_label}, {graph_label} graph)", flush=True)
    print(f"Return:  {total_return:+.2%}", flush=True)
    print(f"Sharpe:  {sharpe:.2f}", flush=True)
    print(f"MaxDD:   {max_dd:.2%}", flush=True)
    print(f"WinRate: {wr:.2%}", flush=True)
    print(f"Trades:  {len(trades)}", flush=True)
    print(f"Final:   ${equity:,.2f}", flush=True)
    print(f"Time:    {total_time:.0f}s ({total_time / total_steps:.1f}s/step)", flush=True)
    print("\nTrade log:", flush=True)
    for t in trades:
        print(f"  {t}", flush=True)


async def main():
    start_ms = int(datetime.fromisoformat(START).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(END).replace(tzinfo=UTC).timestamp() * 1000)

    candles, fng, funding, btc_dom, fed_rate, dxy, fut_vol, store_data = await _fetch_all_data(
        PAIR, START, END, start_ms, end_ms, INTERVAL
    )
    total_steps = len(candles) - LOOKBACK

    graph = build_debate_graph() if _args.debate else build_lite_graph()
    equity = CAPITAL
    position = 0.0
    entry_price = 0.0
    entry_step = 0  # Track when position was opened
    peak_price = 0.0
    peak_equity = CAPITAL  # For drawdown pause
    trades = []
    equity_curve = [equity]
    t_total = time.time()
    last_direction_change = -MIN_HOLD_DAYS
    reversal_count = 0  # Consecutive days signal opposes position
    paused_until = -1  # Step until which trading is paused
    pending_action = None  # For look-ahead bias fix: signal on bar[i], execute on bar[i+1]

    for i in range(LOOKBACK, len(candles)):
        step = i - LOOKBACK + 1
        window = candles[max(0, i - LOOKBACK) : i + 1]
        cur = candles[i]
        ts, _o, _h, _l, c, v = cur[0], cur[1], cur[2], cur[3], cur[4], cur[5]
        date = datetime.fromtimestamp(ts / 1000, UTC).strftime("%m-%d")

        # Execute pending action from previous bar at current bar's open price
        open_price = cur[1]  # Use open price for execution (eliminates look-ahead bias)
        if pending_action is not None:
            pa_action, pa_size_pct = pending_action
            pending_action = None
            days_since_flip = step - last_direction_change
            if _should_execute_trade(pa_action, position, False, step, paused_until, days_since_flip):
                position, entry_price, equity, new_trades = _execute_trade(
                    pa_action, position, equity, open_price, entry_price, pa_size_pct, date
                )
                trades.extend(new_trades)
                peak_price = open_price
                last_direction_change = step
                entry_step = step

        t0 = time.time()
        snapshot, fng_val = _build_snapshot(
            candles,
            i,
            ts,
            c,
            v,
            window,
            fng,
            funding,
            btc_dom,
            fed_rate,
            dxy,
            fut_vol,
            store_data,
        )

        action, confidence, size_pct, elapsed, adx = await _get_action(
            graph,
            snapshot,
            candles,
            i,
            window,
            equity,
            c,
            t0,
            position=position,
            entry_price=entry_price,
            entry_step=entry_step,
            current_step=step,
        )

        # Process all stop checks
        stopped, pnl, trade, reversal_count, peak_price = _process_stops(
            position, reversal_count, entry_price, c, date, peak_price, action
        )
        if stopped:
            equity += pnl
            trades.append(trade)
            position = 0.0
            last_direction_change = step

        # Drawdown pause check
        peak_equity, new_paused = _check_drawdown_pause(equity, position, entry_price, c, peak_equity, step)
        if new_paused > 0:
            paused_until = new_paused

        # ATR-based position sizing override
        if ATR_SIZING:
            size_pct = _calculate_position_size(confidence, equity, c, window, True)

        # Defer execution to next bar's open (eliminates look-ahead bias)
        if action in ("long", "short") and not stopped:
            pending_action = (action, size_pct)

        # Track actual position state for display
        pos_str = _get_position_str(position)

        # MTM
        mtm = _calculate_mtm(position, equity, entry_price, c)
        equity_curve.append(mtm)

        print(
            f"[{step:3d}/{total_steps}] {date} ${c:,.0f} -> {action:5s} conf={confidence:.0%} sz={size_pct:.0%} "
            f"eq=${mtm:,.2f} pos={pos_str} fng={fng_val} adx={adx:.0f} ({elapsed:.0f}s)",
            flush=True,
        )

    # Close open position
    if position != 0:
        fp = candles[-1][4]
        pnl = (fp - entry_price) * position if position > 0 else (entry_price - fp) * abs(position)
        equity += pnl
        trades.append({"side": "close", "price": fp, "pnl": pnl})

    _print_summary(equity, equity_curve, trades, time.time() - t_total, total_steps)


asyncio.run(main())
