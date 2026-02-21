"""Backtest with progress output. Uses lite graph (no debate)."""
import asyncio, time, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, UTC
import pandas as pd
from cryptotrader.backtest.cache import fetch_historical
from cryptotrader.backtest.historical_data import fetch_fear_greed, fetch_funding_rate, fetch_btc_dominance, fetch_fred_series, derive_news_sentiment
from cryptotrader.graph import build_lite_graph
from cryptotrader.models import DataSnapshot, MarketData, OnchainData, NewsSentiment, MacroData

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--pair", default="BTC/USDT")
_p.add_argument("--model", default="openai/glm-5")
_p.add_argument("--start", default="2025-06-01")
_p.add_argument("--end", default="2025-12-31")
_args = _p.parse_args()

_p.add_argument("--stop-loss", type=float, default=0.05, help="Fixed stop loss pct (default 5%)")
_p.add_argument("--trailing-stop", type=float, default=0.03, help="Trailing stop pct from peak (default 3%)")
_p.add_argument("--atr-sizing", action="store_true", help="Use ATR-based position sizing")
_p.add_argument("--version", default="v13", help="Version label for output")
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


def calc_adx(candles: list[list], idx: int, period: int = 14) -> float:
    """Calculate ADX from OHLCV candles ending at idx."""
    if idx < period * 2:
        return 0.0
    window = candles[idx - period * 2:idx + 1]
    plus_dm, minus_dm, tr_list = [], [], []
    for i in range(1, len(window)):
        h, l, pc = window[i][2], window[i][3], window[i - 1][4]
        up = h - window[i - 1][2]
        down = window[i - 1][3] - l
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
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
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    return dx  # Simplified: single DX value (approximates ADX for our purpose)

async def main():
    start_ms = int(datetime.fromisoformat(START).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(END).replace(tzinfo=UTC).timestamp() * 1000)

    print("Fetching candles...", flush=True)
    candles = await fetch_historical(PAIR, INTERVAL, start_ms, end_ms)
    total_steps = len(candles) - LOOKBACK
    print(f"Got {len(candles)} candles, {total_steps} tradeable steps", flush=True)

    print("Fetching Fear & Greed history...", flush=True)
    fng = await fetch_fear_greed(START, END)
    print(f"Got {len(fng)} days of Fear & Greed data", flush=True)

    print("Fetching funding rate history...", flush=True)
    funding = await fetch_funding_rate(PAIR.split("/")[0], START, END)
    print(f"Got {len(funding)} days of funding rate data", flush=True)

    print("Fetching BTC dominance history...", flush=True)
    btc_dom = await fetch_btc_dominance(START, END)
    print(f"Got {len(btc_dom)} days of BTC dominance data", flush=True)

    print("Fetching Fed Rate history...", flush=True)
    fed_rate = await fetch_fred_series("DFF", START, END)
    print(f"Got {len(fed_rate)} days of Fed Rate data", flush=True)

    print("Fetching DXY history...", flush=True)
    dxy = await fetch_fred_series("DTWEXBGS", START, END)
    print(f"Got {len(dxy)} days of DXY data", flush=True)

    print("Fetching futures volume history...", flush=True)
    from cryptotrader.backtest.historical_data import fetch_futures_volume
    fut_vol = await fetch_futures_volume(PAIR.split("/")[0], START, END)
    print(f"Got {len(fut_vol)} days of futures volume data", flush=True)

    graph = build_lite_graph()
    equity = CAPITAL
    position = 0.0
    entry_price = 0.0
    peak_price = 0.0  # For trailing stop
    trades = []
    equity_curve = [equity]
    t_total = time.time()
    last_direction_change = -MIN_HOLD_DAYS

    for i in range(LOOKBACK, len(candles)):
        step = i - LOOKBACK + 1
        window = candles[max(0, i - LOOKBACK):i + 1]
        cur = candles[i]
        ts, o, h, l, c, v = cur[0], cur[1], cur[2], cur[3], cur[4], cur[5]
        date = datetime.fromtimestamp(ts / 1000, UTC).strftime("%m-%d")

        t0 = time.time()
        df = pd.DataFrame(window, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Inject historical data
        date_str = datetime.fromtimestamp(ts / 1000, UTC).strftime("%Y-%m-%d")
        fng_val = fng.get(date_str, 50)
        fr_val = funding.get(date_str, 0.0)
        dom_val = btc_dom.get(date_str, 0.0)
        fed_val = fed_rate.get(date_str, 0.0)
        dxy_val = dxy.get(date_str, 0.0)

        # Derive proxy news sentiment from price action
        sentiment, events = derive_news_sentiment(candles, i)

        # Compute futures volume anomaly (vs 20-day avg)
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

        snapshot = DataSnapshot(
            timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC), pair=PAIR,
            market=MarketData(pair=PAIR, ohlcv=df, ticker={"last": c, "baseVolume": v},
                funding_rate=fr_val, orderbook_imbalance=0.0,
                volatility=df["close"].pct_change().std() or 0.0),
            onchain=OnchainData(
                open_interest=fut_volume,  # Use futures volume as OI proxy
                liquidations_24h={"volume_ratio": vol_ratio, "futures_volume": fut_volume},
            ),
            news=NewsSentiment(sentiment_score=sentiment, key_events=events,
                headlines=[f"BTC at ${c:,.0f}, Fear&Greed={fng_val}"]),
            macro=MacroData(fear_greed_index=fng_val, btc_dominance=dom_val,
                fed_rate=fed_val, dxy=dxy_val))

        state = {
            "messages": [], "data": {"snapshot": snapshot},
            "metadata": {"pair": PAIR, "engine": "paper",
                "models": {k: MODEL for k in ["tech_agent", "chain_agent", "news_agent", "macro_agent"]},
                "debate_model": MODEL, "verdict_model": MODEL, "llm_verdict": True},
            "debate_round": 0, "max_debate_rounds": 0, "divergence_scores": [],
        }
        # ADX trend filter: skip LLM in ranging markets
        adx = calc_adx(candles, i)
        if adx < ADX_THRESHOLD:
            action = "hold"  # Override: no trading in ranging market
            elapsed = time.time() - t0
            # Still need confidence/size_pct for display
            confidence = 0.0
            size_pct = 0.05
        else:
            result = await graph.ainvoke(state)
            verdict = result.get("data", {}).get("verdict", {})
            action = verdict.get("action", "hold")
            confidence = verdict.get("confidence", 0.5)
            elapsed = time.time() - t0

            # Dynamic position sizing based on confidence
            if confidence >= 0.8:
                size_pct = 0.15
            elif confidence >= 0.6:
                size_pct = 0.10
            else:
                size_pct = 0.05

        # --- Stop loss / trailing stop check (before LLM action) ---
        stopped = False
        if position > 0:
            peak_price = max(peak_price, c)
            drawdown = (peak_price - c) / peak_price
            loss = (entry_price - c) / entry_price
            if loss >= STOP_LOSS:
                pnl = (c - entry_price) * position
                equity += pnl
                trades.append({"side": "stop_loss_long", "price": c, "pnl": pnl, "date": date})
                position = 0.0; stopped = True
            elif TRAILING_STOP > 0 and drawdown >= TRAILING_STOP and c > entry_price:
                pnl = (c - entry_price) * position
                equity += pnl
                trades.append({"side": "trail_stop_long", "price": c, "pnl": pnl, "date": date})
                position = 0.0; stopped = True
        elif position < 0:
            peak_price = min(peak_price, c) if peak_price > 0 else c
            drawdown = (c - peak_price) / peak_price if peak_price > 0 else 0
            loss = (c - entry_price) / entry_price
            if loss >= STOP_LOSS:
                pnl = (entry_price - c) * abs(position)
                equity += pnl
                trades.append({"side": "stop_loss_short", "price": c, "pnl": pnl, "date": date})
                position = 0.0; stopped = True
            elif TRAILING_STOP > 0 and drawdown >= TRAILING_STOP and c < entry_price:
                pnl = (entry_price - c) * abs(position)
                equity += pnl
                trades.append({"side": "trail_stop_short", "price": c, "pnl": pnl, "date": date})
                position = 0.0; stopped = True
        if stopped:
            last_direction_change = step  # Reset cooldown after stop

        # --- ATR-based position sizing ---
        if ATR_SIZING:
            atr_val = 0.0
            if len(window) >= 15:
                atr_sum = sum(max(window[j][2]-window[j][3], abs(window[j][2]-window[j-1][4]), abs(window[j][3]-window[j-1][4])) for j in range(len(window)-14, len(window)))
                atr_val = atr_sum / 14
            if atr_val > 0:
                # Risk 1% of equity per ATR unit
                size_pct = min(0.20, max(0.03, (equity * 0.01 / atr_val) * c / equity))

        # Execute with cooldown: don't flip direction within MIN_HOLD_DAYS
        days_since_flip = step - last_direction_change
        can_flip = days_since_flip >= MIN_HOLD_DAYS

        if action == "long" and position <= 0 and (position == 0 or can_flip) and not stopped:
            if position < 0:
                pnl = (entry_price - c) * abs(position)
                equity += pnl
                trades.append({"side": "close_short", "price": c, "pnl": pnl, "date": date})
            size = equity * size_pct / c
            position = size
            entry_price = c
            peak_price = c
            last_direction_change = step
            trades.append({"side": "buy", "price": c, "amount": size, "size_pct": size_pct, "date": date})
        elif action == "short" and position >= 0 and (position == 0 or can_flip) and not stopped:
            if position > 0:
                pnl = (c - entry_price) * position
                equity += pnl
                trades.append({"side": "close_long", "price": c, "pnl": pnl, "date": date})
            size = equity * size_pct / c
            position = -size
            entry_price = c
            peak_price = c
            last_direction_change = step
            trades.append({"side": "sell", "price": c, "amount": size, "size_pct": size_pct, "date": date})

        # Track actual position state for display
        if position > 0:
            pos_str = "LONG"
        elif position < 0:
            pos_str = "SHORT"
        else:
            pos_str = "FLAT"

        # MTM
        if position > 0:
            mtm = equity + (c - entry_price) * position
        elif position < 0:
            mtm = equity + (entry_price - c) * abs(position)
        else:
            mtm = equity
        equity_curve.append(mtm)

        print(f"[{step:3d}/{total_steps}] {date} ${c:,.0f} -> {action:5s} conf={confidence:.0%} sz={size_pct:.0%} eq=${mtm:,.2f} pos={pos_str} fng={fng_val} adx={adx:.0f} ({elapsed:.0f}s)", flush=True)

    # Close open position
    if position != 0:
        fp = candles[-1][4]
        pnl = (fp - entry_price) * position if position > 0 else (entry_price - fp) * abs(position)
        equity += pnl
        trades.append({"side": "close", "price": fp, "pnl": pnl})

    total_time = time.time() - t_total
    total_return = (equity - CAPITAL) / CAPITAL
    # Sharpe
    rets = [(equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            for i in range(1, len(equity_curve)) if equity_curve[i-1] > 0]
    avg = sum(rets) / len(rets) if rets else 0
    std = (sum((r - avg)**2 for r in rets) / len(rets))**0.5 if rets else 1
    sharpe = avg / std * math.sqrt(252) if std > 0 else 0
    # Max DD
    peak = equity_curve[0]
    max_dd = 0
    for v in equity_curve:
        peak = max(peak, v)
        max_dd = min(max_dd, (v - peak) / peak if peak > 0 else 0)
    # Win rate
    pnl_trades = [t for t in trades if "pnl" in t]
    wins = sum(1 for t in pnl_trades if t["pnl"] > 0)
    wr = wins / len(pnl_trades) if pnl_trades else 0

    print(f"\n{'='*50}", flush=True)
    print(f"LLM Strategy (deepseek-chat, lite graph)", flush=True)
    print(f"Return:  {total_return:+.2%}", flush=True)
    print(f"Sharpe:  {sharpe:.2f}", flush=True)
    print(f"MaxDD:   {max_dd:.2%}", flush=True)
    print(f"WinRate: {wr:.2%}", flush=True)
    print(f"Trades:  {len(trades)}", flush=True)
    print(f"Final:   ${equity:,.2f}", flush=True)
    print(f"Time:    {total_time:.0f}s ({total_time/total_steps:.1f}s/step)", flush=True)
    print(f"\nBaseline: SMA +1.51% | BTC B&H ~-25%", flush=True)
    print(f"\nTrade log:", flush=True)
    for t in trades:
        print(f"  {t}", flush=True)

asyncio.run(main())
