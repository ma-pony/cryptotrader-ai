"""Backtest with progress output. Uses lite graph (no debate)."""
import asyncio, time, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, UTC
import pandas as pd
from cryptotrader.backtest.cache import fetch_historical
from cryptotrader.backtest.historical_data import fetch_fear_greed, fetch_funding_rate, fetch_btc_dominance, fetch_fred_series, derive_news_sentiment
from cryptotrader.graph import build_lite_graph
from cryptotrader.models import DataSnapshot, MarketData, OnchainData, NewsSentiment, MacroData

PAIR = "BTC/USDT"
START = "2025-06-01"
END = "2025-12-31"
INTERVAL = "1d"
CAPITAL = 10000
LOOKBACK = 100
MODEL = "openai/deepseek-chat"
MIN_HOLD_DAYS = 3  # Prevent frequent flips during trend reversals
ADX_THRESHOLD = 0  # Disabled — Tech Agent handles regime detection


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
    funding = await fetch_funding_rate("BTC", START, END)
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

    graph = build_lite_graph()
    equity = CAPITAL
    position = 0.0
    entry_price = 0.0
    trades = []
    equity_curve = [equity]
    t_total = time.time()
    last_direction_change = -MIN_HOLD_DAYS  # Allow first trade immediately

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

        snapshot = DataSnapshot(
            timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC), pair=PAIR,
            market=MarketData(pair=PAIR, ohlcv=df, ticker={"last": c, "baseVolume": v},
                funding_rate=fr_val, orderbook_imbalance=0.0,
                volatility=df["close"].pct_change().std() or 0.0),
            onchain=OnchainData(),
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

            # Fixed 10% position size (v6: test data quality, not sizing)
            size_pct = 0.10

        # Execute with cooldown: don't flip direction within MIN_HOLD_DAYS
        days_since_flip = step - last_direction_change
        can_flip = days_since_flip >= MIN_HOLD_DAYS

        if action == "long" and position <= 0 and (position == 0 or can_flip):
            if position < 0:
                pnl = (entry_price - c) * abs(position)
                equity += pnl
                trades.append({"side": "close_short", "price": c, "pnl": pnl, "date": date})
            size = equity * size_pct / c
            position = size
            entry_price = c
            last_direction_change = step
            trades.append({"side": "buy", "price": c, "amount": size, "size_pct": size_pct, "date": date})
        elif action == "short" and position >= 0 and (position == 0 or can_flip):
            if position > 0:
                pnl = (c - entry_price) * position
                equity += pnl
                trades.append({"side": "close_long", "price": c, "pnl": pnl, "date": date})
            size = equity * size_pct / c
            position = -size
            entry_price = c
            last_direction_change = step
            trades.append({"side": "sell", "price": c, "amount": size, "size_pct": size_pct, "date": date})

        # MTM
        if position > 0:
            mtm = equity + (c - entry_price) * position
        elif position < 0:
            mtm = equity + (entry_price - c) * abs(position)
        else:
            mtm = equity
        equity_curve.append(mtm)

        print(f"[{step:3d}/{total_steps}] {date} ${c:,.0f} -> {action:5s} conf={confidence:.0%} sz={size_pct:.0%} eq=${mtm:,.2f} fng={fng_val} adx={adx:.0f} ({elapsed:.0f}s)", flush=True)

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
