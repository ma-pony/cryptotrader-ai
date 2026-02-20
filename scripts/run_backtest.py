"""Backtest with progress output. Uses lite graph (no debate)."""
import asyncio, time, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, UTC
import pandas as pd
from cryptotrader.backtest.cache import fetch_historical
from cryptotrader.graph import build_lite_graph
from cryptotrader.models import DataSnapshot, MarketData, OnchainData, NewsSentiment, MacroData

PAIR = "BTC/USDT"
START = "2025-06-01"
END = "2025-12-31"
INTERVAL = "1d"
CAPITAL = 10000
LOOKBACK = 100
MODEL = "openai/deepseek-chat"

async def main():
    start_ms = int(datetime.fromisoformat(START).replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(END).replace(tzinfo=UTC).timestamp() * 1000)

    print("Fetching candles...", flush=True)
    candles = await fetch_historical(PAIR, INTERVAL, start_ms, end_ms)
    total_steps = len(candles) - LOOKBACK
    print(f"Got {len(candles)} candles, {total_steps} tradeable steps", flush=True)

    graph = build_lite_graph()
    equity = CAPITAL
    position = 0.0
    entry_price = 0.0
    trades = []
    equity_curve = [equity]
    t_total = time.time()

    for i in range(LOOKBACK, len(candles)):
        step = i - LOOKBACK + 1
        window = candles[max(0, i - LOOKBACK):i + 1]
        cur = candles[i]
        ts, o, h, l, c, v = cur[0], cur[1], cur[2], cur[3], cur[4], cur[5]
        date = datetime.fromtimestamp(ts / 1000, UTC).strftime("%m-%d")

        t0 = time.time()
        df = pd.DataFrame(window, columns=["timestamp", "open", "high", "low", "close", "volume"])
        snapshot = DataSnapshot(
            timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC), pair=PAIR,
            market=MarketData(pair=PAIR, ohlcv=df, ticker={"last": c, "baseVolume": v},
                funding_rate=0.0, orderbook_imbalance=0.0,
                volatility=df["close"].pct_change().std() or 0.0),
            onchain=OnchainData(), news=NewsSentiment(), macro=MacroData())

        state = {
            "messages": [], "data": {"snapshot": snapshot},
            "metadata": {"pair": PAIR, "engine": "paper",
                "models": {k: MODEL for k in ["tech_agent", "chain_agent", "news_agent", "macro_agent"]},
                "debate_model": MODEL},
            "debate_round": 0, "max_debate_rounds": 0, "divergence_scores": [],
        }
        result = await graph.ainvoke(state)
        verdict = result.get("data", {}).get("verdict", {})
        action = verdict.get("action", "hold")
        elapsed = time.time() - t0

        # Execute
        if action == "long" and position <= 0:
            if position < 0:
                pnl = (entry_price - c) * abs(position)
                equity += pnl
                trades.append({"side": "close_short", "price": c, "pnl": pnl, "date": date})
            size = equity * 0.1 / c
            position = size
            entry_price = c
            trades.append({"side": "buy", "price": c, "date": date})
        elif action == "short" and position >= 0:
            if position > 0:
                pnl = (c - entry_price) * position
                equity += pnl
                trades.append({"side": "close_long", "price": c, "pnl": pnl, "date": date})
            size = equity * 0.1 / c
            position = -size
            entry_price = c
            trades.append({"side": "sell", "price": c, "date": date})

        # MTM
        if position > 0:
            mtm = equity + (c - entry_price) * position
        elif position < 0:
            mtm = equity + (entry_price - c) * abs(position)
        else:
            mtm = equity
        equity_curve.append(mtm)

        print(f"[{step:3d}/{total_steps}] {date} ${c:,.0f} -> {action:5s} eq=${mtm:,.2f} ({elapsed:.0f}s)", flush=True)

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
