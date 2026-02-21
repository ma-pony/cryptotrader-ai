"""Compare lite graph (no debate) vs full graph (with debate) on key dates."""
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, UTC
import pandas as pd
from cryptotrader.backtest.cache import fetch_historical
from cryptotrader.backtest.historical_data import *
from cryptotrader.models import *
from cryptotrader.graph import build_lite_graph, build_trading_graph

MODEL = "openai/deepseek-chat"
START, END = "2025-06-01", "2025-12-31"
LOOKBACK = 100
# Key dates: trend start, trend reversal, deep bear
TEST_DATES = ["2025-10-05", "2025-10-10", "2025-11-14", "2025-09-25"]


async def build_snapshot(candles, fng, funding, btc_dom, fed, dxy, target_date):
    target_ts = int(datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
    idx = next((i for i, c in enumerate(candles) if c[0] >= target_ts), None)
    if idx is None:
        return None, None
    window = candles[max(0, idx - LOOKBACK):idx + 1]
    ts, o, h, l, c, v = candles[idx]
    df = pd.DataFrame(window, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    sentiment, events = derive_news_sentiment(candles, idx)
    date_str = datetime.fromtimestamp(ts / 1000, UTC).strftime('%Y-%m-%d')
    return DataSnapshot(
        timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC), pair='BTC/USDT',
        market=MarketData(pair='BTC/USDT', ohlcv=df, ticker={'last': c, 'baseVolume': v},
            funding_rate=funding.get(date_str, 0.0), orderbook_imbalance=0.0,
            volatility=df['close'].pct_change().std() or 0.0),
        onchain=OnchainData(),
        news=NewsSentiment(sentiment_score=sentiment, key_events=events,
            headlines=[f'BTC at ${c:,.0f}, Fear&Greed={fng.get(date_str, 50)}']),
        macro=MacroData(fear_greed_index=fng.get(date_str, 50), btc_dominance=btc_dom.get(date_str, 0),
            fed_rate=fed.get(date_str, 0), dxy=dxy.get(date_str, 0))
    ), c


async def run_graph(graph, snapshot, debate_rounds=0):
    state = {
        'messages': [], 'data': {'snapshot': snapshot},
        'metadata': {'pair': 'BTC/USDT', 'engine': 'paper',
            'models': {k: MODEL for k in ['tech_agent', 'chain_agent', 'news_agent', 'macro_agent']},
            'debate_model': MODEL, 'verdict_model': MODEL, 'llm_verdict': True},
        'debate_round': 0, 'max_debate_rounds': debate_rounds, 'divergence_scores': [],
    }
    result = await graph.ainvoke(state)
    return result.get('data', {})


async def main():
    print("Fetching data...")
    candles = await fetch_historical('BTC/USDT', '1d',
        int(datetime.strptime(START, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000),
        int(datetime.strptime(END, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000))
    fng = await fetch_fear_greed(START, END)
    funding = await fetch_funding_rate('BTC', START, END)
    btc_dom = await fetch_btc_dominance(START, END)
    fed = await fetch_fred_series('DFF', START, END)
    dxy = await fetch_fred_series('DTWEXBGS', START, END)

    lite = build_lite_graph()
    full = build_trading_graph()

    for date in TEST_DATES:
        snapshot, price = await build_snapshot(candles, fng, funding, btc_dom, fed, dxy, date)
        if not snapshot:
            print(f"\n{date}: no data"); continue

        print(f"\n{'='*60}")
        print(f"Date: {date}  Price: ${price:,.0f}")

        # Lite (no debate)
        import time
        t0 = time.time()
        lite_data = await run_graph(lite, snapshot, debate_rounds=0)
        t_lite = time.time() - t0

        # Full (2 rounds debate)
        t0 = time.time()
        full_data = await run_graph(full, snapshot, debate_rounds=2)
        t_full = time.time() - t0

        # Compare
        print(f"\n  LITE ({t_lite:.0f}s):")
        for aid, a in lite_data.get('analyses', {}).items():
            print(f"    {aid}: {a['direction']} {a['confidence']:.0%} — {a['reasoning'][:80]}")
        lv = lite_data.get('verdict', {})
        print(f"    → VERDICT: {lv.get('action')} conf={lv.get('confidence', 0):.0%}")

        print(f"\n  FULL w/ debate ({t_full:.0f}s):")
        for aid, a in full_data.get('analyses', {}).items():
            print(f"    {aid}: {a['direction']} {a['confidence']:.0%} — {a['reasoning'][:80]}")
        fv = full_data.get('verdict', {})
        print(f"    → VERDICT: {fv.get('action')} conf={fv.get('confidence', 0):.0%}")

        changed = lv.get('action') != fv.get('action')
        print(f"\n  DECISION CHANGED: {'⚡ YES' if changed else '— no'}")

asyncio.run(main())
