"""A/B test: compare models on the same BTC market data."""
import asyncio, json, time, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import litellm
from cryptotrader.data.market import MarketCollector
from cryptotrader.data.snapshot import SnapshotAggregator
from cryptotrader.agents.tech import TechAgent, compute_indicators
from cryptotrader.agents.chain import ChainAgent
from cryptotrader.agents.news import NewsAgent
from cryptotrader.agents.macro import MacroAgent

MODELS = [
    "openai/glm-5",
    "openai/kimi-k2.5",
    "openai/claude-sonnet-4-6",
    "openai/claude-opus-4-6",
    "openai/deepseek-reasoner",
    "openai/gpt-5.2",
]

AGENT_CLASSES = [
    ("tech", TechAgent),
    ("chain", ChainAgent),
    ("news", NewsAgent),
    ("macro", MacroAgent),
]

async def test_single(agent_cls, agent_id, model, snapshot):
    agent = agent_cls(model=model)
    t0 = time.time()
    try:
        result = await agent.analyze(snapshot)
        elapsed = time.time() - t0
        return {
            "model": model.replace("openai/", ""),
            "agent": agent_id,
            "direction": result.direction,
            "confidence": result.confidence,
            "reasoning": result.reasoning[:300],
            "key_factors": result.key_factors[:5],
            "risk_flags": result.risk_flags[:5],
            "latency_s": round(elapsed, 1),
            "parse_ok": result.reasoning != "" and "mock" not in result.reasoning.lower(),
        }
    except Exception as e:
        return {
            "model": model.replace("openai/", ""),
            "agent": agent_id,
            "error": str(e)[:200],
            "latency_s": round(time.time() - t0, 1),
            "parse_ok": False,
        }

async def main():
    print("Collecting BTC/USDT market data...")
    agg = SnapshotAggregator()
    snapshot = await agg.collect("BTC/USDT", "binance", "1h", 100)
    indicators = compute_indicators(snapshot.market.ohlcv)
    print(f"Price: ${snapshot.market.ticker.get('last', 'N/A')}")
    print(f"RSI: {indicators['rsi']}, MACD hist: {indicators['macd']['histogram']}")
    print(f"Volatility: {snapshot.market.volatility:.4f}")
    print(f"Fear&Greed: {snapshot.macro.fear_greed_index}")
    print(f"\nTesting {len(MODELS)} models x {len(AGENT_CLASSES)} agents = {len(MODELS)*len(AGENT_CLASSES)} calls\n")
    print("=" * 80)

    all_results = []
    for model in MODELS:
        model_short = model.replace("openai/", "")
        print(f"\n>>> {model_short}")
        tasks = [test_single(cls, aid, model, snapshot) for aid, cls in AGENT_CLASSES]
        results = await asyncio.gather(*tasks)
        for r in results:
            all_results.append(r)
            status = "OK" if r.get("parse_ok") else "FAIL"
            direction = r.get("direction", "?")
            conf = r.get("confidence", 0)
            latency = r.get("latency_s", 0)
            print(f"  [{status}] {r['agent']:6s} → {direction:8s} conf={conf:.0%} ({latency}s)")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<22} {'Parse':>5} {'Tech':>10} {'Chain':>10} {'News':>10} {'Macro':>10} {'Avg Lat':>8}")
    print("-" * 80)
    for model in MODELS:
        ms = model.replace("openai/", "")
        mr = [r for r in all_results if r["model"] == ms]
        parse_ok = sum(1 for r in mr if r.get("parse_ok"))
        avg_lat = sum(r.get("latency_s", 0) for r in mr) / max(len(mr), 1)
        cols = []
        for aid, _ in AGENT_CLASSES:
            r = next((r for r in mr if r["agent"] == aid), None)
            if r and r.get("parse_ok"):
                cols.append(f"{r['direction'][:4]} {r['confidence']:.0%}")
            else:
                cols.append("FAIL")
        print(f"{ms:<22} {parse_ok}/4   {'  '.join(f'{c:>8}' for c in cols)}  {avg_lat:>6.1f}s")

    # Save full results
    out = "/Users/pony/Projects/cryptotrader-ai/scripts/benchmark_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull results saved to {out}")

if __name__ == "__main__":
    asyncio.run(main())
