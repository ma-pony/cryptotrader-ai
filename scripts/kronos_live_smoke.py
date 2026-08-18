"""One-shot Kronos cycle smoke test against LIVE OKX data (no execution).

Runs: collect_snapshot (OKX 4h/512) → tag_regime → kronos_signal (full live
aux features) → risk_gate. Stops before execution. Prints the verdict + risk
decision so we can confirm the full Kronos pipeline works on live data before
starting the scheduler loop (which WOULD place demo orders).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/kronos_live_smoke.py
"""

from __future__ import annotations

import asyncio
import sys


async def main() -> int:
    from cryptotrader.config import load_config
    from cryptotrader.graph import build_kronos_backtest_graph
    from cryptotrader.state import build_initial_state

    config = load_config()
    if config.signal_engine != "kronos":
        print(f"⚠️  signal_engine={config.signal_engine!r} (expected 'kronos')")

    pairs = config.scheduler.pairs
    pair = pairs[0].canonical() if pairs else "BTC/USDT:USDT"
    k = config.kronos

    print(f"Kronos live smoke — pair={pair}  exchange={config.exchange_id}  engine={config.engine}")
    print(f"  timeframe={k.timeframe} ohlcv_limit={k.ohlcv_limit} device={k.device} lookback={k.lookback}")
    print("  (signal + risk only — NO order execution)\n")

    extra_meta = {
        "cycle_count": 0,
        "kronos_cfg": {
            "gate_path": k.gate_path,
            "model_name": k.model_name,
            "tokenizer_name": k.tokenizer_name,
            "device": k.device,
            "lookback": k.lookback,
            "pred_len": k.pred_len,
            "sample_count": k.sample_count,
            "step2_short_threshold": k.step2_short_threshold,
            "aux_symbol": k.aux_symbol,
        },
    }

    initial = build_initial_state(
        pair,
        engine=config.engine,
        exchange_id=config.scheduler.exchange_id,
        timeframe=k.timeframe,
        ohlcv_limit=k.ohlcv_limit,
        config=config,
        extra_metadata=extra_meta,
    )

    graph = build_kronos_backtest_graph()
    result = await graph.ainvoke(initial)

    data = result.get("data", {})
    verdict = data.get("verdict", {})
    kmeta = data.get("kronos_meta", {})
    risk = data.get("risk_gate") or data.get("risk_result") or {}

    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    for key in ("action", "confidence", "position_scale", "stop_loss", "take_profit", "reasoning"):
        print(f"  {key:16}: {verdict.get(key)}")
    if kmeta:
        print("\nKRONOS META")
        for k2, v2 in kmeta.items():
            print(f"  {k2:16}: {v2}")
    if risk:
        print("\nRISK GATE")
        print(f"  {risk}")
    print("\n✅ Pipeline ran end-to-end on live OKX data (no order placed).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
