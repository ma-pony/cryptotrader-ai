"""Trigger ONE full Kronos live cycle (with execution) — opens position cleanly.

Mirrors scheduler._run_cycle_impl for signal_engine=kronos: builds the full
build_kronos_graph (incl. execution + journal) and runs one cycle. Use after
flattening to open the fresh Kronos position immediately rather than waiting
for the next 4h scheduled cycle.
"""

from __future__ import annotations

import asyncio


async def main() -> int:
    from cryptotrader.config import load_config
    from cryptotrader.graph import build_kronos_graph
    from cryptotrader.state import build_initial_state

    cfg = load_config()
    pairs = cfg.scheduler.pairs
    pair = pairs[0].canonical() if pairs else "BTC/USDT:USDT"
    k = cfg.kronos
    print(f"Triggering ONE Kronos cycle — pair={pair} engine={cfg.engine} signal={cfg.signal_engine}")

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
        engine=cfg.engine,
        exchange_id=cfg.scheduler.exchange_id,
        timeframe=k.timeframe,
        ohlcv_limit=k.ohlcv_limit,
        config=cfg,
        extra_metadata=extra_meta,
    )
    graph = build_kronos_graph()
    result = await graph.ainvoke(initial)

    data = result.get("data", {})
    v = data.get("verdict", {})
    km = data.get("kronos_meta", {})
    print("\n=== CYCLE RESULT ===")
    print(f"  action={v.get('action')} conf={v.get('confidence')} scale={v.get('position_scale')}")
    print(
        f"  target_scale={km.get('target_position_scale')} "
        f"risk_multiple={km.get('risk_multiple')} signal={km.get('signal')} gate_proba={km.get('gate_proba')}"
    )
    print(f"  SL={v.get('stop_loss')} TP={v.get('take_profit')}")
    print(f"  order={data.get('order')}")
    print(f"  algo_id={data.get('algo_id')}")
    print(f"  execution_error={data.get('execution_error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
