"""Run the configured component profile through the shared TradingCycle."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="BTC/USDT:USDT")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--capital", type=float, default=10_000.0)
    return parser.parse_args()


async def main() -> None:
    from cryptotrader.backtest.engine import BacktestEngine

    args = _arguments()
    engine = BacktestEngine(
        pair=args.pair,
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_capital=args.capital,
    )
    result = await engine.run()
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
    print(f"config revisions: {sorted(set(result.config_revisions))}")
    print(f"cycles: {len(result.cycle_ids)}")


if __name__ == "__main__":
    asyncio.run(main())
