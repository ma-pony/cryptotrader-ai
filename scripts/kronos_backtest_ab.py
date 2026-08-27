"""Compare the configured fusion profile with a Kronos-only profile."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


class _ProfileRepository:
    def __init__(self, profile) -> None:
        self.profile = profile

    async def get(self):
        return self.profile


async def _run(label: str, profile, args: argparse.Namespace):
    from cryptotrader.backtest.engine import BacktestEngine

    result = await BacktestEngine(
        pair=args.pair,
        start=args.start,
        end=args.end,
        interval=args.interval,
        profile_repository=_ProfileRepository(profile),
    ).run()
    return label, result


async def main() -> None:
    from cryptotrader.config import load_config
    from cryptotrader.profiles.models import ComponentWeight

    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="BTC/USDT:USDT")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", default="4h")
    args = parser.parse_args()

    active = load_config().signal_profile_defaults.to_profile()
    kronos_only = replace(
        active,
        components=(
            ComponentWeight("kronos", True, 1.0),
            ComponentWeight("llm_committee", False, 0.0),
        ),
    )
    for label, result in (
        await _run("configured fusion", active, args),
        await _run("Kronos only", kronos_only, args),
    ):
        summary = result.summary()
        print(
            f"{label:20} return={summary['total_return']:>9} "
            f"sharpe={summary['sharpe_ratio']:>6} drawdown={summary['max_drawdown']:>9}"
        )


if __name__ == "__main__":
    asyncio.run(main())
