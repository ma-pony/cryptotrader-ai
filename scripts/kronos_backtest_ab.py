"""Compare the configured fusion profile with a Kronos-only profile."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


async def _run(label: str, snapshot, args: argparse.Namespace):
    from cryptotrader.backtest.engine import BacktestEngine

    result = await BacktestEngine(
        pair=args.pair,
        start=args.start,
        end=args.end,
        interval=args.interval,
        snapshot=snapshot,
    ).run()
    return label, result


async def main() -> None:
    from cryptotrader.bootstrap import BootstrapSettings
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot, SignalComponentConfig
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault

    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="BTC/USDT:USDT")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", default="4h")
    args = parser.parse_args()

    settings = BootstrapSettings.from_environment()
    repository = RuntimeConfigRepository(settings.database_url, CredentialVault(settings.config_master_key))
    active = await repository.get_or_create()
    kronos_only = RuntimeConfigSnapshot(
        active.revision,
        active.document.model_copy(
            update={
                "signals": active.document.signals.model_copy(
                    update={
                        "components": (SignalComponentConfig(component_id="kronos", enabled=True, weight=1.0),),
                    }
                ),
            }
        ),
        active.updated_at,
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
