# ruff: noqa: RUF001 -- Chinese user-facing messages use Chinese punctuation.
"""Compare the configured fusion profile with a Kronos-only profile."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


async def _run(label: str, snapshot, args: argparse.Namespace, service):
    from cryptotrader.backtest.models import BacktestParams

    run = await service.run(
        BacktestParams(
            pair=args.pair,
            start=args.start,
            end=args.end,
            interval=args.interval,
            name=label,
        ),
        snapshot=snapshot,
    )
    return label, run


async def main() -> None:
    from cryptotrader.backtest.comparison import compare_runs
    from cryptotrader.backtest.service import configured_service
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot

    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="BTC/USDT:USDT")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval", default="4h")
    args = parser.parse_args()

    service = configured_service()
    active = await service.repository.get_existing()
    kronos = next((item for item in active.document.signals.components if item.component_id == "kronos"), None)
    if kronos is None:
        raise ValueError("当前配置没有Kronos组件，无法构造对照实验")
    kronos_only = RuntimeConfigSnapshot(
        active.revision,
        active.document.model_copy(
            update={
                "signals": active.document.signals.model_copy(
                    update={
                        "components": (kronos.model_copy(update={"enabled": True, "weight": 1.0}),),
                    }
                ),
            }
        ),
        active.updated_at,
    )
    experiments = (
        await _run("configured fusion", active, args, service),
        await _run("Kronos only", kronos_only, args, service),
    )
    comparison = compare_runs(experiments[0][1], experiments[1][1])
    print(
        f"条件可比：{comparison.comparable}；条件差异：{list(comparison.condition_differences)}；配置差异：{list(comparison.configuration_differences)}；不自动排名"
    )
    for label, run in experiments:
        print(f"{label}: {run.run_id} · {run.status} · /research/{run.run_id}")
        if run.status != "completed":
            print(run.error)
            continue
        summary = run.result.summary()
        print(
            f"{label:20} return={summary['total_return']:>9} "
            f"sharpe={summary['sharpe_ratio']:>6} drawdown={summary['max_drawdown']:>9}"
        )


if __name__ == "__main__":
    asyncio.run(main())
