"""Kronos A/B Backtest Harness.

Runs the existing LLM backtest graph (build_backtest_graph) and a Kronos graph
over the same historical slice, then prints a side-by-side comparison table.

Usage
-----
    python scripts/kronos_backtest_ab.py \
        --start 2025-01-01 --end 2025-03-31 --pair BTC/USDT

If LLM API keys are missing the LLM side is skipped gracefully.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure src/ is on PYTHONPATH when invoked directly
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ── Stats helpers ────────────────────────────────────────────────────────────


def _compute_sharpe(equity_curve: list[float], bars_per_year: int = 2190) -> float:
    """Annualised Sharpe from equity curve (4h bars by default)."""
    if len(equity_curve) < 2:
        return 0.0
    rets = [(equity_curve[i] / equity_curve[i - 1]) - 1.0 for i in range(1, len(equity_curve))]
    mean_r = sum(rets) / len(rets)
    if len(rets) < 2:
        return 0.0
    var_r = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
    std_r = math.sqrt(var_r) if var_r > 0 else 0.0
    if std_r < 1e-12:
        return 0.0
    return (mean_r / std_r) * math.sqrt(bars_per_year)


def _compute_max_drawdown(equity_curve: list[float]) -> float:
    """Maximum drawdown fraction (negative number)."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        peak = max(peak, val)
        dd = (val - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
    return max_dd


def _compute_hit_rate(trades: list[dict]) -> float:
    """Fraction of trades with positive PnL."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return wins / len(trades)


def _summarise(result, strategy: str) -> dict:
    """Extract a uniform metrics dict from a BacktestResult."""
    if result is None:
        return {
            "strategy": strategy,
            "n_trades": 0,
            "sharpe": float("nan"),
            "total_return": float("nan"),
            "hit_rate": float("nan"),
            "max_dd": float("nan"),
            "note": "skipped",
        }

    eq = result.equity_curve
    sharpe = _compute_sharpe(eq) if len(eq) > 1 else result.sharpe_ratio
    max_dd = _compute_max_drawdown(eq) if eq else result.max_drawdown
    hit_rate = _compute_hit_rate(result.trades) if result.trades else result.win_rate

    return {
        "strategy": strategy,
        "n_trades": len(result.trades),
        "sharpe": sharpe,
        "total_return": result.total_return,
        "hit_rate": hit_rate,
        "max_dd": max_dd,
        "note": "",
    }


# ── LLM backtest ─────────────────────────────────────────────────────────────


async def run_llm_backtest(pair: str, start: str, end: str) -> tuple[Any | None, str]:
    """Run existing LLM backtest graph; return (BacktestResult | None, note)."""
    try:
        from cryptotrader.backtest.engine import BacktestEngine

        engine = BacktestEngine(pair=pair, start=start, end=end, use_llm=True)
        result = await engine.run()
        return result, ""
    except Exception as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("api key", "apikey", "openai", "anthropic", "auth")):
            note = f"LLM graph skipped (no keys): {exc}"
        else:
            note = f"LLM graph failed: {exc}"
        logger.warning(note)
        return None, note


async def run_kronos_backtest(
    pair: str,
    start: str,
    end: str,
    graph_builder=None,
) -> tuple[Any | None, str]:
    """Run the configured Kronos graph through the production backtest engine."""
    try:
        if graph_builder is None:
            from cryptotrader.graph import build_kronos_backtest_graph

            graph_builder = build_kronos_backtest_graph

        from cryptotrader.backtest.engine import BacktestEngine
        from cryptotrader.config import load_config

        kronos_cfg = load_config().kronos
        engine = BacktestEngine(
            pair=pair,
            start=start,
            end=end,
            interval=kronos_cfg.timeframe,
            lookback=kronos_cfg.lookback,
            use_llm=False,
            graph_builder=graph_builder,
            graph_metadata={"kronos_cfg": asdict(kronos_cfg)},
        )
        result = await engine.run()
        return result, ""

    except Exception as exc:
        note = f"Kronos backtest failed: {exc}"
        logger.warning(note, exc_info=True)
        return None, note


# ── Table printer ─────────────────────────────────────────────────────────────


def _fmt(val: float, pct: bool = False) -> str:
    if math.isnan(val):
        return "N/A"
    if pct:
        return f"{val:.2%}"
    return f"{val:.3f}"


def _print_table(rows: list[dict]) -> None:
    cols = ["strategy", "n_trades", "sharpe", "total_return", "hit_rate", "max_dd", "note"]
    headers = {
        "strategy": "Strategy",
        "n_trades": "Trades",
        "sharpe": "Sharpe",
        "total_return": "Total Ret",
        "hit_rate": "Hit Rate",
        "max_dd": "Max DD",
        "note": "Note",
    }
    widths = {c: max(len(headers[c]), *(len(_cell(r, c)) for r in rows)) for c in cols}

    def _row(r: dict) -> str:
        return "  ".join(_cell(r, c).ljust(widths[c]) for c in cols)

    sep = "  ".join("-" * widths[c] for c in cols)
    print()
    print("  ".join(headers[c].ljust(widths[c]) for c in cols))
    print(sep)
    for r in rows:
        print(_row(r))
    print()


def _cell(row: dict, col: str) -> str:
    val = row.get(col, "")
    if col in ("total_return", "hit_rate", "max_dd"):
        return _fmt(float(val), pct=True) if val != "" and not isinstance(val, str) else str(val)
    if col == "sharpe":
        return _fmt(float(val)) if val != "" and not isinstance(val, str) else str(val)
    return str(val)


# ── Main ─────────────────────────────────────────────────────────────────────


async def main(pair: str, start: str, end: str, graph_builder=None, skip_llm: bool = False) -> None:
    print(f"\nKronos A/B Backtest: {pair}  {start} → {end}")
    print("=" * 60)

    rows = []

    # ── LLM side ──
    if skip_llm:
        print("\nSkipping LLM side (--skip-llm).")
    else:
        print("\nRunning LLM (build_backtest_graph) …")
        llm_result, llm_note = await run_llm_backtest(pair, start, end)
        if llm_note:
            print(f"  [{llm_note}]")
        llm_summary = _summarise(llm_result, "LLM (existing)")
        if llm_note:
            llm_summary["note"] = llm_note[:60]
        rows.append(llm_summary)

    # ── Kronos side ──
    print("Running Kronos (kronos_signal node) …")
    kronos_result, kronos_note = await run_kronos_backtest(
        pair=pair,
        start=start,
        end=end,
        graph_builder=graph_builder,
    )
    if kronos_note:
        print(f"  [{kronos_note}]")
    kronos_summary = _summarise(kronos_result, "Kronos (gate_v21)")
    if kronos_note:
        kronos_summary["note"] = kronos_note[:60]
    rows.append(kronos_summary)

    # ── Print table ──
    _print_table(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos A/B backtest harness")
    parser.add_argument("--start", default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-03-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--pair", default="BTC/USDT", help="Trading pair (default BTC/USDT)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM side, validate Kronos only")
    args = parser.parse_args()

    asyncio.run(main(pair=args.pair, start=args.start, end=args.end, skip_llm=args.skip_llm))
