"""CLI entry point — arena command."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="arena", help="CryptoTrader AI — Multi-agent debate trading")
console = Console()


@app.command()
def run(
    pair: list[str] = typer.Option(["BTC/USDT"], "--pair", "-p", help="One or more pairs"),
    mode: str = typer.Option("paper", "--mode", "-m", help="paper or live"),
    exchange: str = typer.Option("binance", "--exchange", "-e"),
):
    """Run one analysis cycle for each pair sequentially."""
    asyncio.run(_run(pair, mode, exchange))


async def _run(pairs: list[str], mode: str, exchange_id: str):
    from cryptotrader.graph import build_trading_graph, ArenaState

    graph = build_trading_graph()

    for pair in pairs:
        console.print(f"\n[bold]Arena[/bold] analyzing [cyan]{pair}[/cyan] mode=[green]{mode}[/green]")

        initial: ArenaState = {
            "messages": [], "data": {}, "metadata": {
                "pair": pair, "engine": mode, "exchange_id": exchange_id,
                "timeframe": "1h", "ohlcv_limit": 100,
                "analysis_model": os.environ.get("ARENA_ANALYSIS_MODEL", "openai/deepseek-chat"),
                "debate_model": os.environ.get("ARENA_DEBATE_MODEL", "openai/claude-sonnet-4-6"),
                "models": {
                    "tech_agent": os.environ.get("ARENA_MODEL_TECH", "openai/deepseek-chat"),
                    "chain_agent": os.environ.get("ARENA_MODEL_CHAIN", "openai/deepseek-chat"),
                    "news_agent": os.environ.get("ARENA_MODEL_NEWS", "openai/deepseek-chat"),
                    "macro_agent": os.environ.get("ARENA_MODEL_MACRO", "openai/claude-sonnet-4-6"),
                },
            },
            "debate_round": 0, "max_debate_rounds": 3, "divergence_scores": [],
        }

        result = await graph.ainvoke(initial)

        verdict = result.get("data", {}).get("verdict", {})
        risk = result.get("data", {}).get("risk_gate", {})
        order = result.get("data", {}).get("order")

        table = Table(title=f"Decision Summary — {pair}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Pair", pair)
        table.add_row("Action", verdict.get("action", "N/A"))
        table.add_row("Confidence", f"{verdict.get('confidence', 0):.2%}")
        table.add_row("Divergence", f"{verdict.get('divergence', 0):.2%}")
        table.add_row("Position Scale", f"{verdict.get('position_scale', 0):.2%}")
        table.add_row("Risk Gate", "PASS" if risk.get("passed") else f"REJECT: {risk.get('reason', '')}")
        if order:
            table.add_row("Order", f"{order.get('side', '')} {order.get('amount', 0):.6f} @ {order.get('price', 0):.2f}")
        console.print(table)


# ── Journal subcommands ──

journal_app = typer.Typer(help="Decision journal commands")
app.add_typer(journal_app, name="journal")


@journal_app.command("log")
def journal_log(limit: int = typer.Option(10, "--limit", "-n")):
    """Show recent decisions."""
    asyncio.run(_journal_log(limit))


async def _journal_log(limit: int):
    from cryptotrader.journal.store import JournalStore
    store = JournalStore(None)
    commits = await store.log(limit=limit)
    if not commits:
        console.print("[dim]No decisions recorded yet.[/dim]")
        return
    table = Table(title="Decision Journal")
    table.add_column("Hash", style="cyan")
    table.add_column("Time")
    table.add_column("Pair")
    table.add_column("Action")
    for c in commits:
        action = c.verdict.action if c.verdict else "N/A"
        table.add_row(c.hash, str(c.timestamp), c.pair, action)
    console.print(table)


@journal_app.command("show")
def journal_show(hash: str = typer.Argument(...)):
    """Show decision detail."""
    asyncio.run(_journal_show(hash))


async def _journal_show(hash: str):
    from cryptotrader.journal.store import JournalStore
    store = JournalStore(None)
    commit = await store.show(hash)
    if not commit:
        console.print(f"[red]Commit {hash} not found[/red]")
        return
    console.print_json(data={
        "hash": commit.hash, "pair": commit.pair,
        "timestamp": str(commit.timestamp),
        "debate_rounds": commit.debate_rounds,
        "divergence": commit.divergence,
        "verdict": commit.verdict.action if commit.verdict else None,
        "risk_gate": commit.risk_gate.passed if commit.risk_gate else None,
    })


# ── Backtest command ──

@app.command()
def backtest(
    pair: str = typer.Option("BTC/USDT", "--pair", "-p"),
    start: str = typer.Option(..., "--start", "-s"),
    end: str = typer.Option(..., "--end", "-e"),
    interval: str = typer.Option("4h", "--interval", "-i"),
    capital: float = typer.Option(10000, "--capital"),
):
    """Run backtest on historical data."""
    asyncio.run(_backtest(pair, start, end, interval, capital))


async def _backtest(pair: str, start: str, end: str, interval: str, capital: float):
    from cryptotrader.backtest.engine import BacktestEngine
    console.print(f"[bold]Backtest[/bold] {pair} from {start} to {end} ({interval})")
    engine = BacktestEngine(pair, start, end, interval, capital)
    result = await engine.run()
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in result.summary().items():
        table.add_row(k, str(v))
    console.print(table)


# ── Scheduler commands ──

scheduler_app = typer.Typer(help="Scheduler commands")
app.add_typer(scheduler_app, name="scheduler")


@scheduler_app.command("start")
def scheduler_start():
    """Start the trading scheduler."""
    asyncio.run(_scheduler_start())


async def _scheduler_start():
    from cryptotrader.scheduler import Scheduler
    from cryptotrader.config import load_config
    config = load_config()
    pairs = getattr(config, 'scheduler_pairs', ["BTC/USDT", "ETH/USDT"])
    interval = getattr(config, 'scheduler_interval', 240)
    console.print(f"[bold]Scheduler[/bold] starting: {pairs} every {interval}m")
    s = Scheduler(pairs, interval)
    await s.start()


@scheduler_app.command("status")
def scheduler_status():
    """Show scheduler status."""
    console.print("[dim]Scheduler not running (use 'arena scheduler start')[/dim]")


# ── Dashboard command ──

@app.command()
def dashboard():
    """Launch Streamlit dashboard."""
    import subprocess, sys
    from pathlib import Path
    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


@app.command()
def serve(port: int = typer.Option(8003, "--port")):
    """Start FastAPI server."""
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    app()
