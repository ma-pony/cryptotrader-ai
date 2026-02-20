"""CLI entry point — arena command."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="arena", help="CryptoTrader AI — Multi-agent debate trading")
console = Console()


@app.command()
def run(
    pair: str = typer.Option("BTC/USDT", "--pair", "-p"),
    mode: str = typer.Option("paper", "--mode", "-m", help="paper or live"),
    exchange: str = typer.Option("binance", "--exchange", "-e"),
):
    """Run one analysis cycle."""
    asyncio.run(_run(pair, mode, exchange))


async def _run(pair: str, mode: str, exchange_id: str):
    from cryptotrader.graph import build_trading_graph, ArenaState

    console.print(f"[bold]Arena[/bold] analyzing [cyan]{pair}[/cyan] mode=[green]{mode}[/green]")

    graph = build_trading_graph()
    initial: ArenaState = {
        "messages": [],
        "data": {},
        "metadata": {
            "pair": pair,
            "engine": mode,
            "exchange_id": exchange_id,
            "timeframe": "1h",
            "ohlcv_limit": 100,
        },
        "debate_round": 0,
        "max_debate_rounds": 3,
        "divergence_scores": [],
    }

    result = await graph.ainvoke(initial)

    verdict = result.get("data", {}).get("verdict", {})
    risk = result.get("data", {}).get("risk_gate", {})
    order = result.get("data", {}).get("order")

    table = Table(title="Decision Summary")
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
        "hash": commit.hash,
        "pair": commit.pair,
        "timestamp": str(commit.timestamp),
        "debate_rounds": commit.debate_rounds,
        "divergence": commit.divergence,
        "verdict": commit.verdict.action if commit.verdict else None,
        "risk_gate": commit.risk_gate.passed if commit.risk_gate else None,
    })


@app.command()
def serve(port: int = typer.Option(8003, "--port")):
    """Start FastAPI server."""
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    app()
