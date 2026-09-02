"""CLI entry point."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="trader", help="CryptoTrader AI — pluggable signal fusion trading")
console = Console()
logger = logging.getLogger(__name__)


@app.callback()
def _setup():
    """Initialize logging on every CLI invocation."""
    from cryptotrader.log_config import setup_logging

    setup_logging()

    from cryptotrader.otel import setup_otel

    setup_otel()


# ── Schema operations ──

schema_app = typer.Typer(help="Database schema operations")
app.add_typer(schema_app, name="schema")


@schema_app.command("migrate")
def schema_migrate(
    database_url: Annotated[
        str,
        typer.Option("--database-url", envvar="DATABASE_URL", help="SQLAlchemy database URL"),
    ] = "",
):
    """Install the current Workbench schema without loading Runtime."""
    if not database_url.strip():
        console.print("[red]DATABASE_URL is required.[/red]")
        raise typer.Exit(1)
    asyncio.run(_schema_migrate(database_url))


async def _schema_migrate(database_url: str) -> None:
    from cryptotrader.db import dispose_engine
    from cryptotrader.migrations.workbench import migrate_workbench_schema

    try:
        await migrate_workbench_schema(database_url)
    finally:
        await dispose_engine(database_url)
    console.print("[green]Workbench database schema is current.[/green]")


@app.command()
def run(
    pair: Annotated[list[str] | None, typer.Option("--pair", "-p", help="One or more pairs")] = None,
    confirm_book: Annotated[
        list[str] | None, typer.Option("--confirm-book", help="Confirm every eligible execution book ID")
    ] = None,
):
    """Run explicitly confirmed trading for configured pairs."""
    asyncio.run(_run(pair, confirm_book))


async def _run(pairs: list[str] | None, confirmed_book_ids=None):
    from cryptotrader.runtime import build_runtime

    runtime = await build_runtime()
    try:
        if not pairs or not confirmed_book_ids:
            console.print(
                "[red]Specify --pair and every eligible --confirm-book. Review scope in the web console.[/red]"
            )
            raise typer.Exit(1)
        await _run_pairs_loop(pairs, runtime, confirmed_book_ids)
    finally:
        await runtime.close()


async def _run_pairs_loop(pairs, runtime, confirmed_book_ids):
    for pair in pairs:
        await _run_one_pair(pair, runtime, confirmed_book_ids)


async def _run_one_pair(pair: str, runtime, confirmed_book_ids) -> None:
    from cryptotrader.pair import Pair
    from cryptotrader.tracing import set_trace_id

    trace_id = set_trace_id()
    console.print(f"\n[bold]Trader[/bold] analyzing [cyan]{pair}[/cyan] trace=[dim]{trace_id}[/dim]")

    try:
        scope = await runtime.run_service.trading_scope(Pair.parse(pair))
        decision_id = await runtime.run_service.start_trading(
            Pair.parse(pair), scope.saved_revision, confirmed_book_ids
        )
        outcome = await runtime.task_manager.get(decision_id).task
        if outcome is None:
            raise RuntimeError("trading did not complete")
    except Exception:
        console.print(f"[red]Cycle failed. Trace: {trace_id}[/red]")
        console.print("[yellow]Check the per-book Journal before retrying.[/yellow]")
        raise typer.Exit(1) from None

    _print_result(pair, outcome)


def _print_result(pair: str, outcome):
    """Print one TradingCycle result."""
    from rich.table import Table

    table = Table(title=f"Decision Summary — {pair}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Pair", pair)
    table.add_row("Cycle", outcome.cycle_id)
    table.add_row("Status", outcome.status)
    if outcome.target_position is not None:
        table.add_row("Target", f"{outcome.target_position.side} {outcome.target_position.size_ratio:.2%}")
    table.add_row("Execution", outcome.execution_status)
    table.add_row("Books", str(len(outcome.books)))
    console.print(table)


# ── Journal subcommands ──

journal_app = typer.Typer(help="Decision journal commands")
app.add_typer(journal_app, name="journal")


@journal_app.command("log")
def journal_log(limit: int = typer.Option(10, "--limit", "-n")):
    """Show recent decisions."""
    asyncio.run(_journal_log(limit))


async def _journal_log(limit: int):
    from cryptotrader.journal.store import MultiVenueCycleStore
    from cryptotrader.runtime import build_runtime

    runtime = await build_runtime()
    try:
        store = MultiVenueCycleStore(runtime.repository.database_url)
        cycles = await store.list(limit=limit)
    finally:
        await runtime.close()
    if not cycles:
        console.print("[dim]No trading cycles recorded yet.[/dim]")
        return
    table = Table(title="Trading Cycle Journal")
    table.add_column("Cycle", style="cyan")
    table.add_column("Time")
    table.add_column("Source")
    table.add_column("Status")
    table.add_column("Target")
    for cycle in cycles:
        target = cycle.target_position
        target_text = "—" if target is None else f"{target.side} {target.size_ratio:.2%}"
        table.add_row(
            cycle.cycle_id,
            str(cycle.created_at),
            cycle.market_data_source_id,
            cycle.cycle_status,
            target_text,
        )
    console.print(table)


@journal_app.command("show")
def journal_show(cycle_id: str = typer.Argument(...)):
    """Show decision detail."""
    asyncio.run(_journal_show(cycle_id))


async def _journal_show(cycle_id: str):
    from cryptotrader.journal.store import MultiVenueCycleStore
    from cryptotrader.runtime import build_runtime

    runtime = await build_runtime()
    try:
        store = MultiVenueCycleStore(runtime.repository.database_url)
        cycle = await store.get(cycle_id)
    finally:
        await runtime.close()
    if not cycle:
        console.print(f"[red]Cycle {cycle_id} not found[/red]")
        return
    console.print_json(
        data={
            "cycle_id": cycle.cycle_id,
            "created_at": cycle.created_at.isoformat(),
            "status": cycle.cycle_status,
            "config_revision": cycle.config_revision,
            "market_data_source_id": cycle.market_data_source_id,
            "component_signals": list(cycle.component_signals),
            "fusion": cycle.fused_signal,
            "target_position": cycle.target_position,
            "execution_status": cycle.execution_status,
            "requires_attention": cycle.requires_attention,
            "books": list(cycle.book_results),
        }
    )


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
    from cryptotrader.backtest.models import BacktestParams
    from cryptotrader.backtest.service import configured_service

    console.print(f"[bold]Backtest[/bold] {pair} from {start} to {end} ({interval})")
    run = await configured_service().run(
        BacktestParams(pair=pair, start=start, end=end, interval=interval, initial_equity=capital)
    )
    console.print(f"运行记录：{run.run_id} · {run.status} · /research/{run.run_id}")
    if run.status != "completed":
        console.print(run.error)
        raise typer.Exit(1)
    result = run.result
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in result.summary().items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def web(
    port: int = typer.Option(5173, "--port"),
):
    """Start the React web frontend dev server (requires pnpm)."""
    import subprocess
    from pathlib import Path

    web_dir = Path(__file__).resolve().parent.parent.parent / "web"
    if not (web_dir / "package.json").exists():
        console.print(f"[red]web/ directory not found at {web_dir}[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]Starting web frontend[/bold] at http://localhost:{port}")
    subprocess.run(["pnpm", "dev", "--port", str(port)], cwd=web_dir)


@app.command()
def serve(
    port: int = typer.Option(8003, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev only)"),
    host: str = typer.Option("0.0.0.0", "--host"),
):
    """Start FastAPI server."""
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


# ── Skills subcommands ──

skills_app = typer.Typer(help="Agent Skills commands")
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list():
    """List all known skills and their scopes."""
    from pathlib import Path

    from cryptotrader.agents.skills.loader import parse_skill_md

    skills_dir = Path("agent_skills/_internal")
    if not skills_dir.exists():
        console.print("[dim]No agent_skills/_internal/ directory found.[/dim]")
        return

    table = Table(title="Agent Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Scope")
    table.add_column("Description")

    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            skill = parse_skill_md(skill_file)
            table.add_row(skill.name, skill.scope, skill.description or "")
        except Exception as exc:
            table.add_row(skill_file.parent.name, "[red]corrupt[/red]", str(exc))

    console.print(table)


# ── Agent subcommands ──

agent_app = typer.Typer(help="Signal component commands")
app.add_typer(agent_app, name="agent")


@agent_app.command("list")
def agent_list():
    """List signal components registered by the database-backed Runtime."""
    asyncio.run(_agent_list())


async def _agent_list() -> None:
    from cryptotrader.runtime import build_runtime

    runtime = await build_runtime()
    try:
        enabled = {item.component_id for item in runtime.snapshot.document.signals.components if item.enabled}
        table = Table(title="Registered Signal Components")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Status")
        table.add_column("Description")
        for component in runtime.signal_registry.metadata():
            status = "[green]enabled[/green]" if component.component_id in enabled else "[dim]disabled[/dim]"
            table.add_row(
                component.component_id,
                component.display_name,
                status,
                component.description,
            )
        console.print(table)
    finally:
        await runtime.close()


# ── MCP subcommands ──

mcp_app = typer.Typer(help="MCP data layer management commands")
app.add_typer(mcp_app, name="mcp")


def _mcp_config():
    """Return the code-owned MCP catalog; it is independent of trading config."""
    from cryptotrader.mcp.config import MCPConfig, MCPServerConfig

    return MCPConfig(
        enabled=False,
        servers=[
            MCPServerConfig(name="cryptotrader-binance"),
            MCPServerConfig(name="cryptotrader-macro"),
            MCPServerConfig(name="cryptotrader-onchain"),
            MCPServerConfig(name="cryptotrader-news"),
        ],
    )


@mcp_app.command("list")
def mcp_list():
    """List code-owned MCP servers without loading trading configuration."""
    config = _mcp_config()
    if not config.enabled:
        console.print("[yellow]MCP is disabled. Showing installed servers:[/yellow]")

    table = Table(title="MCP Servers & Tools")
    table.add_column("Server", style="cyan")
    table.add_column("Transport")
    table.add_column("Enabled")
    table.add_column("Tools")
    for server in config.servers:
        enabled = "[green]yes[/green]" if config.enabled and server.enabled else "[red]no[/red]"
        tools = ", ".join(server.tools) if server.tools else "<auto-discover>"
        table.add_row(server.name, server.transport, enabled, tools)
    console.print(table)


if __name__ == "__main__":
    app()
