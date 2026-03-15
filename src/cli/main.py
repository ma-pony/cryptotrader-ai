"""CLI entry point — arena command."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="arena", help="CryptoTrader AI — Multi-agent debate trading")
console = Console()
logger = logging.getLogger(__name__)


@app.callback()
def _setup():
    """Initialize logging on every CLI invocation."""
    from cryptotrader.log_config import setup_logging

    setup_logging()

    from cryptotrader.otel import setup_otel

    setup_otel()


@app.command()
def run(
    pair: Annotated[list[str] | None, typer.Option("--pair", "-p", help="One or more pairs")] = None,
    mode: Annotated[str, typer.Option("--mode", "-m", help="paper or live")] = "paper",
    exchange: Annotated[str, typer.Option("--exchange", "-e", help="Exchange (default: from config)")] = "",
    graph: Annotated[str, typer.Option("--graph", "-g", help="full, lite, debate, supervisor")] = "full",
):
    """Run one analysis cycle for each pair sequentially."""
    if pair is None:
        from cryptotrader.config import load_config

        pair = load_config().scheduler.pairs or ["BTC/USDT"]
    asyncio.run(_run(pair, mode, exchange, graph))


async def _run(pairs: list[str], mode: str, exchange_id: str, graph_mode: str = "full"):
    from cryptotrader.config import load_config
    from cryptotrader.graph import ArenaState, build_debate_graph, build_lite_graph, build_trading_graph
    from cryptotrader.tracing import set_trace_id

    config = load_config()
    if not exchange_id:
        exchange_id = config.exchange_id
    builders = {
        "full": build_trading_graph,
        "lite": build_lite_graph,
        "debate": build_debate_graph,
    }
    if graph_mode == "supervisor":
        from cryptotrader.graph import build_supervisor_graph_v2

        graph = build_supervisor_graph_v2()
    else:
        graph = builders.get(graph_mode, build_trading_graph)()

    # Live mode pre-flight checks
    if mode == "live":
        creds = config.exchanges.get(exchange_id)
        if creds is None or not creds.api_key or not creds.secret:
            console.print(
                f"[red]ERROR: No credentials configured for exchange '{exchange_id}'.[/red]\n"
                f"Set api_key/secret in config/local.toml under [exchanges.{exchange_id}]"
            )
            raise typer.Exit(1)
        if creds.sandbox:
            console.print("[yellow]WARNING: Running in SANDBOX mode (sandbox=true in config)[/yellow]")

    for pair in pairs:
        trace_id = set_trace_id()
        console.print(
            f"\n[bold]Arena[/bold] analyzing [cyan]{pair}[/cyan] mode=[green]{mode}[/green] trace=[dim]{trace_id}[/dim]"
        )

        initial: ArenaState = {
            "messages": [],
            "data": {},
            "metadata": {
                "pair": pair,
                "engine": mode,
                "exchange_id": exchange_id,
                "timeframe": config.data.default_timeframe,
                "ohlcv_limit": config.data.ohlcv_limit,
                "analysis_model": config.models.analysis,
                "debate_model": config.models.debate,
                "verdict_model": config.models.verdict,
                "models": {
                    "tech_agent": config.models.tech_agent,
                    "chain_agent": config.models.chain_agent,
                    "news_agent": config.models.news_agent,
                    "macro_agent": config.models.macro_agent,
                },
                "database_url": config.infrastructure.database_url,
                "redis_url": config.infrastructure.redis_url,
                "convergence_threshold": config.debate.convergence_threshold,
                "max_single_pct": config.risk.position.max_single_pct,
            },
            "debate_round": 0,
            "max_debate_rounds": config.debate.max_rounds,
            "divergence_scores": [],
        }

        from cryptotrader.tracing import add_timing_to_trace, run_graph_traced

        try:
            result, node_trace = await run_graph_traced(graph, initial)
            add_timing_to_trace(node_trace)
        except Exception as exc:
            if mode == "live":
                console.print(f"[red]ERROR: {exc}[/red]")
                console.print("[yellow]Check logs — a partial trade may have been placed.[/yellow]")
                raise typer.Exit(1) from None
            raise

        _print_result(pair, result, node_trace)


def _print_result(pair: str, result: dict, node_trace: list[dict]):
    """Print node trace and decision summary tables."""
    from rich.table import Table

    verdict = result.get("data", {}).get("verdict", {})
    risk = result.get("data", {}).get("risk_gate", {})
    order = result.get("data", {}).get("order")

    trace_table = Table(title="Graph Node Trace")
    trace_table.add_column("Node", style="cyan")
    trace_table.add_column("Duration", style="yellow")
    trace_table.add_column("Output", style="white")
    for t in node_trace:
        trace_table.add_row(t["node"], f"{t['duration_ms']}ms", t["summary"][:120])
    console.print(trace_table)

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
    from cryptotrader.config import load_config

    config = load_config()
    from cryptotrader.journal.store import JournalStore

    store = JournalStore(config.infrastructure.database_url)
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
    from cryptotrader.config import load_config
    from cryptotrader.journal.store import JournalStore

    config = load_config()
    store = JournalStore(config.infrastructure.database_url)
    commit = await store.show(hash)
    if not commit:
        console.print(f"[red]Commit {hash} not found[/red]")
        return
    console.print_json(
        data={
            "hash": commit.hash,
            "pair": commit.pair,
            "timestamp": str(commit.timestamp),
            "debate_rounds": commit.debate_rounds,
            "divergence": commit.divergence,
            "verdict": commit.verdict.action if commit.verdict else None,
            "risk_gate": commit.risk_gate.passed if commit.risk_gate else None,
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
    use_llm: bool = typer.Option(True, "--use-llm/--no-llm", help="Use AI agents (default) or SMA crossover"),
):
    """Run backtest on historical data."""
    asyncio.run(_backtest(pair, start, end, interval, capital, use_llm))


async def _backtest(pair: str, start: str, end: str, interval: str, capital: float, use_llm: bool):
    from cryptotrader.backtest.engine import BacktestEngine

    mode = "AI agents" if use_llm else "SMA crossover"
    console.print(f"[bold]Backtest[/bold] {pair} from {start} to {end} ({interval}) [{mode}]")
    engine = BacktestEngine(pair, start, end, interval, capital, use_llm=use_llm)
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
    from cryptotrader.config import load_config
    from cryptotrader.scheduler import Scheduler

    config = load_config()
    if not config.scheduler.enabled:
        console.print("[red]Scheduler is disabled in config (scheduler.enabled=false)[/red]")
        raise typer.Exit(1)
    pairs = config.scheduler.pairs
    interval = config.scheduler.interval_minutes
    summary_hour = config.scheduler.daily_summary_hour
    console.print(
        f"[bold]Scheduler[/bold] starting: {pairs} every {interval}m (daily summary at {summary_hour}:00 UTC)"
    )
    s = Scheduler(pairs, interval, daily_summary_hour=summary_hour)
    await s.start()


@scheduler_app.command("status")
def scheduler_status():
    """Show scheduler status."""
    asyncio.run(_scheduler_status())


async def _scheduler_status():
    from cryptotrader.config import load_config
    from cryptotrader.portfolio.manager import PortfolioManager

    config = load_config()
    db_url = config.infrastructure.database_url
    pm = PortfolioManager(db_url)
    try:
        portfolio = await pm.get_portfolio()
        daily_pnl = await pm.get_daily_pnl()
        drawdown = await pm.get_drawdown()
    except Exception:
        logger.debug("Failed to load portfolio status", exc_info=True)
        portfolio = {"total_value": 0, "positions": {}}
        daily_pnl = 0.0
        drawdown = 0.0

    table = Table(title="Portfolio Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Value", f"${portfolio.get('total_value', 0):,.2f}")
    table.add_row("Daily PnL", f"${daily_pnl:,.2f}")
    table.add_row("Drawdown", f"{drawdown:.2%}")
    positions = portfolio.get("positions", {})
    if positions:
        for pair, pos in positions.items():
            table.add_row(f"  {pair}", f"{pos['amount']:.6f} @ ${pos['avg_price']:,.2f}")
    else:
        table.add_row("Positions", "(none)")
    console.print(table)


# ── Migrate command ──


@app.command()
def migrate():
    """Apply database schema migrations (create tables if needed)."""
    asyncio.run(_migrate())


async def _migrate():
    from cryptotrader.config import load_config

    config = load_config()
    db_url = config.infrastructure.database_url
    if not db_url:
        console.print("[red]DATABASE_URL not configured — nothing to migrate.[/red]")
        raise typer.Exit(1)
    from sqlalchemy.ext.asyncio import create_async_engine

    from cryptotrader.journal.store import _sa_models

    Base, _ = _sa_models()
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    console.print("[green]Database tables created / verified.[/green]")


# ── Dashboard command ──


@app.command()
def dashboard():
    """Launch Streamlit dashboard."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


@app.command()
def sync():
    """Sync and persist historical market data from all sources."""
    asyncio.run(_sync())


async def _sync():
    from cryptotrader.config import load_config
    from cryptotrader.data.sync import sync_all

    config = load_config()
    console.print("[bold]Syncing market data...[/bold]")
    results = await sync_all(config.providers)
    table = Table(title="Data Sync Results")
    table.add_column("Source", style="cyan")
    table.add_column("Records", style="green")
    for source, count in sorted(results.items()):
        table.add_row(source, str(count))
    console.print(table)


@app.command()
def serve(
    port: int = typer.Option(8003, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev only)"),
    host: str = typer.Option("0.0.0.0", "--host"),
):
    """Start FastAPI server."""
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


# ── Risk subcommands ──

risk_app = typer.Typer(help="Risk management commands")
app.add_typer(risk_app, name="risk")


@risk_app.command("reset-breaker")
def risk_reset_breaker():
    """Reset the daily-loss circuit breaker to allow trading to resume."""
    asyncio.run(_risk_reset_breaker())


async def _risk_reset_breaker():
    from cryptotrader.config import load_config
    from cryptotrader.risk.state import RedisStateManager

    config = load_config()
    redis_url = config.infrastructure.redis_url
    redis_state = RedisStateManager(redis_url)
    await redis_state.reset_circuit_breaker()
    console.print("[green]Circuit breaker reset — trading is now allowed.[/green]")


@app.command("live-check")
def live_check(
    exchange: Annotated[str, typer.Option("--exchange", "-e", help="Exchange (default: from config)")] = "",
):
    """Run pre-flight checks for live trading."""
    asyncio.run(_live_check(exchange))


def _check_credentials(config, exchange_id: str) -> tuple[str, bool, str]:
    creds = config.exchanges.get(exchange_id)
    if creds and creds.api_key and creds.secret:
        sandbox_note = " (SANDBOX)" if creds.sandbox else ""
        return ("Credentials", True, f"{exchange_id}{sandbox_note}")
    missing = []
    if creds is None or not creds.api_key:
        missing.append("api_key")
    if creds is None or not creds.secret:
        missing.append("secret")
    fields = ", ".join(missing) if missing else "api_key/secret"
    hint = f"config/local.toml [exchanges.{exchange_id}]"
    return ("Credentials", False, f"No credentials for {exchange_id} — {fields} missing. Set in {hint}")


async def _check_exchange_api(config, exchange_id: str) -> tuple[str, bool, str]:
    import time

    creds = config.exchanges.get(exchange_id)
    if not creds or not creds.api_key:
        return ("Exchange API", False, "Skipped (no credentials)")
    try:
        from cryptotrader.execution.exchange import LiveExchange

        ex = LiveExchange(exchange_id, creds.api_key, creds.secret, sandbox=creds.sandbox, passphrase=creds.passphrase)
        t0 = time.monotonic()
        bal = await ex.get_balance()
        latency = int((time.monotonic() - t0) * 1000)
        await ex.close()
        return ("Exchange API", True, f"{latency}ms latency, {len(bal)} assets")
    except Exception as e:
        return ("Exchange API", False, str(e))


async def _check_redis(config) -> tuple[str, bool, str]:
    redis_url = config.infrastructure.redis_url
    if not redis_url:
        return ("Redis", False, "Not configured")
    from cryptotrader.risk.state import RedisStateManager

    rsm = RedisStateManager(redis_url)
    if await rsm.ping():
        return ("Redis", True, "Connected")
    return ("Redis", False, "Configured but unreachable")


async def _check_database(config) -> tuple[str, bool, str]:
    db_url = config.infrastructure.database_url
    if not db_url:
        return ("Database", False, "Not configured")
    try:
        from cryptotrader.portfolio.manager import PortfolioManager

        pm = PortfolioManager(db_url)
        portfolio = await pm.get_portfolio()
        total = portfolio.get("total_value", 0)
        return ("Database", True, f"Portfolio: ${total:,.2f}")
    except Exception as e:
        return ("Database", False, str(e))


async def _live_check(exchange_id: str):
    from cryptotrader.config import load_config

    config = load_config()
    if not exchange_id:
        exchange_id = config.exchange_id
    checks = [
        _check_credentials(config, exchange_id),
        await _check_exchange_api(config, exchange_id),
        await _check_redis(config),
        await _check_database(config),
    ]

    # Output
    table = Table(title=f"Live Trading Pre-flight — {exchange_id}")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")

    all_pass = True
    for name, ok, detail in checks:
        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        if not ok:
            all_pass = False
        table.add_row(name, status, detail)

    console.print(table)

    if all_pass:
        console.print("\n[bold green]GO[/bold green] — All checks passed")
    else:
        console.print("\n[bold red]NO-GO[/bold red] — Fix failing checks before live trading")
        raise typer.Exit(1)


# ── Experience subcommands ──

experience_app = typer.Typer(help="Experience memory commands")
app.add_typer(experience_app, name="experience")


@experience_app.command("distill")
def experience_distill(
    session: str = typer.Argument(..., help="Backtest session ID"),
):
    """Distill experience from a backtest session."""
    asyncio.run(_experience_distill(session))


async def _experience_distill(session_id: str):
    from cryptotrader.backtest.session import load_commits, save_experience
    from cryptotrader.learning.reflect import run_agent_reflection

    commits_raw = load_commits(session_id)
    if not commits_raw:
        console.print(f"[red]No commits found for session {session_id}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Distilling experience from {len(commits_raw)} commits...[/bold]")

    results = {}
    agent_ids = ("tech_agent", "chain_agent", "news_agent", "macro_agent")

    for agent_id in agent_ids:
        records = _extract_agent_records(commits_raw, agent_id)
        if len(records) < 3:
            console.print(f"  [dim]{agent_id}: skipped (only {len(records)} records)[/dim]")
            continue
        memory = await run_agent_reflection(agent_id, records, model="")
        results[agent_id] = memory
        console.print(
            f"  [green]{agent_id}[/green]: {len(memory.success_patterns)} patterns, {len(memory.forbidden_zones)} zones"
        )

    save_experience(session_id, results)
    console.print(f"[green]Experience saved to session {session_id}[/green]")


def _extract_agent_records(commits_raw: list[dict], agent_id: str) -> list[dict]:
    """Extract agent records from raw commit dicts for reflection."""
    records = []
    for c in commits_raw:
        analyses = c.get("analyses", {})
        agent_data = analyses.get(agent_id)
        if not agent_data:
            continue
        pnl = c.get("pnl")
        if pnl is None:
            continue
        verdict = c.get("verdict", {})
        summary = c.get("snapshot_summary", {})
        records.append(
            {
                "date": c.get("timestamp", ""),
                "direction": agent_data.get("direction", "neutral"),
                "confidence": agent_data.get("confidence", 0.5),
                "reasoning": agent_data.get("reasoning", ""),
                "key_factors": agent_data.get("key_factors", []),
                "pnl": pnl,
                "verdict_action": verdict.get("action", "hold"),
                "price": summary.get("price", 0),
                "volatility": summary.get("volatility", 0),
                "funding_rate": summary.get("funding_rate", 0),
            }
        )
    return records


@experience_app.command("show")
def experience_show(
    session: str = typer.Argument(..., help="Backtest session ID"),
):
    """Show distilled experience for a backtest session."""
    from pathlib import Path

    path = Path.home() / ".cryptotrader" / "backtest_sessions" / session / "experience.json"
    if not path.exists():
        console.print(f"[red]No experience found for session {session}. Run 'arena experience distill' first.[/red]")
        raise typer.Exit(1)
    data = _load_json_file(path)
    console.print_json(data=data)


@experience_app.command("merge")
def experience_merge(
    session: str = typer.Argument(..., help="Backtest session ID to merge"),
):
    """Merge backtest experience into live memory."""
    asyncio.run(_experience_merge(session))


async def _experience_merge(session_id: str):
    from pathlib import Path

    from cryptotrader.learning.reflect import load_reflections, save_reflection
    from cryptotrader.models import ExperienceMemory, ExperienceRule

    path = Path.home() / ".cryptotrader" / "backtest_sessions" / session_id / "experience.json"
    if not path.exists():
        console.print(f"[red]No experience found for session {session_id}[/red]")
        raise typer.Exit(1)

    data = _load_json_file(path)
    existing = await load_reflections()

    for agent_id, mem_data in data.items():
        bt_mem = ExperienceMemory(
            success_patterns=[
                ExperienceRule(**{**r, "source": "backtest", "source_session": session_id})
                for r in mem_data.get("success_patterns", [])
            ],
            forbidden_zones=[
                ExperienceRule(**{**r, "source": "backtest", "source_session": session_id})
                for r in mem_data.get("forbidden_zones", [])
            ],
            strategic_insights=mem_data.get("strategic_insights", []),
        )

        live_mem = existing.get(agent_id, ExperienceMemory())
        merged = _merge_memories(live_mem, bt_mem)
        await save_reflection(
            Path.home() / ".cryptotrader" / "agent_reflections.db",
            agent_id,
            merged,
        )
        console.print(
            f"  [green]{agent_id}[/green]: merged ({len(merged.success_patterns)} patterns, "
            f"{len(merged.forbidden_zones)} zones)"
        )

    console.print("[green]Backtest experience merged into live memory[/green]")


def _load_json_file(path) -> dict:
    """Load JSON file synchronously."""
    import json

    with open(path) as f:
        return json.load(f)


def _merge_memories(live, bt):
    """Merge backtest experience into live memory (additive)."""
    from cryptotrader.models import ExperienceMemory

    merged_patterns = list(live.success_patterns)
    for rule in bt.success_patterns:
        existing = _find_similar_rule(merged_patterns, rule)
        if existing:
            _merge_rule_stats(existing, rule)
        else:
            merged_patterns.append(rule)

    merged_zones = list(live.forbidden_zones)
    for rule in bt.forbidden_zones:
        existing = _find_similar_rule(merged_zones, rule)
        if existing:
            _merge_rule_stats(existing, rule)
        else:
            merged_zones.append(rule)

    merged_insights = list(live.strategic_insights)
    for insight in bt.strategic_insights:
        if insight not in merged_insights:
            merged_insights.append(insight)

    return ExperienceMemory(
        success_patterns=merged_patterns,
        forbidden_zones=merged_zones,
        strategic_insights=merged_insights,
    )


def _merge_rule_stats(existing, incoming):
    """Merge incoming rule stats into existing rule (weighted average rate)."""
    total = existing.sample_count + incoming.sample_count
    if total > 0:
        existing.rate = (existing.rate * existing.sample_count + incoming.rate * incoming.sample_count) / total
    existing.sample_count = total
    existing.regime_count += 1


def _find_similar_rule(rules, target):
    """Find a rule with the same pattern text."""
    for r in rules:
        if r.pattern == target.pattern:
            return r
    return None


@experience_app.command("sessions")
def experience_sessions():
    """List backtest sessions."""
    from cryptotrader.backtest.session import list_sessions

    sessions = list_sessions()
    if not sessions:
        console.print("[dim]No backtest sessions found.[/dim]")
        return
    table = Table(title="Backtest Sessions")
    table.add_column("Session ID", style="cyan")
    for s in sessions:
        table.add_row(s)
    console.print(table)


# ── Portfolio subcommands ──

portfolio_app = typer.Typer(help="Portfolio management commands")
app.add_typer(portfolio_app, name="portfolio")


@portfolio_app.command("show")
def portfolio_show():
    """Show current portfolio (cash + positions)."""
    asyncio.run(_portfolio_show())


async def _portfolio_show():
    from cryptotrader.config import load_config
    from cryptotrader.portfolio.manager import PortfolioManager

    cfg = load_config()
    pm = PortfolioManager(cfg.infrastructure.database_url)
    p = await pm.get_portfolio()

    table = Table(title="Portfolio")
    table.add_column("Item", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Cash (USDT)", f"${p.get('cash', 0):,.2f}")
    positions = p.get("positions", {})
    for pair, pos in positions.items():
        amount = pos["amount"]
        avg_price = pos["avg_price"]
        side = "Long" if amount > 0 else "Short"
        table.add_row(
            f"{pair} ({side})",
            f"{abs(amount):.6f} @ ${avg_price:,.2f} = ${abs(amount) * avg_price:,.2f}",
        )
    table.add_row("Total Value", f"[bold]${p['total_value']:,.2f}[/bold]")
    console.print(table)


@portfolio_app.command("reset")
def portfolio_reset(
    confirm: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    capital: Annotated[float, typer.Option("--capital", "-c", help="Initial cash")] = 10000.0,
):
    """Reset portfolio to initial state (delete positions + snapshots, set cash)."""
    if not confirm:
        typer.confirm(
            f"This will delete all positions and snapshots, set cash to ${capital:,.2f}. Continue?",
            abort=True,
        )
    asyncio.run(_portfolio_reset(capital))


async def _portfolio_reset(capital: float):
    from cryptotrader.config import load_config
    from cryptotrader.portfolio.manager import PortfolioManager

    cfg = load_config()
    pm = PortfolioManager(cfg.infrastructure.database_url)
    await pm.reset("default")
    await pm.update_cash("default", capital)
    await pm.snapshot("default", capital, capital)
    console.print(f"[green]Portfolio reset. Cash: ${capital:,.2f}[/green]")


if __name__ == "__main__":
    app()
