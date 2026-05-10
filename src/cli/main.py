"""CLI entry point — arena command."""

from __future__ import annotations

# Load .env into os.environ BEFORE any project import. Several CLI
# commands (`arena migrate`, `arena scheduler start`) need DATABASE_URL /
# REDIS_URL / API keys from .env.
from dotenv import load_dotenv

load_dotenv()

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

        cfg_pairs = load_config().scheduler.pairs
        # cfg_pairs is list[Pair] post spec 013; project canonical str expected by _run
        pair = [p.canonical() for p in cfg_pairs] if cfg_pairs else ["BTC/USDT"]
    asyncio.run(_run(pair, mode, exchange, graph))


async def _run(pairs: list[str], mode: str, exchange_id: str, graph_mode: str = "full"):
    from cryptotrader.config import load_config
    from cryptotrader.graph import build_debate_graph, build_lite_graph, build_trading_graph

    config = load_config()
    if not exchange_id:
        exchange_id = config.exchange_id
    builders = {
        "full": build_trading_graph,
        "lite": build_lite_graph,
        "debate": build_debate_graph,
    }
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

    # Cleanup hook — runs on every exit path so the cached LiveExchange (ccxt
    # async aiohttp connector) gets explicitly closed. Without this, every
    # `arena run --mode live` exit logs an aiohttp "Unclosed connector" warning
    # and a deque of dangling ResponseHandlers.
    try:
        await _run_pairs_loop(pairs, mode, exchange_id, graph, config)
    finally:
        from cryptotrader.nodes.execution import close_live_exchanges

        await close_live_exchanges()


async def _run_pairs_loop(pairs, mode, exchange_id, graph, config):
    from cryptotrader.cycle_lock import cycle_lock
    from cryptotrader.risk.state import RedisStateManager

    redis_state = RedisStateManager(config.infrastructure.redis_url)

    for pair in pairs:
        async with cycle_lock(redis_state, pair) as acquired:
            if not acquired:
                console.print(f"[yellow]Skipping {pair}: cycle_lock held (scheduler likely processing it).[/yellow]")
                continue
            await _run_one_pair(pair, mode, exchange_id, graph, config)


async def _run_one_pair(pair: str, mode: str, exchange_id: str, graph, config) -> None:
    from cryptotrader.tracing import add_timing_to_trace, run_graph_traced, set_trace_id

    trace_id = set_trace_id()
    console.print(
        f"\n[bold]Arena[/bold] analyzing [cyan]{pair}[/cyan] mode=[green]{mode}[/green] trace=[dim]{trace_id}[/dim]"
    )

    # ArenaState is a TypedDict (type-only annotation); the dict literal
    # below satisfies it structurally without an explicit annotation.
    initial = {
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


@scheduler_app.command("healthcheck")
def scheduler_healthcheck(
    max_age_seconds: int = typer.Option(
        0, help="Max heartbeat age in seconds. 0 = derive from scheduler.interval_minutes * 2."
    ),
):
    """Exit 0 if scheduler heartbeat is fresh, 1 otherwise (for docker healthcheck)."""
    import time
    from pathlib import Path

    if max_age_seconds <= 0:
        from cryptotrader.config import load_config

        max_age_seconds = max(120, load_config().scheduler.interval_minutes * 60 * 2)

    hb = Path.home() / ".cryptotrader" / "scheduler.heartbeat"
    if not hb.exists():
        console.print(f"[red]heartbeat missing: {hb}[/red]")
        raise typer.Exit(1)
    age = time.time() - hb.stat().st_mtime
    if age > max_age_seconds:
        console.print(f"[red]heartbeat stale: {age:.0f}s > {max_age_seconds}s[/red]")
        raise typer.Exit(1)
    console.print(f"[green]ok: heartbeat {age:.0f}s ago[/green]")


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
        logger.info("Failed to load portfolio status", exc_info=True)
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

        ex = LiveExchange(
            exchange_id,
            creds.api_key,
            creds.secret,
            sandbox=creds.sandbox,
            passphrase=creds.passphrase,
            leverage=creds.leverage,
            margin_mode=creds.margin_mode,
        )
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


# ── Skills subcommands ──

skills_app = typer.Typer(help="Agent Skills commands")
app.add_typer(skills_app, name="skills")


@skills_app.command("curate")
def skills_curate(
    name: str = typer.Argument(..., help="Skill name (directory name under agent_skills/)"),
    llm: bool = typer.Option(False, "--llm", help="Use LLM to generate pattern summaries"),
):
    """Curate a skill file by injecting distilled patterns into its AUTO-DISTILLED-PATTERNS section."""
    from pathlib import Path

    from cryptotrader.learning.curation import curate_skill

    skills_dir = Path("agent_skills")
    skill_path = skills_dir / name / "SKILL.md"
    if not skill_path.exists():
        console.print(f"[red]Skill not found: {skill_path}[/red]")
        raise typer.Exit(1)

    try:
        draft_path = curate_skill(name, skills_dir=skills_dir, use_llm=llm)
        console.print(f"[green]Draft written to:[/green] {draft_path}")
        console.print("[dim]Review the draft and rename to SKILL.md when ready.[/dim]")
    except Exception as exc:
        console.print(f"[red]Curation failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@skills_app.command("propose-new")
def skills_propose_new(
    scope: str = typer.Option("shared", "--scope", help="Scope: 'shared' or 'agent:<id>'"),
):
    """Propose a new skill file based on active patterns not yet covered by existing skills."""
    from pathlib import Path

    from cryptotrader.learning.skill_proposal import propose_new_skill

    skills_dir = Path("agent_skills")
    memory_dir = Path.home() / ".cryptotrader" / "agent_memory"

    try:
        draft_path = propose_new_skill(scope=scope, skills_dir=skills_dir, memory_dir=memory_dir)
        if draft_path is None:
            console.print("[dim]No new patterns found to propose a skill for.[/dim]")
        else:
            console.print(f"[green]Proposal draft written to:[/green] {draft_path}")
            console.print("[dim]Review and rename to SKILL.md when ready.[/dim]")
    except Exception as exc:
        console.print(f"[red]Proposal failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@skills_app.command("list")
def skills_list():
    """List all known skills and their scopes."""
    from pathlib import Path

    from cryptotrader.agents.skills.loader import parse_skill_md

    skills_dir = Path("agent_skills")
    if not skills_dir.exists():
        console.print("[dim]No agent_skills/ directory found.[/dim]")
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


@portfolio_app.command("reset-baseline")
def portfolio_reset_baseline(
    reason: Annotated[str, typer.Option("--reason", "-r", help="Why are you resetting? (audit log)")] = "",
    operator: Annotated[str, typer.Option("--operator", help="Who is doing this (audit log)")] = "",
    account_id: Annotated[str, typer.Option("--account-id", help="Account to reset")] = "default",
    confirm: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")] = False,
):
    """Acknowledge current equity as the new drawdown peak baseline.

    Subsequent ``DrawdownLimit`` checks measure peak/trough only against
    snapshots taken AFTER this reset, so historical losses no longer
    permanently gate new trades. Writes an audit row including the operator
    and the reason — required for any production-grade reset.

    Use this AFTER deciding the past drawdown is "accepted history" and
    you want the system to start a fresh baseline. This does NOT delete any
    snapshot data; only the drawdown computation window changes.
    """
    if not reason:
        console.print("[red]--reason is required for audit log[/red]")
        raise typer.Exit(2)
    if not operator:
        import getpass

        operator = getpass.getuser() or "unknown"
    asyncio.run(_portfolio_reset_baseline(account_id, reason, operator, confirm))


async def _portfolio_reset_baseline(account_id: str, reason: str, operator: str, confirm: bool):
    from cryptotrader.config import load_config
    from cryptotrader.portfolio.manager import PortfolioManager

    cfg = load_config()
    pm = PortfolioManager(cfg.infrastructure.database_url)
    portfolio = await pm.get_portfolio(account_id)
    current_equity = portfolio.get("total_value", 0.0)
    current_dd = await pm.get_drawdown(account_id)

    console.print(f"Account: [cyan]{account_id}[/cyan]")
    console.print(f"Current equity: [bold]${current_equity:,.2f}[/bold]")
    console.print(f"Current drawdown (from existing peak): [yellow]{current_dd * 100:.2f}%[/yellow]")
    console.print(f"After reset, drawdown baseline starts fresh from ${current_equity:,.2f}.")
    console.print(f"Operator: [cyan]{operator}[/cyan]")
    console.print(f"Reason: [cyan]{reason}[/cyan]")

    if not confirm:
        typer.confirm("Proceed with baseline reset?", abort=True)

    row = await pm.record_baseline_reset(
        account_id=account_id,
        baseline_equity=current_equity,
        operator=operator,
        reason=reason,
    )
    console.print(f"[green]Baseline reset recorded.[/green]  id={row['id']}  at={row['reset_at'].isoformat()}")
    console.print("[dim]Note: existing snapshots are preserved; only drawdown computation window changed.[/dim]")


# ── Agent subcommands ──

agent_app = typer.Typer(help="Agent management commands")
app.add_typer(agent_app, name="agent")


@agent_app.command("list")
def agent_list():
    """List registered agents and their configuration."""
    from cryptotrader.config import load_config

    cfg = load_config()
    active = cfg.agents.list_active()

    table = Table(title="Registered Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Skills")

    builtin_ids = {"tech_agent", "chain_agent", "news_agent", "macro_agent"}
    for ac in sorted(active, key=lambda a: a.agent_id):
        agent_type = "builtin" if ac.agent_id in builtin_ids else "custom"
        model_display = ac.model if ac.model else "<default>"
        status = "[green]enabled[/green]" if ac.enabled else "[red]disabled[/red]"
        skill_count = len(ac.skills) + sum(len(v) for v in ac.regime_skills.values())
        table.add_row(ac.agent_id, agent_type, model_display, status, str(skill_count))

    console.print(table)


# ── MCP subcommands ──

mcp_app = typer.Typer(help="MCP data layer management commands")
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("list")
def mcp_list():
    """List registered MCP tools and server health status."""
    from cryptotrader.config import load_config

    cfg = load_config()
    if not cfg.mcp.enabled:
        console.print("[yellow]MCP is disabled (mcp.enabled=false). Showing configured servers:[/yellow]")

    table = Table(title="MCP Servers & Tools")
    table.add_column("Server", style="cyan")
    table.add_column("Transport")
    table.add_column("Enabled")
    table.add_column("Tools")

    for sc in cfg.mcp.servers:
        enabled_display = "[green]yes[/green]" if sc.enabled else "[red]no[/red]"
        tools_display = ", ".join(sc.tools) if sc.tools else "<auto-discover>"
        table.add_row(sc.name, sc.transport, enabled_display, tools_display)

    console.print(table)
    tool_count = sum(len(sc.tools) for sc in cfg.mcp.servers if sc.enabled)
    console.print(f"\nTotal tools configured: {tool_count}")


@mcp_app.command("call")
def mcp_call(
    tool_name: str = typer.Argument(..., help="MCP tool name to call"),
    args: str = typer.Option("{}", "--args", help="JSON arguments for the tool"),
):
    """Call an MCP tool directly for debugging."""
    import json

    from cryptotrader.config import load_config
    from cryptotrader.mcp.registry import MCPRegistry, MCPToolNotFoundError

    cfg = load_config()
    if not cfg.mcp.enabled:
        console.print("[red]MCP is disabled. Set mcp.enabled=true in config.[/red]")
        raise typer.Exit(code=1)

    try:
        parsed_args = json.loads(args)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON args: {e}[/red]")
        raise typer.Exit(code=1) from e

    registry = MCPRegistry.from_config(cfg.mcp)

    async def _call():
        return await registry.call_tool(tool_name, parsed_args)

    try:
        result = asyncio.run(_call())
        console.print_json(json.dumps(result, default=str, indent=2))
    except MCPToolNotFoundError as exc:
        console.print(f"[red]Tool '{tool_name}' not found. Use 'arena mcp list' to see available tools.[/red]")
        raise typer.Exit(code=1) from exc
    except Exception as e:
        console.print(f"[red]Error calling tool: {e}[/red]")
        raise typer.Exit(code=1) from e


# ── Evolution Daemon (spec 022) ──


@app.command("evolution-daemon")
def evolution_daemon(
    once: bool = typer.Option(False, "--once", help="Run once (dry-run) and exit"),
    config_path: str = typer.Option("config/default.toml", "--config", help="Path to TOML config"),
) -> None:
    """Run the evolution reflect daemon (Pareto / regime / skill proposal).

    spec 022 FR-D2: arena evolution-daemon [--once] [--config PATH]
    """
    import os

    if os.getenv("EVOLUTION_DAEMON_ENABLED", "true").lower() not in ("true", "1", "yes"):
        console.print("[evolution-daemon] disabled by EVOLUTION_DAEMON_ENABLED env var; exiting.")
        raise typer.Exit(0)

    from cryptotrader.config import load_config
    from cryptotrader.ops.daemon import EvolutionDaemon

    cfg = load_config(config_path)

    # FR-D5: toml-level enabled check
    if not cfg.evolution_daemon.enabled:
        console.print("[evolution-daemon] disabled by config (evolution_daemon.enabled=false); exiting.")
        raise typer.Exit(0)

    daemon = EvolutionDaemon(config=cfg.evolution_daemon)

    if once:
        result = asyncio.run(daemon.run_once())
        console.print(f"[evolution-daemon] run_once exit_code={result.exit_code}")
        for action in result.actions_run:
            console.print(f"  [{action.name}] {action.status} {action.duration_ms}ms")
        raise typer.Exit(result.exit_code)
    asyncio.run(daemon.run_forever())


if __name__ == "__main__":
    app()
