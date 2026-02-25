# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install          # uv pip install -e ".[dev]"
make test             # pytest tests/ -v
make lint             # ruff check src/ tests/
make run              # arena run --pair BTC/USDT --mode paper
make serve            # arena serve --port 8003 (FastAPI)

# Run a single test
pytest tests/test_risk_gate.py -v
pytest tests/test_risk_gate.py::test_max_position -v

# CLI entry point is "arena" (src/cli/main.py)
arena run --pair ETH/USDT --mode paper
arena backtest --pair BTC/USDT --start 2024-01-01 --end 2024-06-01 --interval 4h
arena journal log --limit 10
arena scheduler start
arena serve --port 8003
arena dashboard
```

## Architecture

Multi-agent crypto trading system built on LangGraph. The core pipeline:

```
Data Collection → Verbal Reinforcement → 4 Agents (fan-out)
  → Cross-Challenge Debate (2-3 rounds) → Convergence Check
  → Verdict → Risk Gate (11 checks) → Execute / Reject → Journal
```

**Three graph variants** in `src/cryptotrader/graph.py`:
- `build_trading_graph()` — Full pipeline with debate loop and convergence check
- `build_lite_graph()` — Skips debate, used for backtesting
- `build_debate_graph()` — Bull/bear adversarial debate with judge (TradingAgents-style)

**Key modules** under `src/cryptotrader/`:
- `agents/` — 4 specialized agents (Tech, Chain, News, Macro) extending `BaseAgent`, each calls LLM via `litellm.acompletion()` with structured JSON output
- `debate/` — Cross-challenge rounds where agents revise after seeing others' analyses; also bull/bear adversarial debate with judge (`researchers.py`). Convergence checked via divergence scores
- `risk/gate.py` — 11 rule-based checks (no LLM): position size, exposure, daily loss, drawdown, CVaR, correlation, cooldown, volatility, funding rate, rate limit, exchange health
- `execution/` — `PaperExchange` (simulator) and `LiveExchange` (ccxt-based)
- `journal/store.py` — Git-like decision commit chain in PostgreSQL (in-memory fallback)
- `learning/verbal.py` — Verbal reinforcement: injects past decision experience into agent prompts
- `backtest/engine.py` — Steps through historical candles using lite graph or SMA crossover
- `data/` — Market (ccxt), on-chain (DefiLlama/CoinGlass/CryptoQuant/WhaleAlert), news (RSS), macro (FRED/CoinGecko/Fear&Greed)
- `models.py` — All Pydantic/dataclass models
- `config.py` — TOML config loading and validation

**Other entry points:**
- `src/api/` — FastAPI server with routes for analyze, health, journal
- `src/cli/main.py` — Typer CLI app (the `arena` command)
- `src/dashboard/app.py` — Streamlit dashboard

## Configuration

- `config/default.toml` — Main config: execution mode, LLM models per agent, debate params, data provider toggles, scheduler settings
- `config/risk.toml` — Risk gate parameters (11 thresholds)
- `config/exchanges.toml.example` — Exchange API credentials template
- `.env` — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DATABASE_URL`, `REDIS_URL`

## Tech Stack

Python 3.12+, uv package manager, Hatchling build system. LLM calls go through litellm (supports multiple providers). LangGraph for orchestration. PostgreSQL + Redis for persistence/caching. Docker Compose for infra (Postgres 16, Redis 7).

## Testing

pytest with `asyncio_mode = "auto"`. Source path is `src/` (`pythonpath = ["src"]` in pyproject.toml). Tests are in `tests/`.
