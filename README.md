# CryptoTrader AI

AI-powered crypto trading system using LangGraph multi-agent debate.

## Overview

4 specialized AI agents (Technical, On-chain, News, Macro) independently analyze market data, then debate through cross-challenge rounds to reach consensus. A hard-coded risk gate (11 rule-based checks, no LLM) enforces position limits, loss limits, and circuit breakers. Every decision is recorded in a Git-like Decision Journal for auditability and experience-based learning.

## Architecture

```
Data Collection → Verbal Reinforcement → 4 Agents (fan-out)
→ Cross-Challenge Debate (2-3 rounds) → Convergence Check
→ Verdict → Risk Gate (11 checks) → Execute / Reject → Journal
```

## Quickstart

```bash
# Install
uv pip install -e ".[dev]"

# Run one analysis cycle (paper trading)
arena run --pair BTC/USDT --mode paper

# View decision journal
arena journal log --limit 10
arena journal show <hash>

# Start API server
arena serve --port 8003
```

## Configuration

- `config/default.toml` — mode, execution engine, model settings, debate rounds
- `config/risk.toml` — 11 risk parameters (position limits, loss limits, cooldowns, etc.)
- `config/exchanges.toml` — exchange API keys (copy from `.example`)
- `.env` — LLM API keys, database URL, Redis URL

## Project Structure

```
src/cryptotrader/
├── models.py          # All data models
├── config.py          # TOML config loading + Pydantic validation
├── graph.py           # LangGraph orchestration
├── data/              # Market, on-chain, news, macro collectors
├── agents/            # 4 analysis agents + base class
├── debate/            # Cross-challenge, convergence, verdict
├── risk/              # Risk gate + 11 rule-based checks
├── execution/         # Order manager, exchange adapters, simulator
├── journal/           # Decision commit chain + search + calibration
└── learning/          # Verbal Reinforcement from historical decisions
src/cli/               # Typer CLI
src/api/               # FastAPI server
```

## API Endpoints

- `POST /analyze` — run full analysis cycle
- `GET /journal/log` — recent decisions
- `GET /journal/{hash}` — decision detail
- `GET /health` — health check
- `GET /metrics` — basic metrics

## License

MIT
