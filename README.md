# CryptoTrader AI

AI-powered crypto trading system using LangGraph multi-agent debate.

## Overview

4 specialized AI agents (Technical, On-chain, News, Macro) independently analyze market data, then debate through cross-challenge rounds to reach consensus. A hard-coded risk gate (12 rule-based checks, no LLM) enforces position limits, loss limits, and circuit breakers. Every decision is recorded in a Git-like Decision Journal for auditability and experience-based learning.

Each agent runs a domain-specific **pre-signal checklist** (inspired by Devin's think-before-act pattern) to reduce overconfidence and hallucination before outputting signals.

**New**: Integrated with OKX and Binance skills for enhanced data (token security audit, real-time prices, sentiment analysis). See [SKILLS_INTEGRATION.md](SKILLS_INTEGRATION.md).

## Architecture

```
Data Collection → Verbal Reinforcement → 4 Agents (fan-out)
→ Cross-Challenge Debate (2-3 rounds) → Convergence Check
→ Verdict → Risk Gate (12 checks) → Execute / Reject → Journal
                                        ↓
                              Portfolio Write-back → Snapshot
```

**Three graph variants:**
- `build_trading_graph()` — Full pipeline with debate loop and convergence check
- `build_lite_graph()` — Skips debate, used for backtesting
- `build_debate_graph()` — Bull/bear adversarial debate with judge (TradingAgents-style)

## Quickstart

```bash
# Install
uv pip install -e ".[dev]"

# Run one analysis cycle (paper trading)
arena run --pair BTC/USDT --mode paper

# Multi-pair analysis
arena run --pair BTC/USDT --pair ETH/USDT --mode paper

# Backtest
arena backtest --pair BTC/USDT --start 2024-01-01 --end 2024-06-01 --interval 4h

# View decision journal
arena journal log --limit 10
arena journal show <hash>

# Scheduler
arena scheduler start       # periodic trading cycles
arena scheduler status      # portfolio & position status

# Dashboard & API
arena dashboard             # Streamlit UI
arena serve --port 8003     # FastAPI server
```

## Data Sources

### Market & On-chain

Real-time data from 5 providers with graceful degradation (works without API keys):

| Provider | Data | Cost | Key Required |
|----------|------|------|-------------|
| Binance | Futures OI, funding rate, liquidations | Free | No |
| DefiLlama | DeFi TVL, 7d change | Free | No |
| CoinGlass | Open interest, liquidations | Free tier (1000/mo) | Yes |
| CryptoQuant | Exchange netflow | Free tier (daily) | Yes |
| Whale Alert | Large transfers | Free tier (10/min) | Yes |

### News & Sentiment

| Source | Data | Cost |
|--------|------|------|
| CoinDesk, CoinTelegraph, Decrypt | Headlines via RSS | Free |
| CoinGecko Community API | Social buzz (Twitter followers, Reddit subs, sentiment votes) | Free |

### Macro

| Source | Data | Cost |
|--------|------|------|
| FRED | Fed funds rate, DXY | Free (key required) |
| CoinGecko | BTC dominance | Free |
| Alternative.me | Fear & Greed index | Free |

## Configuration

### Config files

- `config/default.toml` — mode, models, debate params, provider API keys, scheduler, notifications
- `config/risk.toml` — 11 risk parameters (position limits, loss limits, cooldowns)
- `.env` — LLM API keys, DATABASE_URL, REDIS_URL

### API Keys (environment variables)

```bash
# LLM providers
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# On-chain providers (optional — degrades gracefully without them)
COINGLASS_API_KEY=your_key
CRYPTOQUANT_API_KEY=your_key
WHALE_ALERT_API_KEY=your_key
FRED_API_KEY=your_key

# Infrastructure (optional — uses in-memory fallback)
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://localhost:6379
```

## Risk Gate

11 rule-based checks (no LLM), all configurable in `config/risk.toml`:

| Check | What it does |
|-------|-------------|
| Max Position Size | Cap single position as % of portfolio |
| Total Exposure | Limit total open exposure |
| Daily Loss Limit | Circuit breaker on daily loss threshold |
| Max Drawdown | Reject trades during deep drawdowns |
| CVaR (99%) | Conditional Value-at-Risk from recent returns |
| Correlation | Block correlated positions |
| Cooldown | Enforce minimum time between trades on same pair |
| Post-Loss Cooldown | Extra cooldown after a losing trade |
| Volatility | Reject during extreme vol or flash crashes |
| Funding Rate | Block when funding rate signals crowded positioning |
| Rate Limit | Cap trades per hour/day |
| Exchange Health | Check API latency before execution |

## Notifications

Webhook notifications for 5 event types (configure in `config/default.toml`):

| Event | Trigger |
|-------|---------|
| `trade` | Successful order fill |
| `rejection` | Risk gate rejects a trade |
| `circuit_breaker` | Daily loss limit triggers circuit breaker |
| `daily_summary` | Scheduler emits daily portfolio summary |
| `reconcile_mismatch` | Position reconciliation detects mismatch |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Run full analysis cycle for a pair |
| GET | `/journal/log?limit=10` | Recent decision commits |
| GET | `/journal/{hash}` | Single commit detail |
| GET | `/health` | System status (API, Redis, DB) |
| GET | `/metrics` | Stats: total decisions, win rate, avg divergence |

## Project Structure

```
src/cryptotrader/
├── models.py          # All data models
├── config.py          # TOML config loading + Pydantic validation
├── graph.py           # LangGraph orchestration (3 graph variants)
├── scheduler.py       # Periodic trading cycles + daily summary
├── notifications.py   # Webhook notifications (5 event types)
├── data/
│   ├── market.py      # ccxt market data
│   ├── onchain.py     # Aggregates 5 providers
│   ├── news.py        # RSS + keyword sentiment + CoinGecko social buzz
│   ├── macro.py       # FRED + CoinGecko + Fear&Greed (parallel fetch)
│   └── providers/     # Binance, DefiLlama, CoinGlass, CryptoQuant, WhaleAlert
├── agents/            # 4 agents with pre-signal think checklists
├── debate/            # Cross-challenge, bull/bear adversarial, convergence, verdict
├── risk/              # Risk gate + 11 rule-based checks + Redis state
├── execution/         # Order manager, exchange adapters (live + paper), reconciler
├── portfolio/         # Position tracking + equity snapshots (DB + in-memory)
├── journal/           # Decision commit chain + similarity search + calibration
└── learning/          # Verbal Reinforcement from historical decisions
src/cli/               # Typer CLI (arena command)
src/api/               # FastAPI server
src/dashboard/         # Streamlit dashboard
```

## Development

```bash
make install          # uv pip install -e ".[dev]"
make test             # pytest tests/ -v (165 tests)
make lint             # ruff check src/ tests/
```

## License

MIT
