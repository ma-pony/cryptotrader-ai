# CryptoTrader AI

[简体中文](README.md) · **English**

AI-powered crypto trading system using LangGraph multi-agent debate.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-2458%20passed-brightgreen.svg)]()

## Overview

4 specialized AI agents (Technical, On-chain, News, Macro) independently analyze market data, then debate through cross-challenge rounds to reach consensus. A hard-coded risk gate (11 rule-based checks, no LLM) enforces position limits, loss limits, and circuit breakers. Every decision is recorded in a Git-like Decision Journal for auditability and experience-based learning.

Each agent runs a domain-specific **pre-signal checklist** (inspired by Devin's think-before-act pattern) to reduce overconfidence and hallucination before outputting signals.

### Key Features

- **Multi-agent debate** — 4 agents analyze independently, then cross-challenge each other over 2-3 rounds; debate gate skips debate on consensus or confusion for progressive filtering
- **Three graph modes** — Full debate pipeline with debate gate, lite (backtest), bull/bear adversarial with judge
- **11-check risk gate** — Pure rules, no LLM: position limits, CVaR, correlation, circuit breakers
- **Decision journal** — Git-like immutable commit chain with similarity search; outcomes feed the evolution daemon's pattern distillation
- **Experience feedback loop** — Reflection + Evolution Daemon distill successful patterns into per-agent SKILL.md (verbal-reinforcement historical-case injection removed 2026-05-13)
- **Structured experience memory** — GSSC pipeline (gather → select → structure) with regime-aware retrieval, ExperienceRule/ExperienceMemory JSON, and 5-layer anti-overfitting defense
- **Backtesting engine** — Historical simulation with realistic cost modeling and no look-ahead bias
- **Live trading ready** — ccxt-based exchange adapters with retry, precision, and timeout handling
- **APScheduler automation** — Periodic trading cycles with daily portfolio summaries
- **61+ data sources** — Unified SQLite store across 7 categories with rate limiting per source
- **Trilogy Evolution System** (spec 016→020c, 2026-05-09) — config-driven prompts (`config/agents/<name>.md`) + Memory Evolution (5-signal Maturity FSM + Pareto + IVE failure classification) + Skill Evolution (D-RT-01 retrieval + LLM-inferred metadata) + standalone `evolution-daemon` docker service (daily Pareto rerank + Regime recluster + Skill proposal auto-trigger, soft degrade) + Git Lineage (auto-commit to `evolution` branch with `Auto-Generated-By: spec-020c` trailer)
- **Anthropic prompt cache observability** — `apply_cache_control()` in production; OTel span attrs `llm.cache.{read,creation,hit_rate}`; Prometheus `llm_cache_hit_rate_24h_avg` gauge
- **11 Prometheus gauges** — cache hit rate / IVE failure / daemon run / lineage commit / etc.; dashboard-only (no alertmanager) to avoid alert fatigue

## Architecture

```
Data Collection → Regime Tagging → 4 Agents (fan-out, parallel)
  → Debate Gate → [skip] → Enrich Context → Verdict
                → [debate] → 2 Debate Rounds (parallel per round)
  → Verdict → Risk Gate (11 checks) → Execute / Reject → Journal
                                        ↓
                              Portfolio Write-back → Snapshot
```

**Three graph variants:**
- `build_trading_graph()` — Full pipeline with debate gate (skip on consensus/confusion), 2 debate rounds, AI verdict with downgrade
- `build_lite_graph()` — Skips debate, used for backtesting
- `build_debate_graph()` — Bull/bear adversarial debate with judge (TradingAgents-style)

### How Agents Work

| Agent | Type | Data | Role |
|-------|------|------|------|
| TechAgent | BaseAgent | OHLCV + pure-pandas indicators (RSI, MACD, SMA, BBands, ATR, OBV) | Technical pattern recognition |
| ChainAgent | ToolAgent | OI, funding rate, exchange netflow, whale transfers, DeFi TVL | On-chain signal detection |
| NewsAgent | ToolAgent | RSS headlines + keyword sentiment + CoinGecko social buzz | News & sentiment analysis |
| MacroAgent | BaseAgent | Fed rate, DXY, BTC dominance, Fear & Greed, ETF flows, VIX | Macro regime assessment |

- **BaseAgent**: Single LLM call with structured JSON output
- **ToolAgent**: LangChain agent with tool-calling loop for real-time data queries (falls back to single call in backtest mode to avoid forward-looking bias)

Every agent's system prompt includes a **5-point pre-signal checklist**: contradiction check, evidence grounding, confidence sanity, base rate awareness, and recency trap avoidance. Confidence is calibrated on a 0-1 scale with `data_sufficiency="low"` capping output at 0.3.

### How Debate Works

1. **Round 1**: All 4 agents analyze independently (parallel)
2. **Debate Gate**: Evaluates divergence; skips debate if agents already show consensus (divergence < `consensus_skip_threshold`) or confusion (divergence < `confusion_max_dispersion` with low confidence); otherwise proceeds to debate
3. **Round 2-3**: Each agent sees all others' analyses and must justify holding or revising their position with specific data points (parallel per round)
4. **Convergence check**: Divergence score (population stdev of `confidence × direction`) tracked per round; stops when relative change < 10% or max rounds reached
5. **Verdict**: Single LLM at temperature 0.1 sees all agent outputs, position context (FLAT/LONG/SHORT, entry price, unrealized PnL), price trend, and risk constraints → outputs `{action, confidence, position_scale, reasoning, thesis, invalidation}`

### Learning System

- **Regime tagging**: `tag_regime()` computes regime labels (high_funding, high_vol, trending_up, extreme_fear, etc.) from the data snapshot — used downstream by the evolution daemon's regime-cluster step and by DecisionCommit for retrieval. Verbal-reinforcement historical-case dump into agent prompts was removed 2026-05-13 (re-introduced the kind of prior anchoring round-3 minimal skills had just removed)
- **GSSC pipeline**: `context.py` implements gather → select → structure: collects regime-matched cases and structured rules, scores by relevance, fits within token budget, and injects as structured context into agent prompts
- **Structured experience memory**: `learning/memory.py` defines `ExperienceRule` / `ExperienceMemory` with maturity levels (observation → hypothesis → rule) and empirical win rates. The evolution daemon's `distill_patterns()` populates these from settled trades and writes them into the `AUTO-DISTILLED-PATTERNS` section of each agent's `SKILL.md`. The legacy per-cycle `reflect.py` injection path (LLM "strategy memo" into prompt) was removed 2026-05-13 — same anti-anchoring rationale as bias-correction and verbal-reinforcement
- **Anti-overfitting 5-layer defense**: minimum sample thresholds, maturity gating, regime-aware verification (win rates computed only within matching regime), LLM constraint prompts, code-verified win rates
- **Evolution daemon**: A standalone process runs daily Pareto re-ranking + regime re-clustering + skill-proposal triggers + pattern extraction. Distilled patterns are written into each `agent_skills/<id>/SKILL.md` under the `AUTO-DISTILLED-PATTERNS` section, replacing the legacy bias-correction injection path (removed 2026-05-13: was contradicting the round-3 minimal-skill anti-anchor philosophy)

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI-compatible LLM API key (OpenAI, Anthropic, DeepSeek, etc.)

### Installation

```bash
# Clone and install
git clone https://github.com/your-org/cryptotrader-ai.git
cd cryptotrader-ai
uv sync

# Configure LLM endpoint
cp config/default.toml config/local.toml
# Edit config/local.toml: set [llm] api_key and base_url

# Or use environment variables
export OPENAI_API_KEY=your_key
```

### First Run

```bash
# Run one analysis cycle (paper trading)
arena run --pair BTC/USDT --mode paper

# Multi-pair analysis with full debate
arena run --pair BTC/USDT --pair ETH/USDT --graph full

# View decision journal
arena journal log --limit 10
arena journal show <hash>
```

### Backtesting

```bash
# Basic backtest with AI agents
arena backtest --pair BTC/USDT --start 2024-01-01 --end 2024-06-01 --interval 4h

# Fast backtest with SMA crossover (no LLM calls)
arena backtest --pair BTC/USDT --start 2024-01-01 --end 2024-06-01 --no-llm

# Sync historical data first for richer backtests
arena sync
```

The backtest engine features:
- **No look-ahead bias**: Signal on bar[i], execution at bar[i+1] open
- **Realistic costs**: Configurable slippage (5 bps) + fees (10 bps)
- **Dynamic position sizing**: 35% at high confidence, 12% medium, 6% low
- **Rich data**: ETF flows, OI, long/short ratio, DeFi TVL, VIX, S&P500, stablecoin supply, hashrate
- **Metrics**: Total return, Sharpe ratio (365d annualized), max drawdown, win rate, equity curve

### Scheduler

```bash
# Start periodic trading cycles (requires scheduler.enabled=true in config)
arena scheduler start

# Check portfolio status
arena scheduler status
```

APScheduler-based with `IntervalTrigger` (default 4h) for trading cycles and `CronTrigger` for daily portfolio summaries.

### Dashboard & API

```bash
# Web frontend (React + Vite)
arena web --port 5173

# FastAPI server
arena serve --port 8003
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `arena run --pair BTC/USDT --mode paper` | Single analysis + execution cycle |
| `arena run --pair BTC/USDT --graph full\|lite\|debate` | Choose graph variant |
| `arena backtest --pair BTC/USDT --start DATE --end DATE` | Historical backtest |
| `arena sync` | Sync all historical data to SQLite store |
| `arena serve --port 8003` | Start FastAPI server |
| `arena web` | Launch React web frontend |
| `arena scheduler start` | Start periodic scheduler |
| `arena scheduler status` | Show portfolio & positions |
| `arena journal log --limit 10` | Recent decisions |
| `arena journal show <hash>` | Decision detail |
| `arena migrate` | Create PostgreSQL tables |
| `arena risk reset-breaker` | Reset circuit breaker |
| `arena live-check --exchange binance` | Pre-flight check for live trading |
| `arena experience distill --session {id}` | Distill experience from backtest |
| `arena experience show --session {id}` | Show distilled experience |
| `arena experience merge --session {id}` | Merge backtest experience into live |
| `arena experience sessions` | List backtest sessions |

## Data Sources

### Market & On-chain

Real-time data from 5 providers with graceful degradation (works without API keys):

| Provider | Data | Cost | Key Required |
|----------|------|------|-------------|
| Binance | Futures OI, funding rate, liquidations, long/short ratio | Free | No |
| DefiLlama | DeFi TVL, 7d change, stablecoin supply | Free | No |
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
| FRED | Fed funds rate, DXY, VIX, S&P 500 | Free (key required) |
| CoinGecko | BTC dominance | Free |
| Alternative.me | Fear & Greed index | Free |
| SoSoValue | BTC/ETH ETF daily flows, net assets | Free (key required) |

### Unified Data Store

All data is cached in `~/.cryptotrader/market_data.db` (SQLite, WAL mode):
- 61+ data sources across 7 categories (macro, on-chain, derivatives, DeFi, sentiment, ETF, stablecoin)
- Per-source rate limiting (5 min to 1 hour TTL depending on source)
- Forward-fill for trading-day data (FRED, ETF) to handle weekends/holidays
- `arena sync` bulk-fetches all historical data for backtesting

## Configuration

### Config Files

```
config/
├── default.toml          # Main config (mode, models, risk, scheduler, providers)
├── local.toml            # Local overrides (API keys, gitignored)
└── exchanges.toml.example  # Exchange credential template
```

Config loads `default.toml` first, then deep-merges `local.toml`. Cached after first load.

### Key Config Sections

```toml
[llm]
api_key = ""           # Unified LLM API key
base_url = ""          # API endpoint (e.g. "http://localhost:3000/v1")

[models]               # Per-role model selection (must be available in your LLM gateway)
analysis = "deepseek-v4-flash"
debate = "deepseek-v4-flash"
verdict = "gpt-5.5"
tech_agent = "deepseek-v4-flash"
chain_agent = "deepseek-v4-flash"
news_agent = "deepseek-v4-flash"
macro_agent = "deepseek-v4-flash"
fallback = "deepseek-v4-flash"
# Empty model name → resolves to `models.analysis` then `models.fallback`

[debate]
max_rounds = 3
convergence_threshold = 0.1
skip_debate = true
consensus_skip_threshold = 0.5
confusion_skip_threshold = 0.05
confusion_max_dispersion = 0.2

[risk]
max_stop_loss_pct = 0.05
[risk.position]
max_single_pct = 0.10
max_total_exposure_pct = 0.50
[risk.loss]
max_daily_loss_pct = 0.03
max_drawdown_pct = 0.10

[scheduler]
enabled = false
pairs = ["BTC/USDT", "ETH/USDT"]
interval_minutes = 240
exchange_id = "binance"
daily_summary_hour = 0    # UTC hour (0-23)

[experience]
enabled = true
every_n_cycles = 20
```

### Environment Variables

```bash
# ── API authentication (REQUIRED for production) ──
# Default: AUTH_MODE=enabled — API_KEY MUST be set or process fails to start.
# Use AUTH_MODE=disabled only for local dev (logs WARNING per request).
AUTH_MODE=enabled
API_KEY=$(openssl rand -hex 32)

# ── On-chain providers (optional but recommended) ──
# When unset, providers return None; the chain_agent prompt is told which
# data sources are unavailable so it lowers data_sufficiency accordingly.
COINGLASS_API_KEY=your_key
CRYPTOQUANT_API_KEY=your_key
WHALE_ALERT_API_KEY=your_key
FRED_API_KEY=your_key

# ── Infrastructure (read by config when CRYPTOTRADER_INFRASTRUCTURE__* prefix used) ──
CRYPTOTRADER_INFRASTRUCTURE__DATABASE_URL=postgresql+asyncpg://...
CRYPTOTRADER_INFRASTRUCTURE__REDIS_URL=redis://localhost:6379

# ── Frontend (production builds REJECT VITE_API_KEY at build time) ──
# Use only in dev .env.local for one-time hydration into useSettingsStore.
VITE_API_BASE_URL=http://localhost:8003
# VITE_API_KEY= (dev only; production users enter key in Settings UI)
```

## Risk Gate

11 rule-based checks (no LLM), all configurable in `config/default.toml` under `[risk]`:

| Check | What it does | Default |
|-------|-------------|---------|
| Max Position Size | Cap single position as % of portfolio | 10% |
| Total Exposure | Limit total open exposure | 50% |
| Daily Loss Limit | Circuit breaker on daily loss threshold | 3% |
| Max Drawdown | Reject trades during deep drawdowns | 10% |
| CVaR (99%) | Conditional Value-at-Risk from recent returns | 5% |
| Correlation | Block correlated positions (14 hardcoded groups) | — |
| Cooldown | Minimum time between trades on same pair | 60 min |
| Post-Loss Cooldown | Extra cooldown after a losing trade | 120 min |
| Volatility | Reject during extreme vol or flash crashes | — |
| Funding Rate | Block when funding rate signals crowded positioning | — |
| Rate Limit | Cap trades per hour/day | — |
| Exchange Health | Check API latency before execution | — |

The `close` action (closing existing positions) is **exempt from all risk checks** — reducing exposure should never be blocked.

## Notifications

Webhook notifications for 6 event types (configure in `config/default.toml`):

| Event | Trigger |
|-------|---------|
| `trade` | Successful order fill |
| `rejection` | Risk gate rejects a trade |
| `circuit_breaker` | Daily loss limit triggers circuit breaker |
| `daily_summary` | Scheduler emits daily portfolio summary |
| `reconcile_mismatch` | Position reconciliation detects mismatch |
| `portfolio_stale` | Portfolio data becomes stale or unavailable |

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | Public | Detailed component status (API/Redis/DB/LLM) — 503 if degraded |
| GET | `/metrics` | Public | Prometheus metrics (text format) |
| GET | `/scheduler/status` | Public | Scheduler heartbeat (LB probe) |
| GET | `/api/portfolio/snapshot` | API Key | Current equity, cash, positions |
| GET | `/api/portfolio/equity-curve?range=24h\|7d\|30d\|all` | API Key | Time-series equity points |
| GET | `/api/risk/status` | API Key | Trade counts, circuit breaker, thresholds (live Redis ping) |
| POST | `/api/risk/circuit-breaker/reset` | API Key | Manually clear breaker (409 if not active) |
| GET | `/api/decisions?pair=&from=&to=&page=&size=` | API Key | Paginated decision list |
| GET | `/api/decisions/{commit_hash}` | API Key | Full decision detail (agents/debate/verdict/risk_gate/timeline) |
| GET | `/api/scheduler/status` | API Key | Next pair / next run timestamp |
| GET / POST / PATCH / DELETE | `/api/scheduler/rules` | API Key | Trigger rule CRUD |
| GET | `/api/metrics/summary` | API Key | Counters + p50/p95 latencies |
| POST | `/api/backtest/run` | API Key | Submit async backtest (returns run_id) |
| GET | `/api/backtest/runs/{run_id}` | API Key | Status + progress + result |
| DELETE | `/api/backtest/runs/{run_id}` | API Key | Cancel running backtest |
| GET | `/api/backtest/sessions` | API Key | Saved session names |
| GET | `/api/backtest/sessions/{name}` | API Key | Load saved session |
| POST | `/api/chat/stream` | API Key | SSE stream of full pipeline events |
| POST | `/api/chat/interrupt/{session_id}` | API Key | Soft-interrupt running analysis |
| POST | `/api/chat/steer/{session_id}` | API Key | Inject mid-pipeline guidance |
| GET | `/api/hitl/pending` | API Key | Pending human-approval requests |
| POST | `/api/hitl/{approval_id}/respond` | API Key | Approve / reject |
| GET | `/api/market/{pair}` | API Key | Funding rate / open interest / liquidations |
| GET | `/api/market/{pair}/ohlcv?timeframe=&limit=` | API Key | OHLCV bars |

**Authentication**: Default `AUTH_MODE=enabled` requires `API_KEY` env var; missing key fails startup. `AUTH_MODE=disabled` bypasses auth and logs a WARNING per request (dev only). All comparisons use `secrets.compare_digest` (timing-safe). **Rate limit**: 60 req/min/IP, Redis-backed when configured for multi-process safety; in-memory fallback for single-process dev. **CORS**: Explicit `allow_methods` / `allow_headers` allowlist (no wildcards) since `allow_credentials=true`.

## Execution Layer

### Paper Trading

- Default mode, no real money at risk
- Configurable initial balance (default $10,000)
- Slippage model: `base + amount × price × 1e-8`
- Thread-safe via `asyncio.Lock`

### Live Trading

Production-hardened `LiveExchange` wrapping ccxt:

- **Retry**: Exponential backoff (3 attempts), fatal errors excluded (auth, permissions, insufficient funds)
- **Balance pre-check**: Verifies sufficient funds before every order
- **Precision**: Applies exchange-specific `amount_to_precision()` / `price_to_precision()`
- **Minimum order**: Checks exchange market limits
- **Timeout**: Polls every 2s for up to 30s, auto-cancels stale orders
- **Pre-flight**: `arena live-check` validates credentials, API latency, Redis, and database

```bash
# Verify live trading readiness
arena live-check --exchange binance
```

## Docker Deployment

```bash
# Start full stack (PostgreSQL 16 + Redis 7 + App + Dashboard + Scheduler)
docker compose up -d

# Services:
#   app        — FastAPI on :8003
#   web        — React frontend on :5173
#   scheduler  — Periodic trading cycles
#   postgres   — Decision journal + portfolio persistence
#   redis      — Risk state + cooldowns + circuit breaker
```

The Dockerfile uses multi-stage build with non-root user. Healthcheck polls `/health` every 30s.

## Project Structure

```
src/cryptotrader/
├── models.py          # All data models (DataSnapshot, AgentAnalysis, TradeVerdict, Order, etc.)
├── config.py          # TOML config loading + dataclass validation
├── graph.py           # LangGraph orchestration (3 graph variants)
├── state.py           # ArenaState TypedDict + build_initial_state() factory
├── scheduler.py       # APScheduler-based periodic trading cycles + daily summary
├── notifications.py   # Webhook notifications (6 event types)
├── db.py              # Shared async DB session factory
├── data/
│   ├── store.py       # Unified SQLite store (61+ sources, rate limiting)
│   ├── snapshot.py    # SnapshotAggregator (data collection entry point)
│   ├── market.py      # ccxt OHLCV + ticker + funding rate + volatility
│   ├── onchain.py     # Aggregates 5 providers (parallel fetch)
│   ├── news.py        # RSS + keyword sentiment + CoinGecko social buzz
│   ├── macro.py       # FRED + CoinGecko + Fear&Greed + SoSoValue ETF
│   ├── sync.py        # Bulk historical sync (arena sync)
│   └── providers/     # Binance, DefiLlama, CoinGlass, CryptoQuant, WhaleAlert, SoSoValue
├── agents/
│   ├── base.py        # BaseAgent + ToolAgent + create_llm() factory
│   ├── tech.py        # TechAgent (uses agents/_indicators.py — pure pandas/numpy, no native deps)
│   ├── chain.py       # ChainAgent (ToolAgent with on-chain tools)
│   ├── news.py        # NewsAgent (ToolAgent with news tools)
│   ├── macro.py       # MacroAgent (macro regime analysis)
│   └── data_tools.py  # LangChain @tool definitions (6 chain + 3 news)
├── debate/
│   ├── challenge.py   # Cross-challenge prompt construction
│   ├── convergence.py # Divergence score + convergence detection
│   ├── verdict.py     # AI verdict (LLM) + rules verdict (backtest)
│   └── researchers.py # Bull/bear adversarial debate with judge
├── nodes/             # LangGraph node functions
│   ├── agents.py      # 4-agent fan-out
│   ├── data.py        # Data collection + PnL update + trend context
│   ├── debate.py      # Debate rounds + convergence routing
│   ├── verdict.py     # Verdict + risk check
│   ├── execution.py   # Order placement + stop loss + position update
│   └── journal.py     # Decision logging
├── risk/
│   ├── gate.py        # RiskGate (11 sequential checks)
│   └── state.py       # RedisStateManager (with in-memory fallback)
├── execution/
│   ├── simulator.py   # PaperExchange (paper trading)
│   ├── exchange.py    # LiveExchange (ccxt, production-hardened)
│   ├── order.py       # OrderManager (state machine)
│   └── reconcile.py   # Position reconciliation
├── portfolio/
│   └── manager.py     # PortfolioManager (DB + in-memory)
├── journal/
│   ├── store.py       # JournalStore (PostgreSQL + in-memory fallback)
│   └── commit.py      # DecisionCommit + immutable hash-chained schema
├── learning/
│   ├── regime.py     # tag_regime (market regime labels)
│   ├── memory.py     # ExperienceRule / ExperienceMemory + maturity FSM + Pareto rank
│   ├── context.py     # GSSC engine (gather → select → structure → inject)
│   └── regime.py      # Regime tagging (tag_regime) + Jaccard overlap matching
└── backtest/
    ├── engine.py      # BacktestEngine (LLM + SMA modes)
    ├── session.py     # Backtest session storage (commits, results, experience)
    ├── cache.py       # OHLCV SQLite cache
    ├── historical_data.py  # FnG, funding rate, FRED, futures volume
    └── result.py      # BacktestResult metrics
src/cli/main.py        # Typer CLI (arena command)
src/api/               # FastAPI server (auth, rate limiting, middleware)
web/                   # React 19 + Vite 7 frontend (dashboard, decisions, backtest, risk, metrics)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Package Manager | uv + Hatchling |
| LLM Orchestration | LangChain 1.2+ / LangGraph 1.0+ |
| LLM Providers | ChatOpenAI (OpenAI, DeepSeek, Anthropic compatible) |
| Exchange | ccxt (Binance, OKX, etc.) |
| Data Processing | pandas + numpy (pure-Python indicators in `agents/_indicators.py`) |
| Scheduling | APScheduler 3.x |
| Database | PostgreSQL 16 + SQLAlchemy 2.0 async |
| Cache / State | Redis 7 |
| Local Storage | SQLite (data store + LLM cache + experience memory) |
| API Server | FastAPI + Uvicorn |
| Dashboard | React 19 + Vite 7 + TypeScript |
| CLI | Typer + Rich |

## Development

```bash
make install          # uv pip install -e ".[dev]"
make test             # pytest tests/ -v (347 tests)
make lint             # ruff check src/ tests/
make format           # ruff format src/ tests/
make scheduler        # arena scheduler start
make pre-commit-run   # Run all pre-commit hooks

# Run a single test
uv run pytest tests/test_risk_gate.py -v
uv run pytest tests/test_risk_gate.py::test_max_position -v

# Docker infrastructure
docker compose up -d   # PostgreSQL 16 + Redis 7
arena migrate          # Create database tables
arena sync             # Sync historical data
```

### Code Quality

- **Zero lint errors**: `ruff check src/ tests/` must pass with zero errors
- **No `noqa` comments**: Refactor instead of suppressing (C901 threshold = 10)
- **347 tests**, 1 skip, 70% coverage
- **Async tests**: `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` needed
- **Must use `uv run pytest`** (Python 3.12 venv), not bare `pytest`

## License

MIT
