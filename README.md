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

# Multi-pair analysis
arena run --pair BTC/USDT --pair ETH/USDT --mode paper

# View decision journal
arena journal log --limit 10
arena journal show <hash>

# Start API server
arena serve --port 8003
```

## On-chain Data Providers

Real-time data from 4 providers with graceful degradation (works without API keys):

| Provider | Data | Cost | Key Required |
|----------|------|------|-------------|
| DefiLlama | DeFi TVL, 7d change | Free | No |
| CoinGlass | Open interest, liquidations | Free tier (1000/mo) | Yes |
| CryptoQuant | Exchange netflow | Free tier (daily) | Yes |
| Whale Alert | Large transfers | Free tier (10/min) | Yes |

Additional free sources: FRED (fed rate, DXY), CoinGecko (BTC dominance), Alternative.me (Fear & Greed), CoinDesk/CoinTelegraph RSS (news sentiment).

## Configuration

### Config files

- `config/default.toml` — mode, models, debate params, provider API keys
- `config/risk.toml` — 11 risk parameters (position limits, loss limits, cooldowns)
- `.env` — LLM API keys, DATABASE_URL, REDIS_URL

### API Keys (environment variables)

```bash
# On-chain providers (optional — degrades gracefully without them)
COINGLASS_API_KEY=your_key
CRYPTOQUANT_API_KEY=your_key
WHALE_ALERT_API_KEY=your_key
FRED_API_KEY=your_key

# Infrastructure
DATABASE_URL=postgresql+asyncpg://...  # optional, uses in-memory fallback
REDIS_URL=redis://localhost:6379       # optional, risk checks reject conservatively without it
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Run full analysis cycle for a pair |
| GET | `/journal/log?limit=10` | Recent decision commits |
| GET | `/journal/{hash}` | Single commit detail |
| GET | `/health` | System status (API, Redis, DB) |
| GET | `/metrics` | Stats: total decisions, win rate, avg divergence |

### POST /analyze

```json
// Request
{"pair": "BTC/USDT", "exchange": "binance", "mode": "paper"}

// Response
{
  "pair": "BTC/USDT",
  "direction": "long",
  "confidence": 0.72,
  "position_scale": 0.85,
  "divergence": 0.15,
  "reasoning": "Weighted score=1.400, divergence=0.150",
  "risk_flags": ["funding rate elevated"],
  "debate_rounds": 2
}
```

## Project Structure

```
src/cryptotrader/
├── models.py          # All data models
├── config.py          # TOML config loading + Pydantic validation
├── graph.py           # LangGraph orchestration
├── data/
│   ├── market.py      # ccxt market data
│   ├── onchain.py     # Aggregates 4 providers
│   ├── news.py        # RSS scraping + keyword sentiment
│   ├── macro.py       # FRED + CoinGecko + Fear&Greed
│   └── providers/     # DefiLlama, CoinGlass, CryptoQuant, WhaleAlert
├── agents/            # 4 analysis agents + base class
├── debate/            # Cross-challenge, convergence, verdict
├── risk/              # Risk gate + 11 rule-based checks
├── execution/         # Order manager, exchange adapters, simulator
├── journal/           # Decision commit chain + search + calibration
└── learning/          # Verbal Reinforcement from historical decisions
src/cli/               # Typer CLI (multi-pair support)
src/api/               # FastAPI server
```

## License

MIT
