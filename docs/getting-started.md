# Getting Started

## Prerequisites

- Python 3.11+
- Docker (for PostgreSQL + Redis)

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd cryptotrader-ai
pip install -e ".[dev]"

# Start infrastructure
docker compose up -d postgres redis

# Run one analysis cycle (paper trading)
arena run --pair BTC/USDT --mode paper

# View decisions
arena journal log
arena journal show <hash>

# Run backtest
arena backtest --pair BTC/USDT --start 2025-01-01 --end 2025-12-31 --interval 4h

# Start scheduler
arena scheduler start

# Launch dashboard
arena dashboard

# Start API server
arena serve --port 8003
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | None (in-memory) |
| `REDIS_URL` | Redis connection string | None (conservative reject) |
| `OPENAI_API_KEY` | LLM API key (via litellm) | Required |

## Configuration

Edit `config/default.toml` for trading parameters and `config/risk.toml` for risk limits. See [configuration.md](configuration.md) for details.
