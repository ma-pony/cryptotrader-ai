# CryptoTrader AI

CryptoTrader AI is a pluggable signal-fusion and multi-venue execution system. Kronos, the four-agent LLM committee, and custom components are independent signal sources. Configurable trust weights fuse them into one target position; each execution book then applies its own risk controls, approval, and execution.

## Runtime model

The database `runtime_config` is the only runtime configuration source. A process receives only `DATABASE_URL` and `CONFIG_MASTER_KEY`; the latter encrypts venue credentials. There is no TOML loading, dotenv merge, or single-venue mode switch.

Every web save atomically replaces a fully validated document and increments a global revision. Cycles and HITL proposals freeze that revision. The UI shows save errors without discarding the active state, and a revision change invalidates a pending approval.

```text
market data → Kronos / four-agent internal debate / custom components → fusion → target position
            → per-book risk, optional HITL, connection allocation → cycle audit
```

`target_position` is `side: long | short | flat` plus `size_ratio: 0..1`. It is a desired fraction of a book's risk capacity, never leverage or a raw order quantity. Connections trade only the delta to that target.

## First start

Requires Python 3.12+, Node.js 20+, uv, pnpm, and PostgreSQL. Create a 32-byte AES-GCM master key and start the API:

```bash
uv sync --all-extras
export DATABASE_URL='postgresql+asyncpg://<db-user>:<db-password>@localhost:5432/cryptotrader'
export CONFIG_MASTER_KEY='base64-encoded 32-byte key'
uv run trader serve --port 8003
```

Start the web application in another terminal:

```bash
cd web
pnpm install
pnpm dev
```

Open `http://localhost:5173`. An unconfigured installation opens the setup wizard. Configure LLMs, signal components, market data, venue connections, execution books, risk, scheduling, and notifications in order, then activate the runtime.

Paper, OKX, Bybit, and future adapters may run at the same time. Paper, demo, and testnet connections belong only to `simulated` books; live connections belong only to `real` books. One connection can belong to at most one enabled book. HITL is configurable per book and approves a full plan including protection prices and configuration revision.

## Containers

Compose passes the two runtime variables to the API owner:

```bash
export CONFIG_MASTER_KEY='base64-encoded 32-byte key'
docker compose up --build
```

Compose derives the PostgreSQL address and supplies it as `DATABASE_URL`. Complete initial setup in the web UI; containers do not read local configuration files.

## Useful commands

```bash
uv run trader run --pair BTC/USDT
uv run trader backtest --pair BTC/USDT --start 2025-01-01 --end 2025-03-01
uv run trader journal log
uv run trader journal show <cycle-id>
```

Backtests use an isolated temporary Paper book and never connect demo, testnet, or live venues. Validate live models with simulated connections after web setup; live-money connections are read-only checks and must not receive automated real-money orders.

## Acceptance canaries

After configuration in the web UI, a database-enabled Paper, Demo, or Testnet connection can run a minimal position loop. The script accepts and prints no venue credentials. It verifies that the pair begins with no position, orders, or protections; opens the minimum amount, installs native protection, closes with `reduce-only`, cleans up, and reconnects in a separate process to audit zero residual state. Any failure still cleans up and reports `requires_attention`.

```bash
uv run python scripts/venue_canary.py --connection bybit-testnet --pair BTC/USDT:USDT
uv run python scripts/signal_canary.py --pair BTC/USDT
```

`signal_canary.py` gathers live market evidence, runs Kronos plus the four-agent internal debate, fusion, and target generation, but never enters an execution book. A live connection can only be checked explicitly in read-only mode:

```bash
uv run python scripts/venue_canary.py --connection bybit-live --pair BTC/USDT:USDT --live-read-only
```

## Verification

```bash
uv run pytest --no-cov -q
uv run ruff check src tests scripts
uv run python scripts/import_smoke.py
cd web && pnpm test && pnpm typecheck && pnpm lint
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for system boundaries and data flow.
