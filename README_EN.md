# CryptoTrader AI

CryptoTrader AI is a pluggable signal-fusion and trading system for crypto markets. Kronos, the four-agent LLM committee, and user-defined strategies are peer signal components. Their outputs are fused deterministically by configurable trust weights before the system creates a target position, exit prices, a risk decision, and an execution plan.

## Domain model

Every component returns `ComponentSignal(component_id, direction, confidence, reasoning, details)`, where direction is `long`, `short`, or `neutral` and confidence is between zero and one.

The fusion score may be positive or negative internally. The business-facing result is deliberately explicit:

```text
TargetPosition(side: long | short | flat, size_ratio: 0..1)
```

`size_ratio` is the desired fraction of the configured per-symbol risk capacity. It is neither leverage nor a raw order quantity. Execution trades only the delta between the current and target positions.

## Components and profile

Built-ins:

- `kronos`: Kronos time-series inference and gate, exposed as a pure signal component.
- `llm_committee`: technical, on-chain, news, and macro agents with internal cross-examination, convergence checks, and a final committee summary.

Custom component factories are configured under `[signal_plugins].factories` using `package.module:function` references.

The global Signal Profile controls component enablement and trust weights, the neutral threshold, maximum target ratio, ATR exit settings, and HITL. Enabled weights must total exactly `1.0`. The `/strategy` page edits the profile dynamically. A save creates a new revision; in-flight cycles keep their frozen revision and changes apply on the next cycle.

## One trading pipeline

Live, paper, and backtest modes share `TradingCycle`:

```text
freeze profile → collect point-in-time context → run all enabled components
→ strict success check → weighted fusion → target position → ATR exit policy
→ optional HITL → risk gate → delta execution → trading cycle journal
```

If any enabled component fails, the cycle ends as `component_failed` and cannot trade. Approved HITL plans are recalculated against the latest position without regenerating signals.

## Run locally

Requires Python 3.12+, Node.js 20+, uv, and pnpm.

```bash
uv sync --all-extras
cp config/default.toml config/local.toml
uv run arena serve --port 8003
cd web && pnpm install && pnpm dev
```

Useful commands:

```bash
uv run arena run --pair BTC/USDT --mode paper
uv run arena journal log
uv run arena journal show <cycle-id>
uv run arena scheduler start
uv run arena migrate
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for module boundaries and invariants.
