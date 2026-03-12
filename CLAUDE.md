# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install          # uv pip install -e ".[dev]"
make test             # pytest tests/ -v
make lint             # ruff check src/ tests/
make run              # arena run --pair BTC/USDT --mode paper
make serve            # arena serve --port 8003 (FastAPI)
make scheduler        # arena scheduler start

# Run a single test
pytest tests/test_risk_gate.py -v
pytest tests/test_risk_gate.py::test_max_position -v

# CLI entry point is "arena" (src/cli/main.py)
arena run --pair ETH/USDT --mode paper
arena backtest --pair BTC/USDT --start 2024-01-01 --end 2024-06-01 --interval 4h
arena journal log --limit 10
arena scheduler start   # requires scheduler.enabled=true in config
arena serve --port 8003
arena dashboard
arena experience distill --session {id}   # Distill experience from backtest
arena experience show --session {id}      # Show distilled experience
arena experience merge --session {id}     # Merge backtest experience into live
arena experience sessions                 # List backtest sessions
```

## Architecture

Multi-agent crypto trading system built on LangGraph. The core pipeline:

```
Data Collection → Verbal Reinforcement → 4 Agents (fan-out, parallel)
  → Debate Gate (consensus/confusion check)
    → [debate needed] Cross-Challenge Debate (2 rounds, parallel per round)
    → [skip] straight to verdict
  → Verdict (AI or weighted-downgrade) → Risk Gate (11 checks) → Execute / Reject → Journal
```

**Three graph variants** in `src/cryptotrader/graph.py`:
- `build_trading_graph()` — Full pipeline with debate gate, optional debate rounds, AI verdict
- `build_lite_graph()` — Skips debate, used for backtesting
- `build_debate_graph()` — Bull/bear adversarial debate with judge (TradingAgents-style)

**Progressive Filtering** (reduces LLM calls from 13 to 4-5 when possible):
- **Debate parallelization**: `asyncio.gather()` runs 4 agents concurrently per debate round
- **Debate gate**: `debate_gate()` computes consensus strength; skips debate on strong consensus (`strength > 0.5`) or shared confusion (`|mean| < 0.05` + low dispersion)
- **Verdict downgrade**: When debate skipped + position flat + no circuit breaker → `make_verdict_weighted()` (0 LLM) replaces `make_verdict_llm()` (1 LLM)

**Key modules** under `src/cryptotrader/`:
- `agents/` — 4 specialized agents (Tech, Chain, News, Macro) extending `BaseAgent`, each calls LLM via LangChain `ChatOpenAI` with structured JSON output
- `debate/` — Cross-challenge rounds (parallel per round via `asyncio.gather`); bull/bear adversarial debate with judge (`researchers.py`). `convergence.py` has `compute_divergence()` and `compute_consensus_strength()` for debate gate
- `risk/gate.py` — 11 rule-based checks (no LLM): position size, exposure, daily loss, drawdown, CVaR, correlation, cooldown, volatility, funding rate, rate limit, exchange health
- `execution/` — `PaperExchange` (simulator) and `LiveExchange` (ccxt-based)
- `journal/store.py` — Git-like decision commit chain in PostgreSQL (in-memory fallback)
- `learning/verbal.py` — Verbal reinforcement: returns regime-aware historical cases for GSSC pipeline
- `learning/context.py` — GSSC engine: gather → select → structure experience into agent prompts
- `learning/regime.py` — Regime tagging (`tag_regime()`) and Jaccard overlap matching
- `learning/reflect.py` — Structured experience memory generation (ExperienceMemory JSON), incremental evolution, anti-overfitting verification
- `backtest/engine.py` — Steps through historical candles using lite graph or SMA crossover
- `data/` — Market (ccxt), on-chain (DefiLlama/CoinGlass/CryptoQuant/WhaleAlert), news (RSS), macro (FRED/CoinGecko/Fear&Greed)
- `models.py` — All Pydantic/dataclass models
- `config.py` — TOML config loading and validation

**Experience Memory (GSSC pipeline)**:
- Pipeline: `verbal.py` (regime search) + `reflect.py` (structured rules) → `context.py` (gather → select → structure) → agent prompts
- Regime tagging: `tag_regime()` classifies snapshot into labels (high_funding, high_vol, trending_up, extreme_fear, etc.)
- Anti-overfitting: 5-layer defense — min sample thresholds, maturity levels (observation → hypothesis → rule), regime-aware verification, LLM constraint prompts, code-verified win rates
- Backtest isolation: backtest mode skips experience injection + reflection to prevent contamination
- Session storage: `~/.cryptotrader/backtest_sessions/{id}/` with commits.jsonl, result.json, experience.json
- Config: `ExperienceConfig` (replaces `ReflectionConfig`, alias kept), nested `RegimeThresholdsConfig`

**Other entry points:**
- `src/api/` — FastAPI server with routes for analyze, health, journal
- `src/cli/main.py` — Typer CLI app (the `arena` command)
- `src/dashboard/app.py` — Streamlit dashboard

## Configuration

- `config/default.toml` — Main config: execution mode, LLM models per agent, debate params (incl. `skip_debate`, `consensus_skip_threshold`, `confusion_skip_threshold`, `confusion_max_dispersion`), data provider toggles, scheduler settings
- `config/default.toml` `[risk]` — Risk gate parameters (11 thresholds), `[scheduler]` includes `daily_summary_hour` and `exchange_id`
- `config/default.toml` `[experience]` — Experience memory: reflection cycle, token budget, win rate tolerance, `[experience.regime_thresholds]` for regime tagging
- `config/exchanges.toml.example` — Exchange API credentials template
- `.env` — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DATABASE_URL`, `REDIS_URL`

## Tech Stack

Python 3.12+, uv package manager, Hatchling build system. LLM calls go through LangChain ChatOpenAI with SQLiteCache. LangGraph for orchestration. PostgreSQL + Redis for persistence/caching. Docker Compose for infra (Postgres 16, Redis 7, scheduler service). APScheduler 3.x for periodic trading cycles.

## Data Layer

**Unified SQLite store** at `~/.cryptotrader/market_data.db` (`data/store.py`):
- 61+ data sources, 17,000+ records across 7 categories (macro, on-chain, derivatives, DeFi, sentiment, ETF, stablecoin)
- Rate limiting per source via `fetch_log` table — each source has its own key (e.g., `fred_DFF`, `fred_DTWEXBGS`, not shared `fred`)
- `data/sync.py` — Bulk historical sync for all sources (`arena sync`)
- `data/macro.py` — Live macro collection (FRED, Fear&Greed, CoinGecko, SoSoValue)
- `data/enhanced.py` — Enhanced data aggregation for agents
- Backtest engine loads extended data from store into DataSnapshot (ETF flows, OI, long/short, DeFi TVL, VIX, S&P500, stablecoin supply, hashrate)

## LLM Architecture

**Unified LangChain architecture** — all LLM calls go through `create_llm()` factory in `agents/base.py`:
- `create_llm(model, temperature, timeout, json_mode)` → `ChatOpenAI` with automatic fallback via `.with_fallbacks()`
- Config from `config/default.toml` `[llm]` section (base_url, api_key), fallback from `[models].fallback`
- **SQLiteCache** at `~/.cryptotrader/llm_cache.db` — exact-match caching on `(prompt, llm_string)`
- `acompletion_with_fallback()` kept as backward-compat wrapper (converts dict messages → LangChain messages)

**LLM call sites** (10 total across 7 files, reduced to 4-5 by progressive filtering):
- `agents/base.py` — BaseAgent.analyze() (create_llm, temp=0.2, json_mode), ToolAgent.analyze() (_create_chat_model)
- `debate/verdict.py` — AI verdict (create_llm, temp=0.1, json_mode); skipped when verdict downgraded to weighted
- `debate/researchers.py` — Bull/bear (create_llm, temp=0.3), judge (create_llm, temp=0.1, json_mode)
- `nodes/debate.py` — Cross-challenge debate (create_llm, temp=0.3, json_mode); parallel via `asyncio.gather`, skipped when debate gate triggers
- `agents/langchain_agents.py` — Supervisor agents (_create_chat_model)
- `learning/reflect.py` — Agent self-reflection (create_llm, temp=0.3, json_mode)
- `graph_supervisor.py` — Supervisor graph (uses langchain_agents)

**Key: `deepseek-reasoner`** returns content in `additional_kwargs['reasoning_content']`. `extract_content()` handles this.

**Test mocking**: Mock `langchain_openai.ChatOpenAI.ainvoke`, return `AIMessage(content=...)`.
**litellm removed** — no longer a dependency.

## Dependency Management

**Use `uv` for all package management** — never `pip install` directly:
```bash
uv add <package>              # Add to pyproject.toml dependencies
uv add --group dev <package>  # Add to dev dependency group
uv sync                       # Install all deps from lockfile
uv lock                       # Regenerate lockfile
```

## Testing

pytest with `asyncio_mode = "auto"`. Source path is `src/` (`pythonpath = ["src"]` in pyproject.toml). Tests are in `tests/`.

**IMPORTANT:** Must use `uv run pytest` (Python 3.12 venv), NOT bare `pytest` (may use system Python 3.10).

347 tests pass, 1 skip.

## Lessons Learned & Common Pitfalls

### Data Sync
- **Rate limit key sharing**: Each FRED series (DFF, DTWEXBGS, VIXCLS, SP500) must have its own rate limit key in `store.py`. Sharing a single `"fred"` key causes the second series to always skip.
- **fetch_log vs actual data**: `_record_fetch()` is called even if the actual HTTP request failed. When debugging missing data, check both `fetch_log` and `market_data` tables — clear `fetch_log` if data is missing.
- **API quirks**: Mempool.space uses `d["time"]` not `d["timestamp"]` for difficulty adjustments. CoinGecko global chart requires Pro API (401). DefiLlama rate-limits aggressively (429 on perps volume).
- **Date format consistency**: All store dates use ISO format `YYYY-MM-DD`. Sync functions must normalize dates before storing.

### LLM Integration
- **LangChain `ChatOpenAI`**: Use `max_completion_tokens` (not `max_tokens`). Pass `response_format` via `model_kwargs`.
- **LangChain agents**: Use `create_agent` from `langchain.agents` (not deprecated `create_react_agent`).
- **Fallback pattern**: LangChain's `llm.with_fallbacks([fallback_llm])` replaces custom `acompletion_with_fallback()`.
- **Cache**: LangChain's `SQLiteCache` provides exact-match caching on `(prompt, llm_string)` — any parameter change (model, temperature, data) = separate cache entry.
- **`langchain_agents.py` bypasses config**: Creates `ChatOpenAI` without reading `base_url`/`api_key` from config. Must fix when unifying.

### Architecture
- **graph.py split**: Was 874 lines, split into `state.py` + `nodes/` package (6 modules). Keep nodes modular.
- **JSON parsing**: LLM output often contains markdown fences or extra text. Use `_extract_json()` (balanced-brace extraction) instead of `json.loads()` directly.
- **Thread safety**: `PaperExchange` uses `threading.Lock` for portfolio state. Always lock when reading/writing positions.
- **Redis fallback**: If Redis was configured but is unavailable, system rejects trades conservatively. This is intentional.
- **Config caching**: `load_config()` is cached after first call. Don't expect config changes mid-run.

### Experience Memory
- **Token estimation**: CJK text averages ~1.5 chars/token (not 4). Use `_estimate_tokens()` in `context.py` for mixed content.
- **Regime-aware verification**: `_verify_rules()` must filter records by the rule's `conditions.regime_tags` before computing empirical rate. Global win rates are meaningless for regime-specific rules.
- **Merge stats**: When merging rules (backtest → live), always do weighted average on `rate`: `(rate_a * n_a + rate_b * n_b) / total`.
- **Case regime tags**: `_packets_from_cases()` must tag cases with `tag_regime()` from snapshot_summary, otherwise all cases score equally in regime-aware selection.
- **`conditions.regime_tags` safety**: LLM may return `"high_vol"` (string) instead of `["high_vol"]` (list). Always validate with `isinstance` check.

### Progressive Filtering
- **Debate gate confusion vs disagreement**: Low `|mean_score|` alone doesn't distinguish confusion from disagreement. Must also check `dispersion < confusion_max_dispersion` — high dispersion means agents strongly disagree (debate needed), low dispersion means shared uncertainty (debate useless).
- **Verdict downgrade safety**: Only downgrade to weighted when ALL conditions met: debate skipped + position flat + no circuit breaker. Redis unavailable → conservative, keep AI verdict.
- **`position_context` nullability**: `state["data"].get("position_context")` can be `None` (not just missing). Use `or {}` pattern, not `get(..., {})`.
- **Debate parallelization**: Each agent in a debate round sees a snapshot of others' analyses from round start (not mid-round updates), so `asyncio.gather` is safe. Use `return_exceptions=True` to preserve originals on failure.

### Code Quality Rules
- **禁止 `noqa` 注释** — 不允许用 noqa 跳过 lint 检查。遇到 C901 复杂度警告时，必须重构函数（提取子函数）而不是添加 noqa。遇到 F401 未使用导入时，要么删除导入，要么确保在 `__all__` 中声明为再导出。
- **零 lint 错误** — `uv run ruff check src/ tests/` 必须零错误通过。所有 pre-commit hooks 必须通过。
- **函数复杂度** — C901 阈值为 10。超过时拆分为辅助函数，不要加 noqa。

### Testing
- **Async tests**: Use `asyncio_mode = "auto"` — no need for `@pytest.mark.asyncio`.
- **Mock LLM calls**: Mock `langchain_openai.ChatOpenAI.ainvoke`，返回 `AIMessage(content=...)`.
- **Import paths**: Tests import from `cryptotrader.*` (not `src.cryptotrader.*`) due to `pythonpath = ["src"]`.
