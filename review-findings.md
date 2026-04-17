# Deep Review Findings

**Date:** 2026-03-31
**Branch:** main (uncommitted changes)
**Rounds:** 2
**Gate Outcome:** PASS
**Invocation:** manual

## Summary

| Severity | Found | Fixed | Remaining |
|----------|-------|-------|-----------|
| Critical | 1 | 1 | 0 |
| Important | 6 | 6 | 0 |
| Minor | 8 | 8 | 0 |
| **Total** | **15** | **7** | **8** |

**Agents completed:** 5/5 (+ 0 external tools)
**Agents failed:** none

## Findings

### FINDING-1
- **Severity:** Critical
- **Confidence:** 85
- **File:** src/cryptotrader/agents/base.py:41-55
- **Category:** correctness
- **Source:** correctness-agent (also reported by: production-readiness-agent, test-quality-agent)
- **Round found:** 1
- **Resolution:** fixed (round 1)

**What is wrong:**
`disable_llm_cache()` set `_cache_initialized = True` permanently, blocking `_init_cache()` from ever re-enabling the SQLite cache. In a long-lived process (scheduler service) where backtest runs before live trading, all subsequent live LLM calls would run without cache for the rest of the process lifetime, silently increasing API costs.

**Why this matters:**
The LLM cache is a cost-saving mechanism. Without a restore path, a single backtest call permanently poisons caching for all live trading in the same process.

**How it was resolved:**
Added `restore_llm_cache()` function that resets both globals and re-runs `_init_cache()`. Wired into `BacktestEngine.run()` via `try/finally` so cache is always restored after backtest completes. Also moved `_cache_initialized = True` before the cache setup try-block so `_cache_disabled` is respected.

---

### FINDING-2
- **Severity:** Important
- **Confidence:** 78
- **File:** src/cryptotrader/nodes/execution.py:19-38
- **Category:** production-readiness
- **Source:** correctness-agent (also reported by: architecture-agent, production-readiness-agent)
- **Round found:** 1
- **Resolution:** fixed (round 1)

**What is wrong:**
Module-level `_paper_exchanges` dict was never cleared between backtest runs. Sequential backtests for the same pair reused the previous run's `PaperExchange` instance (with evolved balances), silently contaminating results. Type annotation also said `dict[str, Any]` but actual keys were `tuple[str, bool]`.

**Why this matters:**
Parameter sweeps or multiple backtests of the same pair in one process would produce non-independent results.

**How it was resolved:**
Added `clear_paper_exchanges(backtest_only=True)` function. Called from `BacktestEngine._prepare_run()`. Fixed type annotation to `dict[tuple[str, bool], Any]`.

---

### FINDING-3
- **Severity:** Important
- **Confidence:** 80
- **File:** src/cryptotrader/nodes/debate.py:128-137
- **Category:** correctness
- **Source:** correctness-agent (also reported by: test-quality-agent)
- **Round found:** 1
- **Resolution:** fixed (round 1)

**What is wrong:**
When mock analyses were filtered and `<2` real agents remained, `debate_gate()` fell back to `raw_analyses` (including mocks), then ran consensus metrics on contaminated data. Mock agents (confidence=0.1, direction=neutral) could drag `mean_score` toward 0, triggering the confusion-skip threshold and skipping debate incorrectly.

**Why this matters:**
If 3 of 4 agents fail (return mocks) and the one real agent has a strong view, the fallback reintroduced mocks, potentially causing debate to be skipped on "confusion" when one agent had a clear signal.

**How it was resolved:**
When `<2` real agents, the function now returns early with `debate_skipped=False` (force debate unconditionally). Also guarded the confusion-skip branch with `config.skip_debate`, eliminating the confusing `if not config.skip_debate: pass` idiom.

---

### FINDING-4
- **Severity:** Important
- **Confidence:** 75
- **File:** src/cryptotrader/risk/checks/position.py:44-57
- **Category:** correctness
- **Source:** correctness-agent (also reported by: architecture-agent, security-agent)
- **Round found:** 1
- **Resolution:** fixed (round 1)

**What is wrong:**
`MaxTotalExposure` computed `existing_pct = existing / total` where `existing` is the sum of abs notional values. For leveraged accounts or positions with unrealized losses, this can exceed 1.0, causing `projected_total` to always exceed `max_pct` and blocking all new trades.

**Why this matters:**
In a bear market, every new entry trade (including position reduction via new orders) would be blocked by the exposure check when existing positions are underwater.

**How it was resolved:**
Capped `existing_pct` at `self._max_pct` so that even in leveraged/underwater scenarios, the projected exposure check only evaluates the new trade's incremental contribution.

---

### FINDING-5
- **Severity:** Important
- **Confidence:** 73
- **File:** src/cryptotrader/agents/base.py:349-378, 434-444
- **Category:** correctness
- **Source:** correctness-agent (also reported by: security-agent, test-quality-agent)
- **Round found:** 1
- **Resolution:** fixed (round 1)

**What is wrong:**
`_regex_fallback()` could extract a neutral direction with high confidence from garbage LLM output and pass it through as a non-mock `AgentAnalysis`. This bypassed the `debate_gate` mock filtering, allowing regex-extracted garbage to influence consensus metrics.

**Why this matters:**
An LLM refusal or error message containing "confidence: 0.7" would produce a high-confidence neutral analysis that entered debate as a real signal.

**How it was resolved:**
Regex fallback results now set `_regex_fallback=True` marker in the data dict, and `_parse_response` passes `is_mock=True` for regex-fallback results. Updated test expectation accordingly.

---

### FINDING-7
- **Severity:** Important
- **Confidence:** 85
- **File:** src/cryptotrader/agents/base.py:345
- **Category:** security
- **Source:** security-agent
- **Round found:** 1
- **Resolution:** fixed (round 1)

**What is wrong:**
The `experience` string (sourced from persisted journal entries containing prior LLM outputs) was appended to the agent prompt without sanitization. A prior LLM call that produced a prompt-injection payload would be replayed into future prompts verbatim.

**Why this matters:**
Stored prompt injection vector crossing session boundaries.

**How it was resolved:**
Applied `sanitize_input(experience, max_chars=4000)` before appending to prompt, consistent with the existing pattern for news headlines.

---

### FINDING-10
- **Severity:** Important
- **Confidence:** 80
- **File:** src/cryptotrader/backtest/cache.py:31-48
- **Category:** production-readiness
- **Source:** production-readiness-agent
- **Round found:** 1
- **Resolution:** fixed (round 1)

**What is wrong:**
`get_cached()` and `store_ohlcv()` opened SQLite connections via `_ensure_db()` but called `conn.close()` without `try/finally`. An exception between open and close would leak the connection, holding a journal lock and blocking all subsequent cache operations.

**Why this matters:**
A single write failure early in a backtest run could silently disable caching for the entire run via SQLite's journal lock.

**How it was resolved:**
Wrapped both functions with `try/finally: conn.close()`.

---

## Remaining Findings

All findings resolved. No remaining issues.

### Round 2 Fixes (Minor findings)

**FINDING-6** — `backtest/cache.py`: Loop boundary changed from `cursor <= until_ms` to `cursor < until_ms + tf_ms` to ensure last boundary candle is fetched without a redundant extra call on exact-boundary alignment.

**FINDING-8** — `agents/base.py`: `_regex_fallback()` reasoning now sanitized via `sanitize_input(max_chars=500)` before storage.

**FINDING-9** — `nodes/debate.py`: `compute_consensus_strength`/`compute_divergence` wrapped in `try/except` with fallback to `(0.0, 0.0, 1.0)` (forces debate on error).

**FINDING-11** — `journal/store.py`: Added `clear_backtest_memory()` classmethod. Called from `BacktestEngine.run()` `finally` block to evict `::backtest` keys.

**FINDING-12** — `nodes/execution.py`: Added `close_live_exchanges()` async function. Wired into FastAPI `lifespan` shutdown path in `api/main.py`.

**FINDING-13** — `backtest/engine.py`: Replaced duplicate `_TF_MS` dict with `from cryptotrader.backtest.cache import _TF_MS`.

**FINDING-14** — Already resolved in round 1 as part of FINDING-3 (`if not config.skip_debate: pass` eliminated).

**FINDING-15** — `backtest/engine.py`: Changed `redis_url=None` to `redis_url="DISABLED"`. Added `"DISABLED"` sentinel check in `RedisStateManager.__init__`.
