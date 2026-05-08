# Spec 015 — Runtime Safety Guardrails

**Status**: Active
**Created**: 2026-05-08
**Driver**: Three rounds of trader-grade audits + ~13 hours of live monitoring (cycles 13:30 CST 2026-05-07 → 12:47 CST 2026-05-08) revealed twelve+ runtime-level failure modes that no existing spec addresses. This spec collects them into a single coherent safety-guardrail layer applied at:
1. **Data layer** — input validation (DXY, ATR)
2. **Verdict layer** — server-side guardrails + LLM contract enforcement
3. **Risk gate layer** — pre-trade concentration / cooldown / portfolio sanity
4. **Execution layer** — exchange-side oversized-order learning

## Problem Statement

The previous architecture passed a "happy-path" verdict from LLM straight to execution. Three failure classes emerged in production:

1. **Garbage-in, gospel-out**: Macro data labelled "DXY" was actually FRED DTWEXBGS (broad index, base 2006=100, ~118). LLM read 118 as ICE-DXY-extreme, anchoring all 4 agents to permanent bearish stance for 24+ hours.
2. **LLM-prompt-only contracts are unenforced**: prompt said "MUST cite applied: skill / MUST give concrete invalidation / MUST size by confidence" — LLM honored ~60% of the time. Server had no enforcement, so non-compliant verdicts shipped at full size.
3. **Silent dropouts on edge paths**: pair-level cycles silently failed when a guardrail field hit a dataclass field-shape mismatch (`TradeVerdict(**vd)` raised `TypeError` on extra `guardrails` key); when same-pair cooldown had a cache-key shape mismatch (`(redis_url, leverage)` tuple vs `redis_url` str); when portfolio_unknown rejection didn't journal_rejection. None produced commits, none surfaced in logs above WARN level, none triggered alerts.

Result: 5/20 historical decisions were silently lost. SOL pnl was retroactively rewritten cycle-after-cycle from +$90.56 to +$54.17. macro_concentration false-rejected 5/8 add-to-existing trades. The system gave the appearance of correctness while degrading.

## Goals

**G1**: Every LLM-emitted verdict MUST satisfy mechanical safety constraints (sizing, attribution, stop distance, R:R) BEFORE reaching the risk gate; non-compliant verdicts auto-degrade in confidence rather than crash or ship.

**G2**: Pre-trade risk checks MUST count the right thing (e.g., adding to existing same-direction position does not consume a concentration slot).

**G3**: Realized PnL once captured at close time is **immutable** — never recomputed by retroactive backfill code paths.

**G4**: Failed cycle paths MUST produce a journal entry of some kind (rejection commit, error commit) — silent drops are never acceptable.

**G5**: External data sources (FRED, OKX) MUST be sanity-checked against known plausible bands; out-of-band values are treated as missing rather than poisoning downstream prompts.

## User Stories

**US-1 (Operator)**: As the operator, when an LLM verdict skips the `applied:` skill citation, I want the system to halve the trade size automatically rather than ship a bigger uninstrumented position whose P&L can't feed back into the skill PnL tracker.

**US-2 (Operator)**: As the operator, when 3 same-direction shorts are already open, the bot may add to any one of those existing positions but MUST NOT open a 4th distinct short pair — independent of correlation-group rules.

**US-3 (Operator)**: As the operator, when I close a position at +$90.56 realized P&L, the recorded value MUST stay +$90.56 forever — not get rewritten to +$54.17 the next cycle by a snapshot-based backfill.

**US-4 (Trader)**: As the trader, when a verdict includes a stop $100 from entry on a pair with $200 ATR(14h), the system MUST recognize that stop as inside the noise band and downgrade confidence rather than ship a trade that's mathematically expected to be whipsaw-stopped.

**US-5 (Trader)**: As the trader, when the verdict has stop-distance $1k risk and target-distance $1k reward (R:R 1:1), the system MUST refuse to trade at full confidence — required minimum R:R = 1.5.

**US-6 (Operator)**: As the operator, when OKX returns 51202 ("market order amount exceeds maximum") on DOGE, the next attempt within 30 minutes MUST automatically use a limit order at ±0.3% slippage rather than retrying market orders that will keep failing.

## Functional Requirements

### Data Layer

- **FR-G1**: `data/macro.py` MUST sanity-check every FRED series fetch against an explicit plausible range (`_FRED_PLAUSIBLE_RANGES`). Out-of-range values MUST be returned as `0.0` (treated as missing) and MUST NOT be cached. Both fresh API responses and stale cached values MUST go through this check.

- **FR-G2**: `MacroData.dxy` field is sourced from FRED `DTWEXBGS` (Trade-Weighted Broad USD Index, 2006=100, range 95-130). Agent prompts MUST label this value precisely so the LLM does NOT interpret it as ICE DXY (range 95-110). Specifically the macro_agent prompt MUST include the raw series ID and explicit "NOT the ICE DXY ticker" disclaimer.

- **FR-G3**: `nodes/data.py:_build_trend_context` MUST compute and emit `atr_14` (Average True Range over the last 14 OHLCV bars on the snapshot timeframe) when ≥ 15 bars are available. The verdict prompt's price-context block MUST surface ATR as both absolute price and percent-of-current.

### Verdict Layer

- **FR-G4**: `_post_process_verdict` MUST run as a server-side gauntlet of guardrails after LLM output and before journal write. Each guardrail can ONLY lower confidence or reduce position_scale, never raise. The guardrails are applied in order:
  1. **Confidence-based sizing cap**: `position_scale ≤ max(0, (cf − 0.5) × 2)` for long/short.
  2. **FR-026 enforcement**: directional verdict (long/short/close) without `applied:` citation in reasoning → cf × 0.5.
  3. **Silent-agent cap**: any agent with `is_mock=True` or `confidence==0` → cf at most `cf − 0.20`.
  4. **Stop distance (N2)**: stop distance < `max(1.5×ATR, 1% of entry)` → cf × 0.5. Skipped when entry_price unavailable.
  5. **R:R (N7)**: R:R = `|target−entry| / |entry−stop|` < 1.5 → cf × 0.5. Missing target_price on long/short → same penalty. Records computed R:R in `verdict.risk_reward_ratio` when valid.

- **FR-G5**: `TradeVerdict` dataclass MUST include `target_price: str` field (free-text price level the LLM emits, parsed to numeric by the server). Empty string is allowed for hold/close.

- **FR-G6**: `state.data.verdict` may contain non-TradeVerdict-fielded keys (`guardrails` audit list, `risk_reward_ratio`). Code consuming this dict via `TradeVerdict(**vd)` MUST filter to dataclass fields first (already required by `journal/store.py`; `nodes/verdict.py:risk_check` MUST follow the same pattern).

- **FR-G7**: verdict prompt's INVALIDATION + TARGET section MUST include both the formal stop-distance rule and a worked numeric example so the LLM knows what acceptable inputs look like before emission.

### Debate Layer

- **FR-G8**: `nodes/debate.py:_debate_one_agent` MUST enforce confidence anti-ratchet: when an agent's `direction` is unchanged from the prior round and `after_confidence > before_confidence + 0.02`, the `new_findings` field MUST start with the literal token `[NEW]` (case-insensitive). Violation: `after_confidence` is snapped to `before_confidence + 0.02`. Direction flips bypass this rule (a real change-of-mind doesn't require the [NEW] discipline; it targets confidence drift on an unchanged stance).

### Risk Gate Layer

- **FR-G9**: `MacroConcentrationCheck` MUST count distinct pairs that will be in the target direction *after* the proposed trade. Add-to-existing same-direction trades do NOT consume a slot; only opening a new pair (from flat) or flipping (from opposite direction) does. The verdict's own pair MUST be excluded from "existing same-direction count" and added back exactly once via `target_pairs.add(my_pair)`.

- **FR-G10**: `PositionConfig` MUST expose `max_same_direction_positions: int` (default 3). Set to a value ≥ pairs_count to disable.

- **FR-G11**: Same-pair cooldown (`set_cooldown(pair, minutes)` after a successful trade) MUST be persisted regardless of risk-gate cache shape. `nodes/execution.py:_update_trade_tracking` MUST instantiate a fresh `RedisStateManager(redis_url)` directly rather than reach into `_risk_gate_cache` (whose key shape was changed to `(redis_url, leverage)` tuple in 2026-05-07 leverage refactor; the bare-string lookup silently failed for ~24 hours of trading).

- **FR-G12**: `_update_one_commit_pnl` MUST treat any commit whose `pnl is not None` as final. The previous compound guard `pnl is not None AND fill_price is not None` was always false on close commits (top-level `fill_price` column is NULL on closes — fill price lives on the order row), causing realized P&L to be retroactively recomputed every cycle using the *current* price. The guard MUST simplify to `if pnl is not None: return False`.

### Execution Layer

- **FR-G13**: `execution/exchange.py:LiveExchange.place_order` MUST precheck the venue's market-order size cap (`market.limits.amount.max` or OKX `info.maxMktSz`) before submission. If amount exceeds cap, the order MUST be downgraded to a limit order at price × (1 ± 0.003) with amount clipped to the cap.

- **FR-G14**: A module-level failure ledger `_oversized_market_failures: dict[(exchange_id, pair, side), float]` MUST cache exchange-side oversized-market rejections for 30 minutes. Subsequent orders matching the cached `(pair, side)` MUST preemptively use limit-order downgrade even if size appears compliant. Detection heuristic uses substring match on error text (`"exceeds maximum"`, `"51202"`, `"max_num_orders"`).

### API / Telemetry Layer

- **FR-G15**: Per-decision token ledger and per-node latency timings MUST be captured into `decision_commit.token_usage` and `decision_commit.latency_breakdown`. `tracing.run_graph_traced` MUST resolve trace_id (state → contextvars → uuid4) and call `start_ledger()` in the OUTER coroutine context (not inside an init_decision node whose ContextVar set doesn't propagate to sibling subtasks LangGraph spawns via astream).

- **FR-G16**: `api/routes/decisions.py:VerdictSlim` MUST expose `thesis` and `invalidation` fields. The serializer MUST coerce non-string sources (None, MagicMock in tests) via `_as_str` before pydantic validation.

## Non-Functional Requirements

- **NFR-1**: All guardrails MUST be unit-tested with explicit boundary cases (at-cap, just-below-cap, just-above-cap). Total new test count for this spec: ≥ 39.
- **NFR-2**: Guardrail enforcement MUST NOT increase per-cycle latency by > 50ms (regex parsing + arithmetic only, no I/O).
- **NFR-3**: All guardrail decisions MUST be auditable: lowered confidence and the firing guardrail name MUST be recorded in `verdict.guardrails: list[str]` (audit trail; not used by code except for journaling).

## Acceptance Criteria

**SC-1**: 50 consecutive trading cycles produce ≥ 99% commit-or-rejection journal coverage (no silent pair drops). Today's baseline: ~75-80% due to documented bugs.

**SC-2**: Across the same 50 cycles, ≥ 95% of directional verdicts include `applied:` citations. Pre-fix baseline (10-decision sample): 60%.

**SC-3**: Across 50 cycles, every close action's `pnl` value at journal time stays equal to its value 24 hours later (no retroactive overwrite).

**SC-4**: Across 50 cycles, no `max_total_exposure` rejections occur in normal operation (dual-cap risk model is not the binding constraint when within configured bands).

**SC-5**: When 3 same-direction shorts are open, ≥ 90% of subsequent verdicts on those same 3 pairs (action="short", which means add-to-existing) pass `MacroConcentrationCheck`. Pre-fix baseline: 0% pass rate (all 5 sampled add-to-existing trades were误拒).

**SC-6**: Run-level test coverage: ≥ 39 new tests across debate-anti-ratchet, post-process-verdict, macro-concentration, FRED sanity, contract shape repairs. All pre-existing 2049 tests still pass.

**SC-7**: When any FRED series returns out-of-band, `agent_memory/cases/<commit>.md` MUST NOT contain that value in any agent reasoning string (sanity check zeroed it before the LLM saw it).

## Implementation Status

This spec retroactively documents work landed in the 2026-05-07 → 2026-05-08 monitoring loop. All ~870 LOC of code changes (15 modified files + 4 new test files + 1 new check) are uncommitted on `main` at spec creation time and committed alongside this spec.

| FR | Implementation file(s) | Test file(s) |
|---|---|---|
| FR-G1, FR-G2 | `src/cryptotrader/data/macro.py`, `src/cryptotrader/agents/macro.py` | `tests/test_macro_coverage.py` (+11) |
| FR-G3 | `src/cryptotrader/nodes/data.py` | covered indirectly via verdict tests |
| FR-G4, FR-G7 | `src/cryptotrader/nodes/verdict.py`, `src/cryptotrader/debate/verdict.py` | `tests/test_post_process_verdict.py` (+23) |
| FR-G5 | `src/cryptotrader/models.py` | dataclass smoke tests in existing suite |
| FR-G6 | `src/cryptotrader/nodes/verdict.py:risk_check` | covered via test_nodes.py update |
| FR-G8 | `src/cryptotrader/nodes/debate.py` | `tests/test_debate_anti_ratchet.py` (+6) |
| FR-G9, FR-G10 | `src/cryptotrader/risk/checks/concentration.py`, `src/cryptotrader/risk/gate.py`, `src/cryptotrader/config.py` | `tests/test_macro_concentration.py` (+10) |
| FR-G11 | `src/cryptotrader/nodes/execution.py:_update_trade_tracking` | manual verification (cooldown-active log lines) |
| FR-G12 | `src/cryptotrader/nodes/data.py:_update_one_commit_pnl` | DB inspection + monitoring (SOL pnl=$90.5646 stable) |
| FR-G13, FR-G14 | `src/cryptotrader/execution/exchange.py` | manual verification (DOGE limit-fallback log line) |
| FR-G15 | `src/cryptotrader/tracing.py`, `src/cryptotrader/chat/analysis_runner.py` | manual verification (token_usage non-zero in 22:31 cycle) |
| FR-G16 | `src/api/routes/decisions.py` | `tests/test_contract_shape.py` repaired |

## Out of Scope

The following items were identified during the same audit but are deferred to a future spec:

- **N1** (long-signal asymmetry): system never emits long verdicts despite genuine bullish signals at times. Diagnosis incomplete.
- **N3** (skill library imbalance): macro/news/chain skills are far rarer than tech skills; reflection job tuning required.
- **N6** (drawdown_pct discrepancy): `/api/risk/status` reports drawdown 22% while `/api/portfolio/snapshot` reports 0.22% — calculation reconciliation needed.
- **N8** (regime detection): no trending-vs-chop labelling; bot uses same "breakdown short" template universally.
- **N10** (funding-carry cost modeling): persistent shorts incur ~0.05% equity/day in funding without LLM awareness.
- **N11** (skill maturity FSM not visible in prompt): agents cite skills without knowing observed/probationary/active status.
- **N12** (cross-cycle learning): consecutive same-pair same-direction trades after a loss don't trigger re-thinking.

## Changelog

- 2026-05-08: Spec created retroactively documenting the 12+N4+N2+N7+M2 fixes from the trader-audit monitoring loop. All FRs cross-referenced to the implementing files and test files at commit time.
