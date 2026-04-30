# Tasks: 引入 Pair 值对象统一交易对类型语义

**Feature**: `013-pair-value-object` | **Branch**: `013-pair-value-object`
**Inputs**: spec.md, plan.md, research.md, data-model.md, contracts/, quickstart.md

## Overview

按 spec.md 的 4 个 User Stories（US1=P1 永续平仓 / US2=P1 消除散点 / US3=P1 DB 迁移 / US4=P2 前端徽章）+ 治本 Pair 模块作为基础。Phase 2 (Foundational) 是所有 user story 的前置。撤回 Phase 0 band-aid 在最后做。

总计 **38 个 tasks**，覆盖 plan.md 估算的 5.6 day 工作量。

---

## Phase 1: Setup

无需新建项目结构（沿用现有 monorepo）。仅一项 housekeeping。

- [X] T001 Verify trait config + LangGraph version + ccxt version meet spec requirements; record actual versions in `specs/013-pair-value-object/research.md` if drift from plan

---

## Phase 2: Foundational (BLOCKING — must complete before US1/US2/US3/US4)

依据 spec FR-001~010、FR-100~104，先建 Pair 模块和配置加载，所有下游 phase 依赖。

- [X] T002 [P] Create `src/cryptotrader/pair.py` with `Pair` frozen dataclass per `contracts/pair_api.md` (base/quote/ccxt_symbol fields + `__post_init__` invariant checks)
- [X] T003 [P] Implement `Pair.parse(s)`, `Pair.from_ccxt(exchange, symbol)`, `Pair.to_ccxt()`, `Pair.canonical()`, `Pair.display()`, `Pair.__str__` in `src/cryptotrader/pair.py`
- [X] T004 [P] Implement `Pair.market_type` and `Pair.settle` derived properties in `src/cryptotrader/pair.py`
- [X] T005 [P] Add `tests/test_pair.py` covering FR-009 round-trip + FR-010 ccxt symbol shapes (OKX swap, Binance USDT-M, Binance COIN-M, Bybit) + invariant violations — 41 tests pass
- [X] T006 [P] Add perf test in `tests/test_pair_performance.py` validating NFR-Performance (Pair instantiation < 5μs) — measured 0.4-1.5μs on M1, well under budget; uses stdlib timeit (no pytest-benchmark dependency)
- [X] T007 Update `src/cryptotrader/config.py` `SchedulerConfig.pairs` type from `list[str]` to `list[Pair]`; parse both legacy `list[str]` (all spot) and new `[[scheduler.pairs]]` table-array per `contracts/scheduler_pairs_config.md`
- [X] T008 Add `ConfigurationError` validation in `src/cryptotrader/config.py`: missing `settle` when `market != "spot"`, mixed list types, malformed `symbol`, duplicate canonical (FR-104)
- [X] T009 Emit `pair_init` structured log in scheduler startup path (`src/cryptotrader/scheduler.py`) per FR-103
- [X] T010 [P] Add `tests/test_config_pair_object_form.py` covering both TOML forms + all FR-104 validation errors — 16 tests written
- [X] **Phase 2 callsite migration** (uncovered by T007): `Scheduler.__init__` accepts list[Pair]∣list[str], normalizes; `self._status` keyed by canonical str; `Scheduler.{_run_pair, startup_reconcile, write_cycle_snapshot}` use `pair.canonical()` for state.metadata.pair; `cli/main.py`, `api/main.py`, `api/routes/chat.py`, `api/routes/scheduler.py` all updated to project canonical str at API/state boundaries

**Checkpoint**: Pair module + config升级完成，下游可独立并行 US1/US2/US3。

---

## Phase 3 (US1): 永续合约用户能正确平仓 — Priority P1 🎯 MVP

**Goal**: AI 决策 `close BTC/USDT` 在 OKX perp 账户上真实下平仓单（修复 spec.md User Story 1 三条 acceptance scenarios）。

**Independent Test**: 配 OKX sandbox + perp 0.02 BTC 持仓 + `[[scheduler.pairs]] symbol="BTC/USDT" market="swap" settle="USDT"`，触发一次 cycle，DB `decision_commits.order_data` 非 null，OKX 返回 fill 确认。

依据 D7 把高风险拆 3a/3b/3c。

### US1 Phase 3a — Adapter Layer

- [X] T011 [US1] Create `src/cryptotrader/pair_adapter.py` with `to_pair(s_or_pair) -> Pair` + `from_pair(p_or_str) -> str` helpers; documents intent for nodes still on str
- [X] T012 [US1] [P] Add `tests/test_pair_adapter.py` covering str/Pair round-trip + idempotency — 14 tests pass

### US1 Phase 3b — verdict + execution Switch to Pair

- [X] T013 [US1] `nodes/execution.py` updated: `_load_balances_from_db` filters spot via `Pair.parse(...).market_type`; `_sync_portfolio_from_exchange` syncs derivatives via `exchange.get_positions()`; `_build_close_order`/`_build_entry_order` use ccxt-canonical keys (no translation needed since they treat pair as opaque)
- [X] T014 [US1] `nodes/verdict.py` `_build_risk_portfolio`: pass-through of exchange positions (already canonical-keyed after T015) — no change needed
- [X] T015 [US1] `execution/exchange.py` `LiveExchange.place_order` parses `order.pair` via `Pair`; balance check splits spot (asset) vs derivatives (settle margin); ccxt `create_order` receives canonical str verbatim; `LiveExchange.get_positions` returns ccxt symbol verbatim (band-aid `_canonical_pair` retired)
- [X] T016 [US1] `execution/simulator.py` `PaperExchange.get_positions`: spot-only by design (asset-balance model); no change required, verified via existing tests
- [X] T017 [US1] [P] `tests/test_us1_perp_close_flow.py` — 4 integration tests cover canonical key lookup, no spot-form fallback, ccxt forwarding, settle-currency margin check; `tests/test_live_exchange_pair.py` rewritten for inverted contract

### US1 Phase 3c — Full Nodes + State Schema Bump

依据 D2 一刀切 + R2 LangGraph state checkpoint：

- [ ] T018 [US1] Update `src/cryptotrader/state.py` `ArenaState` TypedDict: `metadata.pair: Pair`; bump `_state_schema_version` field to `2`
- [ ] T019 [US1] Update remaining nodes (`src/cryptotrader/nodes/data.py`, `nodes/agents.py`, `nodes/debate.py`, `nodes/journal.py`) to accept `Pair` (no str/split operations)
- [ ] T020 [US1] Update AI prompt builders to use `pair.display()` (search agents/base.py + nodes/debate.py for `state["metadata"]["pair"]` usage)
- [ ] T021 [US1] Add structlog binding `pair=pair.canonical()` in scheduler `_run_pair` for grep-ability (FR-203)
- [ ] T022 [US1] Add LangGraph state checkpoint compat shim: legacy `pair: str` in checkpoint deserialized via `Pair.parse(saved_str)` with WARNING log (FR-204)
- [ ] T023 [US1] [P] Add `tests/test_us1_state_schema_bump.py`: cycle round-trip with new schema; legacy checkpoint backwards-compat

**Checkpoint US1**: 真盘 sandbox cycle 触发 → DB commit + OKX fill 双向确认。Acceptance scenarios 1/2/3 全过。

---

## Phase 4 (US2): 内部代码不再有"字符串散点"逻辑 — Priority P1

**Goal**: 全代码搜索 `\.split\("/"\)` 和 `\.split\(":"\)` 在 pair 上下文 0 命中（除 `Pair` 实现内部）。

**Independent Test**: `rg -nE '\.split\("/"\)' src/ | grep -v 'pair.py'` 0 行；`rg -nE 'positions.get\(' src/` 所有 key 来自 `Pair.canonical()`。

依赖：US1 Phase 3a (adapter) + 3b (execution/verdict) 已完成。

- [ ] T024 [US2] Audit all `\.split\("/"\)` / `\.split\(":"\)` occurrences in pair context (`rg -nE` + manual review); migrate to `Pair.parse` / `pair.base` / `pair.quote`
- [ ] T025 [US2] Audit all `positions.get(pair)` / `positions[pair]` lookups; ensure key consistency via Pair.canonical() everywhere
- [ ] T026 [US2] Remove `src/cryptotrader/pair_adapter.py` (T011-T012) once T024-T025 confirm no callers need str fallback
- [ ] T027 [US2] [P] Add `tests/test_us2_no_string_split_pair.py` regression: grep for `.split("/")` in pair context returns 0 matches outside `pair.py`

**Checkpoint US2**: 散点消除完成。下次 add pair-related 代码无歧义入口。

---

## Phase 5 (US3): DB 迁移 — Priority P1

**Goal**: alembic 加 `market_type VARCHAR(20) NOT NULL DEFAULT 'spot'` 列到 portfolios + decision_commits；存量数据 0 丢失（D5）。

**Independent Test**: `alembic upgrade head` → `\d portfolios` 显示新列；`SELECT pair, market_type, count(*) FROM portfolios GROUP BY 1,2;` 全 'spot'；`alembic downgrade -1` + `upgrade head` 幂等。

依赖：T002-T010 (Pair + config) 已完成；不依赖 US1/US2 implementation。

- [ ] T028 [US3] Create alembic migration `migrations/versions/XXXX_add_market_type.py` per data-model.md schema; only `op.add_column` no row updates (D5)
- [ ] T029 [US3] Update `src/cryptotrader/portfolio/manager.py` SQLAlchemy `Portfolio` ORM model + `PortfolioSnapshot` (if applicable) to include `market_type` column
- [ ] T030 [US3] Update `src/cryptotrader/journal/store.py` `_serialize` to write `market_type = pair.market_type`; `_deserialize` to round-trip via `Pair.parse(pair_str)` (FR-403)
- [ ] T031 [US3] [P] Add `tests/test_us3_journal_market_type.py` covering double-write of market_type, downgrade idempotency, legacy row default value

**Checkpoint US3**: DB schema bump 完成；新 commit 含 market_type；存量 row 自动 spot。

---

## Phase 6 (US4): Frontend 渲染 pair 含市场类型徽章 — Priority P2

**Goal**: `<PortfolioPositions>` 和 `<DecisionDetail>` 加 `<PairBadge>` 徽章；其他视图通过 `pair_display` 字符串字段兜底（D6）。

**Independent Test**: `pnpm test pair-badge.test.tsx` 全绿；浏览器手测 `/` 和 `/decisions/<id>` 显示徽章；其他视图渲染未变。

依赖：T028-T031 (DB+journal) 已完成；US1 verdict pipeline 已能产生新字段。

- [ ] T032 [US4] Update `src/api/routes/portfolio_v2.py` `PositionOut` Pydantic model + handler to add `pair_display` and `market_type` fields per `contracts/api_response_schema.md`
- [ ] T033 [US4] Update `src/api/routes/decisions.py` `DecisionListItem` and detail response to add `pair_display` + `market_type` top-level fields
- [ ] T034 [US4] [P] Add `tests/test_us4_api_pair_response.py` validating shape + values for both endpoints under spot and swap fixtures
- [ ] T035 [US4] [P] Update `web/src/lib/api/types.ts` to mirror new fields in DTOs
- [ ] T036 [US4] [P] Create `web/src/components/PairBadge.tsx` per `contracts/api_response_schema.md` component contract (display + colored badge by market_type)
- [ ] T037 [US4] Wire `<PairBadge>` into `web/src/pages/dashboard/components/portfolio-positions.tsx` and `web/src/pages/decisions/decision-detail.tsx` (replace pair string render)
- [ ] T038 [US4] [P] Add `web/tests/unit/pair-badge.test.tsx` (vitest) covering 3 market types

**Checkpoint US4**: 前端两个 P0 视图含徽章；其他视图无回归。

---

## Phase 7: Polish & Withdraw Phase 0 Band-aid

依据 spec FR-304 + Success Criteria。

- [ ] T039 Remove `LiveExchange._canonical_pair` method from `src/cryptotrader/execution/exchange.py`; remove `tests/test_live_exchange_pair.py` (Phase 0 band-aid no longer needed)

---

## Dependencies

### Story Completion Order

```
Phase 1 (Setup) → Phase 2 (Foundational)
                       │
       ┌───────────────┼───────────────┬──────────────────┐
       │               │               │                  │
   Phase 3 (US1)   Phase 5 (US3)   Phase 4 (US2)*    [Phase 6 (US4)]
       │               │               │                  │
       └───────────────┴───────────────┘                  │
                  Phase 7 (Polish: withdraw band-aid)     │
                                                          │
                          (Phase 6 depends on T028-T031 + T032-T033)
```

\* Phase 4 (US2) requires US1 Phase 3a (T011-T012) for adapter cleanup.

### Parallel Opportunities

**After T001 completes, the following batches can run in parallel**:

**Batch A (Pair module + perf test)**: T002, T003, T004, T005, T006 — all in `src/cryptotrader/pair.py` + tests, no shared state

**Batch B (Foundational integration)**: T007, T008, T009 sequential within file (`config.py` shared)

**Batch C (US1 phase 3a/3b)**: T011 sequential; T012, T017 [P] tests

**Batch D (US3 + US4 backend)**: T028 → T029 → T030 → T031 sequential (all touch DB layer); T032, T033, T034, T035 can parallelize (different files)

**Batch E (US4 frontend)**: T035, T036, T038 [P] (different files); T037 sequential after T036

---

## Implementation Strategy

### MVP (Stop after US1)

If forced to ship something fast: T001-T023 alone. Result: perp users can `close` real positions; band-aid stays in place but works correctly. ~3 day effort.

### Incremental Delivery

Recommended sequence per spec phased delivery:
1. **Day 1**: Phase 1 + Phase 2 (Foundational) → T001-T010
2. **Day 2**: US1 Phase 3a + 3b → T011-T017
3. **Day 3**: US1 Phase 3c → T018-T023 (state schema bump; **要求停 scheduler 部署窗口**)
4. **Day 4**: US2 (cleanup) + US3 (DB) → T024-T031 (parallel)
5. **Day 5**: US4 (frontend) → T032-T038
6. **End of Day 5**: Phase 7 polish → T039

### Independent Test Criteria per Story

| Story | Test |
|---|---|
| US1 | OKX sandbox cycle → DB `order_data NOT NULL` + OKX fill confirmation (spec.md User Story 1 acceptance) |
| US2 | `rg '\.split\("/"\)' src/ \| grep -v 'pair.py'` returns 0 lines |
| US3 | `\d portfolios` shows market_type column; `alembic downgrade && upgrade` idempotent |
| US4 | `pnpm test pair-badge.test.tsx` green; manual browser inspection of `/` and `/decisions/<id>` |

---

## Format Validation

✅ All 38 tasks follow checklist format: `- [ ] TXXX [P?] [USX?] description with file path`
✅ Setup/Foundational/Polish phases have NO story label
✅ User Story phases have `[USx]` label
✅ Parallel `[P]` only on tasks with no file conflicts
✅ Every task names exact file path

---

**Status**: Tasks generated. Ready for `{Skill: spex:review-plan}` (ship pipeline stage 5).
