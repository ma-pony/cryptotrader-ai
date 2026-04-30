# Implementation Plan: 引入 Pair 值对象统一交易对类型语义

**Branch**: `013-pair-value-object` | **Date**: 2026-04-30 | **Spec**: [spec.md](./spec.md)

## Summary

把交易对（trading pair）从散点字符串升级为 `Pair` 值对象。ccxt unified symbol 是唯一真实来源（即 `Pair.canonical()` ≡ `Pair.to_ccxt()` ≡ ccxt 的 `BTC/USDT` / `BTC/USDT:USDT` 格式），通过 frozen dataclass 提供 `display()` / `market_type` 等派生属性。配置 `[scheduler].pairs` 升级为对象数组（每个 pair 独立指定 `market`/`settle`），向后兼容旧 `list[str]` 形式。`state.metadata.pair` / `Order.pair` / DB `pair` 列保持字符串（即 canonical）但加显式 `market_type` 列；nodes 层全面接收 `Pair` 实例。前端最小切片：`PortfolioPositions` + `DecisionDetail` 加 `<PairBadge>` 徽章，其他视图通过 `pair_display` 字符串字段兜底。

Phase 0 band-aid 已合并（commit `d05a0bf`，`LiveExchange._canonical_pair` 通过 `exchange.market()` 元信息标准化 ccxt symbol）。本 plan 推进 Phase 1-5。

## Technical Context

**Language/Version**: Python 3.12 (后端) / TypeScript 5.9 (前端)
**Primary Dependencies**: LangGraph 1.0+ / SQLAlchemy 2.0+ / Alembic 1.13+ / ccxt 4.x / FastAPI / React 19 / Vite 7
**Storage**: PostgreSQL via SQLAlchemy async + alembic migrations；表受影响：`portfolios`、`portfolio_snapshots`、`decision_commits`
**Testing**: pytest + pytest-asyncio + ccxt mocking (no live calls)；vitest (frontend)
**Target Platform**: Linux server (Docker)；macOS dev；前端 Chrome/Edge/Firefox/Safari
**Project Type**: web-service (FastAPI + React frontend) + LangGraph 多代理交易系统
**Performance Goals**: `Pair` 实例化 < 5μs；不应进入 trading_cycle 热路径瓶颈
**Constraints**:
- LangGraph state checkpoint 跨 cycle 共享 — schema 升级必须单 PR 一刀切（D2）
- DB 不可逆 migration — 只加列，存量 row 不动（D5）
- 前端 30+ 处 string pair 散点 — 最小切片，留 spec 跟踪后续（D6）
- ccxt 是真实来源 — `Pair.canonical()` ≡ ccxt unified symbol（D1）
**Scale/Scope**: 单 exchange (OKX) ×2 pairs 当前；spec 设计支持未来扩展到 N exchange × M pair 混 spot/perp

## Constitution Check

`.specify/memory/constitution.md` 当前是 template placeholder（未填写实际原则），无具体 gate。本 spec 自我约束：

- ✅ **测试先行**：每个 Phase 都要求新增 unit test，FR-009/010 明示 round-trip 覆盖
- ✅ **无破坏性变更**：D3 决议保留 `Order.pair: str`；D5 不动 DB 存量 row；D6 frontend 旧视图不动
- ✅ **观测性**：FR-103 启动日志 `pair_init`；FR-203 structlog 字段 `pair=pair.canonical()`
- ✅ **独立可回滚**：Phase 1-5 每个 phase 独立可部署，3 拆 3a/3b/3c 进一步降低单步风险
- ✅ **简洁性**：Pair 仅 3 字段（`base/quote/ccxt_symbol`），不重建类型枚举（D1）

无 constitution gate 失败。

## Project Structure

### Documentation (this feature)

```text
specs/013-pair-value-object/
├── plan.md              # This file
├── spec.md              # Already exists (committed aa57ceb)
├── research.md          # Phase 0 output — see below
├── data-model.md        # Phase 1 output — see below
├── contracts/           # Phase 1 output — Pair API + config schema
│   ├── pair_api.md
│   ├── scheduler_pairs_config.md
│   └── api_response_schema.md
├── quickstart.md        # Phase 1 output — dev/test instructions
└── tasks.md             # Phase 2 output (created by /speckit-tasks)
```

### Source Code (repository root)

```text
src/cryptotrader/
├── pair.py              # NEW — Pair value object (Phase 1)
├── pair_adapter.py      # NEW — adapter helpers (Phase 3a, removed at end of 3c)
├── config.py            # MOD — parse [scheduler].pairs into list[Pair] (Phase 2)
├── state.py             # MOD — metadata.pair: Pair (Phase 3c)
├── nodes/
│   ├── data.py          # MOD — accept Pair (Phase 3c)
│   ├── agents.py        # MOD — accept Pair (Phase 3c)
│   ├── debate.py        # MOD — accept Pair (Phase 3c)
│   ├── verdict.py       # MOD — accept Pair (Phase 3b)
│   ├── execution.py     # MOD — accept Pair (Phase 3b)
│   └── journal.py       # MOD — Pair ↔ canonical str serialize (Phase 3c)
├── execution/
│   ├── exchange.py      # MOD — remove _canonical_pair band-aid (after Phase 3c)
│   └── simulator.py     # MOD — get_positions returns canonical key (Phase 2)
├── portfolio/
│   └── manager.py       # MOD — get_portfolio uses pair.canonical() keys (Phase 2)
└── journal/
    └── store.py         # MOD — write/read market_type column (Phase 4)

src/api/routes/
├── portfolio_v2.py      # MOD — add pair_display + market_type fields (Phase 5)
└── decisions.py         # MOD — same (Phase 5)

migrations/versions/
└── XXXX_add_market_type.py  # NEW — alembic migration (Phase 4)

web/src/
├── lib/api/types.ts     # MOD — add pair_display + market_type to DTOs (Phase 5)
└── components/
    └── PairBadge.tsx    # NEW — display + market type badge (Phase 5)

tests/
├── test_pair.py                       # NEW — Pair value object (Phase 1)
├── test_pair_adapter.py               # NEW — adapter helpers (Phase 3a)
├── test_config_pair_object_form.py    # NEW — new pairs config syntax (Phase 2)
├── test_live_exchange_pair.py         # MOD — remove band-aid tests (after 3c)
├── test_journal_market_type.py        # NEW — DB market_type column (Phase 4)
└── test_api_pair_response.py          # NEW — API response schema (Phase 5)

web/tests/unit/
└── pair-badge.test.tsx                # NEW — PairBadge rendering (Phase 5)
```

## Phase 0: Outline & Research

研究产物 → `research.md`。本 spec 的 design decisions D1-D7 已经在 brainstorm 阶段 resolve（spec.md `## Design Decisions` 表）。Phase 0 主要做实施层面的技术调研。

### Research Topics

1. **ccxt unified symbol 的边界情况**
   - **Question**: ccxt 的 perp/swap symbol 是否一律 `BASE/QUOTE:SETTLE`？futures (delivery) 和 options 呢？inverse perp（COIN-M）呢？
   - **Method**: 跑 OKX、Binance、Bybit 在 `ccxt.markets` 字典上抽样
   - **Decision impact**: 决定 `Pair.parse()` / `Pair.from_ccxt()` 的容错策略

2. **LangGraph state schema migration 模式**
   - **Question**: LangGraph 推荐怎么处理 state schema 版本升级？checkpoint 兼容性怎么测？
   - **Method**: 查 langgraph docs `MemorySaver` / `Checkpointer` migration patterns
   - **Decision impact**: Phase 3c 的 checkpoint migration 实现路径

3. **Alembic + SQLAlchemy 加列性能**
   - **Question**: 给 `decision_commits` 加非空带默认列，PG 是否要全表锁？
   - **Method**: 查 PG 文档 + 试 `decision_commits` 表大小（已有 ~14 行）
   - **Decision impact**: 当前数据量小，全表锁可接受；但记录下来供未来万行规模时的迁移参考

4. **TOML 配置混合 list[str] / list[dict]**
   - **Question**: tomllib 怎么解析 `pairs = ["BTC/USDT", {symbol="ETH/USDT", market="swap", settle="USDT"}]`？
   - **Method**: 写 5 行 demo 验证
   - **Decision impact**: FR-100 双形式兼容的具体写法

5. **frozen dataclass 性能**
   - **Question**: NFR-Performance 要求 `Pair` 实例化 < 5μs。frozen dataclass 在 hot path 里 OK 吗？
   - **Method**: pytest-benchmark 跑 1M 次实例化
   - **Decision impact**: 如超标，回落到 NamedTuple

**Output**: `research.md` 含 5 项研究的 Decision/Rationale/Alternatives

## Phase 1: Design & Contracts

### data-model.md (新增数据实体)

唯一新增实体：`Pair`。其他都是现有 model 的字段类型升级。

```python
@dataclass(frozen=True)
class Pair:
    base: str            # "BTC"
    quote: str           # "USDT"
    ccxt_symbol: str     # "BTC/USDT" 或 "BTC/USDT:USDT"

    @classmethod
    def parse(cls, s: str) -> Pair: ...
    @classmethod
    def from_ccxt(cls, exchange, symbol: str) -> Pair: ...

    def to_ccxt(self) -> str: ...
    def canonical(self) -> str: ...   # ≡ ccxt_symbol
    def display(self) -> str: ...

    @property
    def market_type(self) -> Literal["spot", "swap", "future"]: ...
```

数据库变更（Phase 4）：
- `portfolios` + `decision_commits` + `portfolio_snapshots` 加 `market_type VARCHAR(20) NOT NULL DEFAULT 'spot'` 列
- 存量 row 自动取默认值，不修改任何 row

### contracts/

#### pair_api.md
完整列出 `Pair` 类 8 个 public method/property 的签名 + invariant + raises 列表 + 1-2 行 doctest 示例。

#### scheduler_pairs_config.md
TOML 配置语法（FR-100），含旧 list[str] 形式的兼容映射规则与新对象数组形式的字段约束。

#### api_response_schema.md
`/api/portfolio/snapshot` + `/api/decisions/{id}` 响应里新增的 `pair_display` + `market_type` 字段的 JSON Schema。

### quickstart.md
开发者怎么本地验证：
- 跑 `pytest tests/test_pair.py` 确认 Pair 单测全绿
- 起本地 docker compose；curl `/api/portfolio/snapshot` 看新字段
- 在 sandbox 模拟一次 cycle，看 `cycle complete` log 含 `market_type`
- 对照 spec 的 Acceptance Scenarios 1-3 (User Story 1) 手测 OKX swap close 路径

### Agent context update
跑 `.specify/scripts/bash/update-agent-context.sh claude` 自动更新 agent context。spex superpowers trait 要求随后 `git checkout -- CLAUDE.md` 还原 — CLAUDE.md 是用户维护文件，不应自动改写。

## Phase 2: Tasks (created by /speckit-tasks)

按 spec 的 phased delivery（5.6 day 总估时）拆分成可执行 tasks：

- **P1 tasks** — Pair 模块（FR-001 ~ 010）：~6 tasks
- **P2 tasks** — config 升级（FR-100 ~ 104）：~5 tasks
- **P3a tasks** — adapter 层：~3 tasks
- **P3b tasks** — verdict + execution 切 Pair：~5 tasks
- **P3c tasks** — 全 nodes + state schema：~7 tasks
- **P4 tasks** — DB 加列：~4 tasks
- **P5 tasks** — API + frontend：~6 tasks
- **撤回 0 task** — 删除 band-aid：1 task

预计 ~37 个 tasks，由 `/speckit-tasks` 生成。

## Risk Register

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| LangGraph state schema bump 破坏 in-flight cycle | 中 | 高 | 部署前停 scheduler、清空 checkpoint；Phase 3c 单 PR 一刀切 |
| ccxt 真盘 symbol 与 sandbox 差异 | 低 | 中 | Phase 5 quickstart 含真盘 / sandbox 双侧验证 |
| 前端最小切片不够，其他视图渲染断裂 | 中 | 低 | API 双写 `pair_display` 字段，旧视图自动 fallback |
| alembic 加列在大表上锁等待 | 低 | 中 | 当前 decision_commits ~14 行，无影响；记录万行规模迁移策略到 research.md |
| Pair 实例化超 5μs SLO | 低 | 低 | pytest-benchmark gate；超标降级 NamedTuple |

## Re-evaluate Constitution Check

无 constitution 实际原则，跳过。

---

**Status**: Planning Complete. Ready for `/speckit-tasks` to generate `tasks.md`.
