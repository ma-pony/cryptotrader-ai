# Feature Specification: 引入 Pair 值对象统一交易对类型语义

**Feature Branch**: `013-pair-value-object`
**Created**: 2026-04-30
**Status**: Draft
**Input**: 现状是项目把 `"BTC/USDT"` 当作纯字符串到处传，但实际承载的是带"市场类型"语义的复合概念（spot vs perp swap vs futures），ccxt 边界返回的 `BTC/USDT:USDT` 与项目内部 `BTC/USDT` 不匹配，导致 read 侧 lookup 静默 miss、write 侧可能下错市场。需要引入 `Pair` 值对象统一表达。

## Design Decisions (2026-04-30 brainstorm)

| 问题 | 决定 | 理由 |
|---|---|---|
| **D1 — Pair 是否自建市场类型枚举？** | 否。**Pair 把 ccxt 统一 symbol 作为唯一真实来源**，不重建类型系统 | 项目目前只跑 OKX，跨 exchange 类型差异由 ccxt market metadata 处理，自建枚举要不停跟 ccxt 对账 |
| **D2 — `state.metadata.pair` 升级是否允许 str/Pair 共存？** | 否，**单 PR 一刀切**（state schema bump + nodes + journal serializer 同 PR） | LangGraph checkpoint 跨 cycle 共享 state，半升级会破坏 `prev_analyses` 等共享字段 |
| **D3 — `Order.pair` 字段类型** | 保持 `str`，约束为 `pair.canonical()` 形式（即 ccxt unified symbol） | 不破坏数十处 Order 构造点；str 在内部已无歧义 |
| **D4 — `place_order` 怎么知道用 spot 还是 perp？** | 配置里**每个 pair 独立指定 market_type**，scheduler.pairs 升级为对象数组 | 灵活性：未来同一账户可能 spot+perp 混用；迁移路径：旧 list[str] 形式视作全 spot |
| **D5 — DB 存量数据是否改值？** | 否，**只加列**（`market_type` 默认 `"spot"`），存量 row 不动 | 不可逆 + 外部 SQL 引用 `pair = 'BTC/USDT'` 不破；新 commit 双写新列 |
| **D6 — 前端范围** | **最小切片**：`<PortfolioPositions>` + `<DecisionDetail>` 加徽章，其他视图先用 `pair_display` 字符串字段兜底 | 前端 30+ 处 string 用法，全量 Pair interface 重构跟主修复脱节，留作单独 spec |
| **D7 — Phase 3 拆分** | 拆 3a/3b/3c：先加 adapter 层、再切 verdict+execution、最后 state schema bump | 风险最大的 phase，分三步可独立验证、灰度回滚 |

## Background

2026-04-30 排查"close on flat → 0 真实交易"事件时确认：

| 数据来源 | 看到的 pair 字符串 |
|---|---|
| `config/local.toml` `[scheduler].pairs` | `"BTC/USDT"` |
| `state["metadata"]["pair"]` | `"BTC/USDT"` |
| DB `portfolios` 表 | `"BTC/USDT"` |
| AI prompts / journal | `"BTC/USDT"` |
| **OKX ccxt `fetch_positions` 返回 dict 的 key** | **`"BTC/USDT:USDT"`** (perp swap) |

OKX sandbox 用户实际持有 `0.02 BTC` 永续合约。AI 决定 `close BTC/USDT` 时，`_build_close_order` 走到 `positions.get("BTC/USDT")` 找不到（因为 dict 里是 `"BTC/USDT:USDT"`），返回 None → 不下平仓单。`_build_entry_order` 同样的 lookup 偏差更危险：算 `target_amount - existing` 时把 existing 当作 0，可能在已有 perp 仓上叠满仓单。

短期已用 `LiveExchange._canonical_pair` 在 ccxt 边界把 `BTC/USDT:USDT` 标准化回 `BTC/USDT`（commit pending），治标。**本 spec 跟踪的是把"市场类型"作为一等公民引入项目模型**的治本工作。

## User Scenarios & Testing *(mandatory)*

### User Story 1 — 永续合约用户能正确平仓 (Priority: P1)

作为 OKX 永续合约账户用户，配置 `engine = "live"` + `[exchanges.okx].market_type = "swap"` 后，AI 决策 `close BTC/USDT` 必须真的在我的 perp 账户上下卖单平掉对应数量的 BTC 永续合约，而不是静默 no-op。

**Why P1**：直接阻断生产 trading。今日 cycle 全部 `close on flat` 没产生任何真实交易；如果 AI 切换到 long/short 决策，会下错单到不存在资金的 spot 账户，导致 `InsufficientFunds` 异常或更糟（spot/perp 混账户时下到错的产品上）。

**Acceptance Scenarios**:
1. **Given** OKX sandbox 已有 0.02 BTC perp 持仓，配置 `market_type = "swap"`，**When** AI verdict = `close`，**Then** `place_order` 向 ccxt 提交的 symbol = `"BTC/USDT:USDT"`，amount = 0.02，side = sell；交易所确认 fill；DB `decision_commits.order_data` 非 null
2. **Given** 同上，**When** AI verdict = `long` scale=0.5，已有 0.02 BTC perp，**Then** 下单 amount = `target - 0.02`（不是 target，避免叠仓）
3. **Given** 配置 `market_type = "spot"`，**When** AI verdict = `long`，**Then** ccxt symbol 为 `"BTC/USDT"`（不带 `:USDT`），目标到 spot 账户

### User Story 2 — 内部代码不再有"字符串拼接 / 拆分 pair"的散点逻辑 (Priority: P1)

作为代码维护者，我修改任何涉及 pair 的代码时不需要担心字符串格式分歧（`BTC/USDT` vs `BTC/USDT:USDT` vs `BTCUSDT`），所有 pair 都是 `Pair` 值对象，要拿"显示给 AI 的名字"调 `pair.display()`，要拿"传给 ccxt 的 symbol"调 `pair.to_ccxt()`，要"做字典 key"用 `pair`（dataclass(frozen=True) 自动 hashable）。

**Why P1**：当前散点（band-aid 至少 4 处：`execution.py:228/280/314`、`data.py:308`，未来一定会再发生类似 mismatch）是技术债，每次新加 pair-相关代码都有踩坑风险。

**Acceptance Scenarios**:
1. **Given** 全代码搜索 `\.split\("/"\)` 和 `\.split\(":"\)` 在 pair 上下文，**Then** 0 命中（除 `Pair.parse()` / `Pair.from_ccxt()` 的实现内）
2. **Given** 全代码搜索 `positions.get(` 和 `positions[`，**Then** 所有 key 类型都是 `Pair`（mypy/pyright 强制）
3. **Given** ccxt 返回任意未知 symbol 格式，**When** 通过 `Pair.from_ccxt(exchange, symbol)`，**Then** 解析为正确的 `Pair(base, quote, market_type, settle)` 或抛 `UnknownMarketTypeError`

### User Story 3 — DB 迁移把存量字符串安全升级 (Priority: P1)

作为数据库管理员，运行 alembic migration 后，`portfolios` / `decision_commits` / `portfolio_snapshots` 表里的 `pair` 字段从 `"BTC/USDT"` 升级为带类型信息的形式（候选：`"BTC/USDT-spot"` / `"BTC/USDT-swap-USDT"`），历史数据 100% 可读，前端展示时回落为简短 `"BTC/USDT"`（hover/详情显示完整类型）。

**Why P1**：存量决策 / 持仓历史不能因模型升级丢失。

### User Story 4 — Frontend 渲染 pair 含市场类型徽章 (Priority: P2)

前端 portfolio / decisions / market view 每处渲染 pair 的地方都加一个小徽章：`spot` / `perp` / `futures`，点击徽章可看完整 ccxt symbol。

## Functional Requirements *(mandatory)*

### Pair Value Object (FR-001 ~ 010)

依据 D1：Pair 把 ccxt unified symbol 作为唯一真实来源，不重建类型枚举。

- **FR-001**: 新增 `src/cryptotrader/pair.py` 模块定义 `Pair` 为 `@dataclass(frozen=True)`：
  ```python
  @dataclass(frozen=True)
  class Pair:
      base: str            # "BTC"
      quote: str           # "USDT"
      ccxt_symbol: str     # "BTC/USDT" 或 "BTC/USDT:USDT" — 唯一真实来源
  ```
  无显式 `market_type` 字段；`market_type` 是 derived property（`"swap" if ":" in ccxt_symbol else "spot"`）。
- **FR-002**: `Pair.parse(s: str) -> Pair` 解析 canonical 字符串（即 ccxt unified symbol），无 ccxt 元数据时由 `:` 后缀 + base/quote 字面拆分推断
- **FR-003**: `Pair.from_ccxt(exchange, symbol: str) -> Pair` 通过 `exchange.market(symbol)` 元信息构造，缺 metadata 时降级为 `parse()`
- **FR-004**: `Pair.to_ccxt() -> str` 直接返回 `self.ccxt_symbol`（已是 ccxt 统一格式）
- **FR-005**: `Pair.display() -> str` 返回 UI/AI 友好形式（`"BTC/USDT (perp)"` 或 `"BTC/USDT"`）
- **FR-006**: `Pair.canonical() -> str` ≡ `Pair.to_ccxt()` — DB/state/config 的标准序列化即 ccxt unified symbol（无后缀的就是 spot，带 `:SETTLE` 的就是 perp/futures）
- **FR-007**: `Pair.market_type` derived property: `"swap" / "spot" / "future"`，根据 `ccxt_symbol` 后缀和 ccxt market flag 推断
- **FR-008**: `Pair.__hash__` / `Pair.__eq__` 由 frozen dataclass 自动生成，可作 dict key
- **FR-009**: 单元测试覆盖 round-trip：parse → to_ccxt → from_ccxt → 同一对象
- **FR-010**: 单元测试覆盖 ccxt 各 exchange 的 symbol 形态（OKX swap、Binance USDT-M、Binance COIN-M、Bybit、dYdX）— 所有都应能 round-trip 而不丢信息

### Config (FR-100 ~ 109)

依据 D4：每个 pair 独立指定 market type，scheduler.pairs 升级为对象数组；旧 list[str] 形式视作全 spot。

- **FR-100**: `[scheduler].pairs` 接受两种形式：
  - 旧（向后兼容）：`pairs = ["BTC/USDT", "ETH/USDT"]` — 全部视作 spot
  - 新：
    ```toml
    [[scheduler.pairs]]
    symbol = "BTC/USDT"
    market = "swap"      # 可选，默认 "spot"
    settle = "USDT"      # market != "spot" 时必填
    ```
  内部统一解析为 `list[Pair]`
- **FR-101**: 删除草案里 `[exchanges.<id>].market_type` 设计（D4 改为 per-pair）
- **FR-102**: 配置加载时 `Pair` 实例化一次，缓存为 `list[Pair]` 挂在 `cfg.scheduler.pairs`
- **FR-103**: 启动时打印一行结构化日志 `pair_init: spot=[..] swap=[..]` 供运维核对
- **FR-104**: 配置校验：`market != "spot"` 时 `settle` 必须存在，否则启动时抛 `ConfigurationError`

### State / Nodes (FR-200 ~ 219)

依据 D2：单 PR 一刀切；依据 D3：Order.pair 保持 str (= pair.canonical())。

- **FR-200**: `state["metadata"]["pair"]` 类型从 `str` 改为 `Pair`，并 bump LangGraph state schema 版本号字段
- **FR-201**: `nodes/data.py` `nodes/agents.py` `nodes/debate.py` `nodes/verdict.py` `nodes/execution.py` `nodes/journal.py` 接收 `Pair`，内部不再做 split / suffix 操作
- **FR-202**: AI prompt 模板使用 `pair.display()`
- **FR-203**: structlog 字段 `pair=pair.canonical()` 保持 grep-able
- **FR-204**: state checkpoint 反序列化时通过 `Pair.parse(saved_str)` 重建；旧 checkpoint 加载向 `state.metadata.pair` 写入 `Pair.parse(legacy_str)` 视作 spot

### Exchange Boundary (FR-300 ~ 319)

依据 D3：Order.pair 保持 str = pair.canonical()。

- **FR-300**: `LiveExchange.get_positions() -> dict[str, dict]` — key 是 `pair.canonical()`（即 ccxt unified symbol）。下游可用 `Pair.parse(key)` 重建对象。
- **FR-301**: `LiveExchange.place_order(order: Order)` — 直接用 `order.pair`（已是 ccxt unified symbol）提交到 ccxt，无需翻译
- **FR-302**: `Order.pair: str`（不变），但**约束注释**写明：必须是 `pair.canonical()` 形式（ccxt unified symbol）。Order 构造点用 `Pair(...).canonical()` 生成
- **FR-303**: `PaperExchange.get_positions()` 同步：返回 dict 的 key 与 LiveExchange 一致（spot pair 直接，无 perp 概念时不需要后缀）
- **FR-304**: 撤回 `LiveExchange._canonical_pair` band-aid（D5 后所有写入用 ccxt unified symbol 直接做 dict key）

### Storage (FR-400 ~ 419)

依据 D5：只加列不改值。存量 row `pair` 字段保留原值，新增 `market_type` 列默认 `"spot"`，新 commit 双写。

- **FR-400**: `portfolios` 表加列 `market_type VARCHAR(20) NOT NULL DEFAULT 'spot'`；存量 row 自动取默认值
- **FR-401**: `decision_commits` 表同样加 `market_type` 列
- **FR-402**: `portfolio_snapshots` 表 account_id 不变；如有 position 关联表，同步加列
- **FR-403**: `JournalStore._serialize` 写入 `pair = pair.canonical(), market_type = pair.market_type`；`_deserialize` 读取后用 `Pair.parse(pair_str)` 重建（market_type 列冗余但作为 sanity check）
- **FR-404**: 提供 `arena db migrate` 命令一键执行 alembic upgrade，不修改任何 row 数据
- **FR-405**: 外部 SQL 引用 `pair = 'BTC/USDT'` 仍能匹配存量 spot 数据；新 perp commit 用 `pair = 'BTC/USDT:USDT'`

### API / Frontend (FR-500 ~ 519)

依据 D6：最小切片。后端始终返回 `pair_display` 字符串字段，前端只在两个 P0 视图里加徽章。

- **FR-500**: `/api/portfolio/snapshot` 响应里每个 position 含三个字段：`pair` (= canonical, ccxt 形式)、`pair_display` (UI 友好)、`market_type` (`"spot" | "swap"`)
- **FR-501**: `/api/decisions/{id}` 顶层加 `pair_display` + `market_type`
- **FR-502**: 前端 `<PairBadge pair_display={...} market_type={...}/>` 组件 — 显示文本 + 小徽章
- **FR-503**: `<PortfolioPositions>` 表格里每行 pair 列改用 `<PairBadge>`
- **FR-504**: `<DecisionDetail>` 头部 pair 标题改用 `<PairBadge>`
- **FR-505**: 其他视图（market view、chat、TradingView widget 等）继续用 `pair_display` 字符串字段，**不引入** `Pair` TS interface — 留作单独 spec
- **FR-506**: 前端不需要 `Pair.split()` / `Pair.startsWith()` 类的本地 helper；后端给的字符串字段直接展示

## Non-Functional Requirements

- **NFR-Migration**: 升级期间允许 `Pair` 与 `str` 临时共存（adapter pattern），但所有共存代码必须在同一个 PR 内消除
- **NFR-Test-Coverage**: `Pair` 模块单测 ≥ 95%；所有现有 pair-相关 integration test 全部迁移到 `Pair`
- **NFR-Backwards-Compat**: 现有 DB 数据 0 丢失；前端 URL 中带 `?pair=BTC/USDT` 仍能匹配
- **NFR-Performance**: `Pair` 实例化开销 < 5μs（frozen dataclass 自带），不应成为热路径瓶颈

## Out of Scope

- ❌ Spot ↔ perp 之间的资金搬运（OKX unified account 自带）
- ❌ 跨 exchange 同一 base/quote 的歧义消解（现在仅单一 exchange，未来 spec）
- ❌ `Order.pair` 之外的其他 model 字段（如 `MarketData.pair`）首轮先不动，第二阶段统一

## Phased Delivery

依据 D7：Phase 3 拆 3a/3b/3c。每个 phase 都可独立部署、回滚。估时为单人粗估，含测试。

| Phase | Scope | 估时 | 风险 |
|---|---|---|---|
| **0** ✅ | band-aid: `LiveExchange._canonical_pair` 已合并（commit d05a0bf） | — | — |
| **1** | FR-001 ~ 010：`pair.py` 模块 + 单测；0 调用方接入 | 0.5 day | 低 — 类型 API 设计错只需返工模块 |
| **2** | FR-100 ~ 104：config 升级；适配 `[scheduler].pairs` 新形式 + 旧 list[str] 兼容；不动 nodes | 1 day | 低 — 用户配置文件改一次，单元测试覆盖 |
| **3a** | adapter 层：`pair_adapter.py` 提供 `Pair ↔ str` 转换 helper；nodes 仍用 str 但走 helper（消除散点 split） | 0.5 day | 低 — 纯重构，无行为变化 |
| **3b** | `nodes/verdict.py` + `nodes/execution.py` 切 Pair；其他 nodes 仍 str | 1 day | 中 — 这两个节点是 trading 路径核心 |
| **3c** | 剩余 nodes + state schema bump + checkpoint 兼容；撤掉 adapter | 1 day | 高 — LangGraph state 全量切换；需要 checkpoint migration 测试 |
| **4** | FR-400 ~ 405：alembic 加列；JournalStore 双写 | 0.5 day | 低 — 只加列不改值 |
| **5** | FR-500 ~ 506：API 加字段；前端 `<PairBadge>` 组件 + 两个视图接入 | 1 day | 低 — 范围严格控制，其他视图不动 |
| **撤回 0** | Phase 3c 完成后删除 `LiveExchange._canonical_pair` | 0.1 day | 低 — 由 FR-304 触发 |

总估时：~5.6 day（不含 review / 真盘 sandbox 测试）。**关键路径**：1 → 2 → 3a → 3b → 3c → 撤回 0。Phase 4/5 可与 3 并行。

## Success Criteria

- 全代码搜索 `BTC/USDT:USDT` 0 命中（除 `Pair.to_ccxt()` 实现）
- 全代码搜索 `\.split\("/"\)` 在 pair 上下文 0 命中
- `LiveExchange._canonical_pair` 删除
- 一个 OKX swap 用户的完整 cycle (decide → place_order → fill → snapshot → journal) 端到端通过
- pytest 全绿，新增 `Pair` 模块覆盖率 ≥ 95%

## Resolved Questions (2026-04-30 brainstorm)

1. ~~`Pair.canonical()` 用 `"BTC/USDT-swap-USDT"` 还是 `"BTC/USDT:USDT"`？~~
   **决议**：用 `ccxt unified symbol` 作为 canonical，即 `"BTC/USDT"` (spot) / `"BTC/USDT:USDT"` (perp swap)。理由：跟 D1 一致，ccxt 是真实来源；spot 没有冗余后缀。
2. ~~前端 URL `?pair=` 用 canonical 还是 display？~~
   **决议**：URL 用 canonical（即 ccxt symbol），后端解析；display 仅用于渲染。`?pair=BTC/USDT` 老链接仍能匹配存量 spot 数据。
3. ~~`MarketData.pair` `OnchainData.pair` 这些"数据领域"的 pair 是否同步升级？~~
   **决议**：本 spec 只动"交易领域"（Order / Position / verdict / risk / journal）。数据领域 pair 留作单独 spec 跟踪。

## Open Questions

无（brainstorm 后全部 resolve）。如果实施过程中冒出新问题，通过 spec amendment 追加。
