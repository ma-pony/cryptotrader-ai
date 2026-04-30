# Feature Specification: 引入 Pair 值对象统一交易对类型语义

**Feature Branch**: `013-pair-value-object`
**Created**: 2026-04-30
**Status**: Draft
**Input**: 现状是项目把 `"BTC/USDT"` 当作纯字符串到处传，但实际承载的是带"市场类型"语义的复合概念（spot vs perp swap vs futures），ccxt 边界返回的 `BTC/USDT:USDT` 与项目内部 `BTC/USDT` 不匹配，导致 read 侧 lookup 静默 miss、write 侧可能下错市场。需要引入 `Pair` 值对象统一表达。

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
- **FR-001**: 新增 `src/cryptotrader/pair.py` 模块定义 `Pair` 为 `@dataclass(frozen=True)`：
  ```python
  @dataclass(frozen=True)
  class Pair:
      base: str            # "BTC"
      quote: str           # "USDT"
      market_type: Literal["spot", "swap", "futures"] = "spot"
      settle: str | None = None  # required when market_type != "spot"
  ```
- **FR-002**: `Pair.parse(s: str) -> Pair` 解析项目 canonical 字符串形式
- **FR-003**: `Pair.from_ccxt(exchange, symbol: str) -> Pair` 通过 `exchange.market(symbol)` 元信息构造
- **FR-004**: `Pair.to_ccxt() -> str` 输出 ccxt 统一 symbol（spot 返回 `BASE/QUOTE`，swap 返回 `BASE/QUOTE:SETTLE`）
- **FR-005**: `Pair.display() -> str` 返回 AI/UI 友好形式（`"BTC/USDT (perp)"` 或 `"BTC/USDT"`）
- **FR-006**: `Pair.canonical() -> str` DB / state / 配置的标准序列化形式（候选 `"BTC/USDT-swap-USDT"`）
- **FR-007**: `Pair.__hash__` / `Pair.__eq__` 由 frozen dataclass 自动生成，可作 dict key
- **FR-008**: 抛 `UnknownMarketTypeError(symbol, available_types)` 而不是默认 fallback
- **FR-009**: 单元测试覆盖 round-trip：parse → to_ccxt → from_ccxt → 同一对象
- **FR-010**: 单元测试覆盖 ccxt 各 exchange 的 symbol 形态（OKX swap、Binance USDT-M、Binance COIN-M、Bybit、dYdX）

### Config (FR-100 ~ 109)
- **FR-100**: `[exchanges.<id>]` 新增 `market_type = "spot" | "swap" | "futures"`，默认 `"spot"`
- **FR-101**: `[scheduler].pairs` 仍接受字符串（向后兼容），但内部解析为 `Pair`，`market_type` 取自所属 exchange
- **FR-102**: 配置加载时 `Pair.parse()` 一次，缓存为 `list[Pair]`
- **FR-103**: 启动时打印一行结构化日志 `pair_init: spot=[..] swap=[..]` 供运维核对

### State / Nodes (FR-200 ~ 219)
- **FR-200**: `state["metadata"]["pair"]` 类型从 `str` 改为 `Pair`
- **FR-201**: `nodes/data.py` `nodes/agents.py` `nodes/debate.py` `nodes/verdict.py` `nodes/execution.py` `nodes/journal.py` 接收 `Pair`，内部不再做 split / suffix 操作
- **FR-202**: AI prompt 模板使用 `pair.display()`
- **FR-203**: structlog 字段 `pair=pair.canonical()` 保持 grep-able

### Exchange Boundary (FR-300 ~ 319)
- **FR-300**: `LiveExchange.get_positions() -> dict[Pair, dict]`（key 改 `Pair`）
- **FR-301**: `LiveExchange.place_order(order: Order)` 内部调用 `pair.to_ccxt()` 转 ccxt symbol
- **FR-302**: `Order.pair` 类型 `Pair`
- **FR-303**: `PaperExchange` 同步签名变更
- **FR-304**: 撤回 `LiveExchange._canonical_pair` band-aid（不再需要）

### Storage (FR-400 ~ 419)
- **FR-400**: `portfolios.pair` 列升级，alembic migration 把 `"BTC/USDT"` → `"BTC/USDT-spot"`（无类型信息历史一律视作 spot）
- **FR-401**: `decision_commits.pair` 同上
- **FR-402**: `portfolio_snapshots`：account_id 仍是 `default`，但 `position` 关联表（如有）同步
- **FR-403**: `JournalStore._serialize` / `_deserialize` 在 `Pair` ↔ `Pair.canonical()` 之间转换
- **FR-404**: 提供 `arena db migrate` 命令一键升级

### API / Frontend (FR-500 ~ 519)
- **FR-500**: `/api/portfolio/snapshot` 响应里 `positions[].pair` 序列化为 `pair.canonical()`，新增 `pair_display` 字段（人类可读）
- **FR-501**: `/api/decisions/{id}` 同上
- **FR-502**: 前端 `Pair` TypeScript 类型镜像后端结构
- **FR-503**: 前端组件 `<PairBadge pair={...}/>` 统一渲染 + market type 徽章
- **FR-504**: 现有 `pair.startsWith()` `pair.split()` 之类的前端散点全部改用 `Pair` 类的 helper

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

- **Phase 0** (本会话已完成 band-aid)：`LiveExchange._canonical_pair` 通过 `exchange.market()` 元信息把 ccxt perp symbol 标准化回 spot 形式，治标修复 close-on-flat。
- **Phase 1**（FR-001 ~ 010）：`Pair` 模块 + 完整单测，0 调用方接入。
- **Phase 2**（FR-100 ~ 109 + FR-300 ~ 319）：config + Exchange 边界接入，nodes 层暂时通过 adapter 拿 `pair.canonical()` 字符串。
- **Phase 3**（FR-200 ~ 219）：state/nodes 全面 `Pair` 化，撤掉 adapter。
- **Phase 4**（FR-400 ~ 419）：DB migration + 历史数据升级。
- **Phase 5**（FR-500 ~ 519）：API + frontend 升级。

## Success Criteria

- 全代码搜索 `BTC/USDT:USDT` 0 命中（除 `Pair.to_ccxt()` 实现）
- 全代码搜索 `\.split\("/"\)` 在 pair 上下文 0 命中
- `LiveExchange._canonical_pair` 删除
- 一个 OKX swap 用户的完整 cycle (decide → place_order → fill → snapshot → journal) 端到端通过
- pytest 全绿，新增 `Pair` 模块覆盖率 ≥ 95%

## Open Questions

1. `Pair.canonical()` 用 `"BTC/USDT-swap-USDT"` 还是 `"BTC/USDT:USDT"`？前者无歧义但 DB 索引可能更长；后者跟 ccxt 一致但 spot 形式带 `-spot` 后缀有点啰嗦
2. 前端 URL `?pair=` 用 canonical 还是 display？涉及分享链接的可读性
3. `MarketData.pair` `OnchainData.pair` 这些"数据领域"的 pair 是否同步升级？还是仅"交易领域"升级？
