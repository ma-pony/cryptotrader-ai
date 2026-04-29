# Feature Specification: WS 实时行情 + 自适应轮询

**Feature Branch**: `009-ws-realtime-adaptive-polling`
**Created**: 2026-04-17
**Status**: Draft

---

## 背景与动机

当前 CryptoTrader-AI 前端所有数据均通过 React Query 固定 10 秒轮询获取。Dashboard 的总权益、持仓盈亏、价格等核心指标存在 10 秒延迟；MarketView 的资金费率、未平仓合约等边栏数据每 30 秒才刷新一次。在加密货币市场波动剧烈的时段，这一延迟会导致交易者无法及时感知价格风险，错失最优决策时机。

本 Feature 通过引入 **Binance WebSocket 实时价格流**，将价格类数据的延迟从 10 秒降低至毫秒级；同时通过**自适应轮询**机制，在市场非活跃时段自动降低轮询频率以节省带宽，在活跃时段保持高频刷新。WS 断线时自动降级为 REST 轮询，确保可用性不受影响。

---

## 用户场景与验收测试 *(必填)*

### User Story 1 — Dashboard 核心指标实时刷新 (Priority: P0)

作为交易员，我希望 Dashboard 上的「总权益」「24h PnL」「持仓未实现盈亏」等指标能在价格变动后秒级内更新，而不是等待 10 秒轮询，从而能及时感知浮亏扩大或止盈机会。

**Why this priority**: Dashboard 指标卡是系统最高频查看的页面，价格延迟直接影响风险感知。P0 意味着该功能是本 Feature 的核心交付物，其他功能均围绕它展开。

**Independent Test**: 打开 Dashboard，在 Binance 测试网或模拟数据源中触发 BTC/USDT 价格变动，验证「总权益」MetricCard 在 2 秒内数值更新，而非等待 10 秒。

**Acceptance Scenarios**:
1. **Given** Dashboard 已加载，WebSocket 连接成功，**When** Binance WS 推送 BTC/USDT 最新价格，**Then** `MetricCardsRow` 中受该价格影响的指标卡（总权益、持仓 PnL）在 2 秒内更新显示新数值，且无页面闪烁。
2. **Given** WebSocket 连接中断（网络波动），**When** 断线超过 3 秒，**Then** 前端自动降级为每 10 秒一次的 REST 轮询，数据继续刷新，不显示空白或报错。
3. **Given** WebSocket 重连成功，**When** WS 恢复推送，**Then** 前端自动切回 WS 模式，REST 轮询停止，连接状态指示器回到「已连接」状态。

### User Story 2 — MarketView 侧边栏实时更新 (Priority: P1)

作为交易员，我希望 MarketView 页面的资金费率（Funding Rate）和未平仓合约（Open Interest）能以更高频率刷新，以便我在资金费率结算前后做出对冲决策。

**Why this priority**: 资金费率每 8 小时结算一次，临近结算时实时数据至关重要，但其他时段刷新频率可以降低，属于 P1 优化。

**Independent Test**: 打开 MarketView，选择 Binance / BTC/USDT，模拟资金费率数据变动，验证侧边栏 Funding Rate 卡片数值在 5 秒内更新（WS 模式）；断开 WS 后验证降级为 30 秒轮询（非活跃）或 10 秒轮询（活跃时段）。

**Acceptance Scenarios**:
1. **Given** MarketView 已加载并订阅 BTC/USDT，**When** Binance WS 推送新的资金费率数据，**Then** `MarketSidebar` 中的资金费率卡片在 3 秒内显示更新后的数值。
2. **Given** 当前时间处于非活跃时段（波动率低），**When** WebSocket 不可用，**Then** 系统自动使用 60 秒轮询间隔，而非 30 秒。
3. **Given** 当前时间处于活跃时段（波动率高或临近资金费率结算），**When** WebSocket 不可用，**Then** 系统自动使用 10 秒轮询间隔。

### User Story 3 — WS 连接状态可见性 (Priority: P1)

作为交易员，我希望能在界面上清楚地看到当前数据源是「实时 WS」还是「轮询降级」模式，从而对数据新鲜度有正确预期。

**Why this priority**: 数据透明度对于交易决策至关重要。当系统降级时，用户需要知道数据可能有延迟。

**Independent Test**: 故意断开网络或阻断 WebSocket 端口，验证 Dashboard 或全局导航栏出现「数据延迟中（轮询模式）」的状态指示，并在恢复后消失。

**Acceptance Scenarios**:
1. **Given** WebSocket 连接正常，**When** 查看页面，**Then** 连接状态指示器显示绿色「实时」标识（或无特殊提示，默认静默）。
2. **Given** WebSocket 断线降级为轮询，**When** 查看页面，**Then** 指示器显示黄色「轮询模式」标识，包含当前轮询间隔。
3. **Given** WebSocket 重连中（指数退避），**When** 查看页面，**Then** 指示器显示「重连中…」动态状态。

### User Story 4 — 自适应轮询节省非活跃时段资源 (Priority: P2)

作为系统运维，我希望在市场非活跃时段（如深夜低波动率窗口）自动降低轮询频率，减少不必要的 API 请求，降低后端和网络负载。

**Why this priority**: P2 是锦上添花的优化，不影响核心功能，但可降低长期运营成本和 API Rate Limit 压力。

**Independent Test**: 通过 mock 将系统时间切换到非活跃时段，或注入低波动率信号，验证 `useAdaptivePolling` hook 将 `refetchInterval` 从 10 秒调整为 60 秒。

**Acceptance Scenarios**:
1. **Given** 系统检测到当前 24 小时价格波动率低于阈值（如 < 1%），**When** WS 已断线降级为轮询，**Then** 轮询间隔自动调整为 60 秒。
2. **Given** 系统检测到波动率超过阈值或临近整点（UTC 0/8/16 时资金费率结算），**When** WS 已断线，**Then** 轮询间隔自动收紧至 10 秒。
3. **Given** WebSocket 连接正常，**When** 自适应轮询逻辑运行，**Then** 轮询完全停止（WS 优先），不发送任何 REST 轮询请求。

### 边界条件

- **WebSocket 握手失败**（如 Binance 限流、网络防火墙屏蔽 wss://）：系统应立即降级为 REST 轮询，不阻塞页面渲染，错误信息仅记录到 console，不弹出报错 toast。
- **多标签页/多订阅冲突**：同一浏览器多个标签页各自独立维护 WS 连接，Context 不跨 tab 共享，各自独立降级。
- **交易对切换**：在 MarketView 切换交易对时，旧订阅必须先 unsubscribe，再建立新订阅，避免消息混流。
- **开发模式双重挂载**：组件卸载时必须正确关闭 WS 连接，避免开发模式下产生多余连接。
- **WebSocket 消息速率过高**（如 BTC/USDT ticker 每 100ms 一条）：前端应做节流（throttle），避免每条消息都触发 UI 重渲染，节流窗口 200ms。
- **数据类型不一致**：Binance WS ticker 返回字符串格式价格（`"43250.00"`），需在 Context 层解析为 `number` 后再分发。
- **页面不可见时（`document.visibilityState === 'hidden'`）**：暂停 WS 消息处理，降低 CPU 占用；页面恢复可见时重新激活订阅。

---

## 需求规格 *(必填)*

### 功能需求

**WS 连接管理**

- **FR-001**: 系统应在 `MarketDataProvider` 中建立并维护到 Binance 公共 WebSocket 行情流的连接，支持 ticker 流订阅。
- **FR-002**: 每个交易对订阅通过独立的 stream 频道进行，Context 维护一个订阅计数器，当某交易对订阅数降为 0 时自动发送 `UNSUBSCRIBE` 消息，释放带宽。
- **FR-003**: `useMarketDataWS` hook 应提供 `subscribe(pair: string)`、`unsubscribe(pair: string)`、`connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'degraded'` 三个接口。
- **FR-004**: WebSocket 消息到达后，系统更新 Context 内部的 `priceMap: Record<string, TickerData>`，所有订阅同一交易对的组件均自动获得最新数据，无需额外请求。

**WS-first + REST Fallback**

- **FR-005**: 当 WebSocket 连接失败或断开时，系统应在 3 秒内自动触发 REST 轮询降级，继续通过 `/api/market/{pair}` 端点获取行情数据。
- **FR-006**: 降级后系统以指数退避策略尝试重连 WebSocket（初始 1 秒，最大 30 秒，最多重试 10 次）；重连成功后自动停止 REST 轮询，恢复 WS 模式。
- **FR-007**: 在 WS 连接正常时，禁止同时触发 REST 轮询，避免双重请求浪费网络资源。

**自适应轮询**

- **FR-008**: `useAdaptivePolling` hook 应根据当前市场活跃度返回动态的 `refetchInterval`：活跃时段返回 `10_000`（10 秒），非活跃时段返回 `60_000`（60 秒）。
- **FR-009**: 活跃时段判断依据包含（任意满足一项）：（1）最近 15 分钟价格波动率超过可配置阈值（默认 1%）；（2）当前 UTC 时间处于资金费率结算窗口前后 30 分钟（即 UTC 7:30-8:30、15:30-16:30、23:30-0:30）。
- **FR-010**: 自适应轮询策略仅在 WS 降级模式下生效；WS 正常时 `refetchInterval` 返回 `false`（React Query 不轮询）。

**Dashboard 集成**

- **FR-011**: `usePortfolioSnapshot` hook 应整合 WS 价格数据：当 WS 连接正常时，持仓的 `unrealized_pnl` 和 `unrealized_pnl_pct` 字段应基于 WS 最新价格实时计算，而非等待 REST 轮询刷新。
- **FR-012**: `MetricCardsRow` 中的「总权益」卡片应在 WS 价格更新时实时重算总权益（持仓市值 + 现金），不依赖 REST 轮询。
- **FR-013**: `PositionsTable` 中每行持仓的未实现 PnL 数值应在 WS 价格更新时实时更新，不等待轮询。

**MarketView 集成**

- **FR-014**: `MarketSidebar` 组件应使用 `useMarketDataWS` 订阅当前选中的交易对，在 WS 连接正常时通过 WS 数据更新资金费率和未平仓合约显示。
- **FR-015**: MarketView 切换交易对时，旧交易对的 WS 订阅应在组件 unmount（或 pair 变更）时立即取消，新交易对订阅应在 mount 后立即建立。

**连接状态 UI**

- **FR-016**: 全局布局（侧边栏或顶部导航栏）应包含一个 `WSStatusIndicator` 组件，在 WS 降级时显示黄色告警标识和当前轮询间隔文字，在 WS 正常时静默（不显示任何特殊标识）。
- **FR-017**: 连接状态变化（connected → disconnected、disconnected → connected）应在状态稳定 2 秒后才触发 UI 更新，避免短暂抖动导致状态指示器频繁闪烁。

**数据处理**

- **FR-018**: WS ticker 消息中的价格字段（字符串类型）应在 Context 层统一解析为 `number`，下游组件不承担类型转换责任。
- **FR-019**: WS 消息处理应有 200ms 节流，同一交易对在 200ms 窗口内只触发一次 React 状态更新。
- **FR-020**: 当 `document.visibilityState === 'hidden'` 时，暂停 Context 内的消息分发（不影响 WS 连接本身）；页面重新可见时恢复分发，并立即触发一次 REST 请求同步最新快照。

### 非功能需求

- **NFR-001（性能）**: WS 价格更新触发的 React 重渲染范围应限制在订阅了对应交易对的叶子组件，不引起整个 Dashboard 重渲染。通过 React Context 按交易对分片或 Zustand selector 实现。
- **NFR-002（包体积）**: 本 Feature 不引入新的第三方 WebSocket 库，使用浏览器原生 `WebSocket` API 实现。新增 JS bundle 增量（gzip 后）不超过 5 KB。
- **NFR-003（安全）**: 前端直连 Binance 公共 WS 数据流（只读行情，无需 API Key）；用户私密 API Key 不通过 WS 传输，不暴露在 WS URL 中。
- **NFR-004（可测试性）**: `useMarketDataWS` 和 `useAdaptivePolling` 应可在测试环境中通过注入 mock WebSocket 实例进行单元测试，不依赖真实网络连接。
- **NFR-005（可观测性）**: WS 连接状态变化、降级事件、重连尝试次数应通过 `console.info` 记录，格式为 `[WSMarketData] <event>`，便于生产环境调试。

### 关键实体

- **`MarketDataProvider`**: React Context Provider，封装 WebSocket 连接生命周期、订阅管理、降级逻辑，向子树暴露 `priceMap` 和 `connectionStatus`。
- **`useMarketDataWS`**: 消费 `MarketDataProvider` Context 的自定义 hook，提供 `subscribe`、`unsubscribe`、`getPrice(pair)`、`connectionStatus` 接口。
- **`useAdaptivePolling`**: 独立 hook，基于波动率信号和时间窗口计算当前应使用的 `refetchInterval`，不依赖 Context。
- **`WSStatusIndicator`**: 纯展示组件，接收 `connectionStatus` prop，渲染连接状态标识（绿色/黄色/动态）。
- **`TickerData`**: WS ticker 消息的类型定义，包含 `pair: string`、`price: number`、`priceChangePercent: number`、`volume24h: number`、`ts: number`（毫秒时间戳）。
- **`priceMap`**: `Record<string, TickerData>`，由 `MarketDataProvider` 维护，键为标准化交易对名称（如 `"BTCUSDT"`）。
- **活跃时段（Active Window）**: 系统内部概念，由波动率阈值或资金费率结算窗口定义的高频刷新时段。

---

## 成功标准 *(必填)*

### 可量化的验收指标

- **SC-001**: 在 WS 连接正常时，Dashboard MetricCard「总权益」的数据更新延迟从当前 ≤10 秒降低至 ≤2 秒（基于 Binance WS ticker 推送延迟 + 前端节流 200ms）。
- **SC-002**: WS 断线后 3 秒内完成降级，REST 轮询自动启动，期间无数据中断（最后一次 WS 数据保持显示，不出现空白）。
- **SC-003**: WS 重连成功后，REST 轮询在 5 秒内停止，不出现 WS + REST 双重请求并发的情况。
- **SC-004**: 在 WS 正常时，MarketView 切换交易对后，旧订阅在 1 秒内取消，新订阅在 1 秒内建立并收到第一条数据。
- **SC-005**: 非活跃时段 + WS 降级时，REST 轮询间隔为 60 秒，相比默认 10 秒减少 83% 的 API 请求量。
- **SC-006**: 新增 bundle 增量（gzip 后）不超过 5 KB，使用 `vite-bundle-visualizer` 或 `rollup-plugin-bundle-size` 验证。
- **SC-007**: `useMarketDataWS` 和 `useAdaptivePolling` 的单元测试覆盖率不低于 80%（基于 mock WebSocket）。
- **SC-008**: `WSStatusIndicator` 在降级状态下的可访问性（ARIA label）通过 `axe-core` 验证，无 WCAG 2.1 AA 级违规。

---

## 假设

- Binance 公共 WebSocket 数据流（`wss://stream.binance.com:9443/ws/<streamName>`）在用户浏览器环境中可直接访问，无需后端代理（用户网络未屏蔽 Binance）。
- 前端直连 Binance WS 只订阅公共只读行情（ticker），不涉及账户鉴权，不需要 API Key。
- OKX 交易所不在本 Feature 的 WS 实时化范围内；OKX 数据仍沿用现有 REST 轮询方案（MarketSidebar 已有 `exchange` 切换逻辑，WS 仅对 `exchange === 'binance'` 生效）。
- 「波动率」的计算来源为最近 15 分钟内 WS ticker 推送的 `priceChangePercent`（24h 变化率）；如 WS 尚未连接或尚未收到足够数据，默认视为「非活跃时段」。
- 加密货币市场 24/7 运行，「活跃时段」不依赖传统证券市场开盘时段，仅依赖波动率信号和资金费率结算窗口。
- `lightweight-charts` 现有的 `EquityChart` 组件基于历史权益曲线（`/api/portfolio/equity-curve`），不在本 Feature 的实时化范围内；K 线实时更新指的是 MarketView 中 TradingView Widget 本身已自带实时数据，不需要前端额外处理。
- React Query 的 `queryClient` 缓存层保持不变；WS 数据通过独立的 `MarketDataProvider` Context 分发，不写入 React Query 缓存（避免 staleTime 逻辑冲突）。
- 后端 `/api/portfolio/snapshot` 端点返回的持仓数据包含 `avg_price` 和 `size` 字段，前端可基于 WS 最新价格自行计算 `unrealized_pnl = (latest_price - avg_price) * size`，无需后端实时推送持仓盈亏。
- 本 Feature 不修改后端任何代码，所有改动均在 `web/` 目录内进行。

---

## 范围外

- 后端 WebSocket 代理（复杂度高，前端直连即可满足需求）
- OKX 实时 WebSocket 订阅（OKX WS 协议与 Binance 差异较大，留待后续 spec）
- 订单簿（Order Book）深度数据的实时推送
- 用户账户级别的私有 WebSocket 流（持仓变更推送等，涉及 API Key 安全）
- K 线图历史数据的实时增量更新（TradingView Widget 自行处理）
- 服务端推送事件（SSE）的扩展（`/api/chat/stream` 已有独立 SSE 方案）

---

## 依赖

- **前置条件**: Spec 001（Frontend Rewrite）已完成，`web/` 目录 React 项目结构就绪
- **外部依赖**: Binance WebSocket API（公共市场数据流，无 SLA 保证，故需降级机制）
- **无新增 npm 依赖**（使用浏览器原生 `WebSocket` API）
