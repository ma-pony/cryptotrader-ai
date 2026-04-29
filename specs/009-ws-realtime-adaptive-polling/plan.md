# 技术实施方案：WS 实时行情 + 自适应轮询

**Feature Branch**: `009-ws-realtime-adaptive-polling`
**依据 Spec 版本**: 2026-04-17 Draft
**范围**: 纯前端（`web/` 目录），不修改任何后端代码

---

## 技术上下文

### 现有架构快照

| 层次 | 现状 | 问题 |
|------|------|------|
| `usePortfolioSnapshot` | `refetchInterval: 10_000`，固定 10 秒轮询 `/api/portfolio/snapshot` | Dashboard 价格延迟最高 10 秒 |
| `MarketSidebar` | 位于 `web/src/pages/market/components/market-sidebar.tsx`，内部 `useQuery` + `refetchInterval: 30_000`，30 秒轮询 `/api/market/{pair}` | 资金费率临近结算时实时性不足 |
| `MarketPage` | 位于 `web/src/pages/market/index.tsx`，`pair` 由 `useState(DEFAULT_PAIR)` 固定为 `'BTC/USDT'`，暂无切换 UI；`exchange` 可通过 `ExchangeSelector` 切换 | pair 切换逻辑需在实现 FR-015 时补充 |
| `main.tsx` Provider 栈 | `QueryClient → BrowserRouter → ThemeProvider → I18nProvider` | 无全局 WebSocket Context |
| `App.tsx` 路由 | `AppShell` 包裹所有页面，`TopBar` 与 `Sidebar` 为固定布局组件 | `WSStatusIndicator` 需挂载到 `TopBar` |
| `TopBar` | 仅含 `ApiKeyBadge`、主题切换、语言切换，无连接状态区域 | 需在 `<ApiKeyBadge />` 右侧插入 `WSStatusIndicator` |
| 类型系统 | `Position`（`api.schema.ts`）包含 `avg_price`、`size`、`unrealized_pnl`、`unrealized_pnl_pct` | 可在前端基于 WS 价格实时重算 PnL |
| 测试框架 | Vitest + Testing Library，无端到端依赖 | 需为 WS mock 提供注入点 |

### Binance 公共 WebSocket 协议

- **URL 格式**: `wss://stream.binance.com:9443/ws/<streamName>`
- **单流订阅**: `wss://stream.binance.com:9443/ws/btcusdt@ticker`
- **组合流**: `wss://stream.binance.com:9443/stream?streams=btcusdt@ticker/ethusdt@ticker`（推荐，减少连接数）
- **ticker 关键字段**（字符串类型，需解析为 number）：
  - `s`：交易对（如 `"BTCUSDT"`）
  - `c`：最新价格（close price）
  - `P`：24h 价格变化百分比
  - `v`：24h 成交量（quote asset）
  - `E`：事件时间（毫秒时间戳）
- **订阅/取消消息格式**：发送 JSON `{"method":"SUBSCRIBE","params":["btcusdt@ticker"],"id":1}`
- **心跳**：Binance WS 服务端每 3 分钟发送 ping，客户端需回复 pong（浏览器原生 WebSocket 自动处理 ping/pong）

---

## 架构决策

### ADR-001：使用 React Context（非 Zustand）管理 WS 连接

**决策**：`MarketDataProvider` 使用 `createContext` + `useReducer`，而非将 `priceMap` 放入 Zustand store。

**理由**：
- WS 连接属于"副作用资源"，生命周期与 Provider 的 mount/unmount 强绑定，`useEffect` 清理函数天然适合管理
- Zustand store 为全局单例，无法随组件树局部卸载；多标签页场景下各自独立的 WS 连接更符合 spec 的隔离要求
- `priceMap` 更新频率极高（200ms 节流后仍可达 5 次/秒），写入 Zustand 会触发全局 selector 重算；React Context 配合 `useMemo` / `useCallback` 可将重渲染范围限制在消费者组件

**NFR-001 实现**：Context 将 `subscribe`/`unsubscribe` 函数与 `getPrice(pair)` 分开暴露——`getPrice` 使用 `useCallback(pair => priceMapRef.current[pair])` 从 `ref` 读取，调用方不需要将 `priceMap` 放入自身 state，从而避免整体 re-render。对于需要实时响应价格的叶子组件（`MetricCardsRow`、`PositionsTable` 行），改用 `useSyncExternalStore` 精确订阅单个交易对的价格更新。

### ADR-002：WS 降级策略使用"3 秒稳定计时器"

**决策**：WebSocket `onclose` / `onerror` 触发后，启动 3 秒倒计时，超时后才切换 `connectionStatus → 'degraded'` 并启动 REST 轮询（FR-005）。`onopen` 触发后，再次等待 2 秒稳定后切换为 `'connected'`（FR-017）。

**理由**：网络短暂抖动（< 3 秒）会导致 WS 瞬断再连，无需触发降级逻辑；2 秒稳定延迟防止状态指示器频繁闪烁。

### ADR-003：订阅计数器 + 单一组合流

**决策**：`MarketDataProvider` 内部维护 `subscriptions: Map<string, number>`（key=标准化交易对，value=订阅者数量）。当 map 变更时，重新构建组合流 URL 或发送 `SUBSCRIBE`/`UNSUBSCRIBE` 消息（FR-002）。

**理由**：避免同一交易对被多个组件重复订阅产生多余 WS 消息；组合流模式（`/stream?streams=...`）通过单一 WS 连接订阅多交易对，减少握手开销。

**注意**：由于 `MarketPage` 当前使用 `const [pair] = useState(DEFAULT_PAIR)` 固定交易对，FR-015（pair 切换时先 unsubscribe 再 subscribe）将在 `useMarketDataWS` 的 `useEffect` 依赖数组 `[pair]` 中自动处理——即使 MarketPage 未提供切换 UI，hook 的清理机制已满足 spec 要求。

### ADR-004：自适应轮询独立于 WS Context

**决策**：`useAdaptivePolling` 是纯函数 hook，不依赖 `MarketDataProvider`。它接收 `{ wsStatus, priceChangePercent, enabledAt }` 参数，返回 `refetchInterval: number | false`（FR-008~010）。

**理由**：解耦 WS 状态管理与轮询策略，便于独立单元测试；调用方（`usePortfolioSnapshot` 改造版、`MarketSidebar`）自行决定如何获取输入参数。

**TopBar 特例**：`TopBar` 中的 `WSStatusIndicator` 仅需要 `connectionStatus`，不需要自适应轮询间隔——TopBar 没有交易对上下文，无法计算 `priceChangePercent`。`refetchInterval` prop 改为从 `useMarketDataWS()`（无 pair）的 `connectionStatus` 派生后，`WSStatusIndicator` 内部根据状态显示固定文字（"轮询模式"），具体间隔数字由 Dashboard 或 MarketSidebar 各自的 `useAdaptivePolling` 提供。

### ADR-005：PnL 实时重算在 hook 层完成

**决策**：`usePortfolioSnapshot` 保持其 React Query 调用不变，但额外接受 `priceMap` 作为输入（通过 `useMarketDataWS` 获取），在 `select` 选项中用 WS 价格实时重算 `unrealized_pnl` 和 `equity`（FR-011、FR-012）。

**理由**：spec 假设后端快照包含 `avg_price` + `size`，前端可自行计算 `unrealized_pnl = (latestPrice - avg_price) * size`；将计算逻辑放在 `select` 中，React Query 缓存的原始数据不被污染，WS 断线后降级回轮询数据时不会闪烁。

---

## 文件结构（新增/修改）

```
web/src/
├── contexts/                              ← 新目录
│   └── market-data/
│       ├── market-data-context.ts         ← 新增：Context 定义 + TickerData 类型
│       ├── market-data-provider.tsx       ← 新增：Provider 实现（WS 连接生命周期）
│       └── index.ts                       ← 新增：公开 API 统一导出
│
├── hooks/
│   ├── use-market-data-ws.ts              ← 新增：消费 MarketDataContext 的 hook
│   ├── use-adaptive-polling.ts            ← 新增：自适应轮询 interval 计算 hook
│   └── use-portfolio-snapshot.ts          ← 修改：整合 WS 价格实时重算 PnL
│
├── components/
│   ├── layout/
│   │   └── top-bar.tsx                    ← 修改：插入 WSStatusIndicator
│   └── ws-status-indicator/
│       ├── ws-status-indicator.tsx        ← 新增：连接状态展示组件
│       └── index.ts                       ← 新增：导出
│
├── pages/
│   ├── dashboard/
│   │   ├── components/
│   │   │   ├── metric-cards-row.tsx       ← 修改：从 WS 读取实时价格触发重渲染
│   │   │   └── positions-table.tsx        ← 修改：实时 PnL 显示
│   │   └── index.tsx                      ← 修改：传递 WS 数据给子组件
│   └── market/                            ← 注意：实际路径为 web/src/pages/market/（非 market-view）
│       ├── index.tsx                      ← 修改：添加 pair 切换 UI 支持 FR-015（可选）
│       └── components/
│           └── market-sidebar.tsx         ← 修改：集成 WS 订阅 + 自适应轮询
│
├── main.tsx                               ← 修改：注入 MarketDataProvider 到 Provider 栈
│
└── __tests__/                             ← 新增目录
    hooks/
    ├── use-market-data-ws.test.ts          ← 新增：WS hook 单元测试
    ├── use-adaptive-polling.test.ts        ← 新增：自适应轮询单元测试
    └── market-data-provider.test.tsx       ← 新增：Provider 集成测试
```

---

## 前端设计

### 1. `TickerData` 类型定义

```typescript
// web/src/contexts/market-data/market-data-context.ts
export interface TickerData {
  pair: string;           // 标准化交易对，如 "BTCUSDT"
  price: number;          // 最新成交价（已从字符串解析）
  priceChangePercent: number; // 24h 涨跌幅（已从字符串解析）
  volume24h: number;      // 24h 成交量
  ts: number;             // 事件时间（毫秒）
}

// FR-003 要求的四种状态（connecting / connected / disconnected / degraded）
// + 'reconnecting' 扩展状态（spec FR-003 未定义，此处为状态机必要扩展）：
//   原因：状态机需要区分"已断开且不再重试（disconnected）"和"正在重连中（reconnecting）"
//   两个语义不同的状态。若合并为单一 disconnected，UI 无法区分"永久离线"与"短暂重连"，
//   导致 WSStatusIndicator 无法正确显示重连进度动画（见 ADR-002 重连逻辑）。
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'degraded' | 'reconnecting';

export interface MarketDataContextValue {
  connectionStatus: ConnectionStatus;
  subscribe: (pair: string) => void;
  unsubscribe: (pair: string) => void;
  getPrice: (pair: string) => TickerData | undefined;
  // 用于 useSyncExternalStore：订阅特定交易对的价格变更通知
  subscribeToPrice: (pair: string, callback: () => void) => () => void;
}
```

### 2. `MarketDataProvider` 内部状态机

```
初始状态: DISCONNECTED
  │
  ▼ mount / 有订阅者
CONNECTING ──────── onerror/onclose ──► DISCONNECTED（3s 计时）
  │                                           │
  ▼ onopen                                    ▼ 3s 超时
CONNECTED（2s 延迟后 UI 更新）          DEGRADED（启动 REST 降级轮询）
  │                                           │
  └──── onclose/onerror ──► RECONNECTING ──── ▼ 最多重试 10 次
                             指数退避：          重试失败 → 维持 DEGRADED
                             1s→2s→4s→...→30s
```

重连逻辑使用 `useRef` 存储 `retryCount` 和 `retryTimerId`，避免闭包陷阱。

### 3. `useMarketDataWS` — 消费侧 Hook

```typescript
// web/src/hooks/use-market-data-ws.ts
export function useMarketDataWS(pair?: string) {
  // 1. 消费 MarketDataContext
  // 2. useEffect: pair 变更时 unsubscribe 旧 pair，subscribe 新 pair
  // 3. 返回：{ connectionStatus, tickerData, subscribe, unsubscribe }
  // 4. pair 为 undefined 时仅返回 connectionStatus，不做订阅
}
```

### 4. `useAdaptivePolling` — 自适应轮询 Hook

```typescript
// web/src/hooks/use-adaptive-polling.ts
const FUNDING_WINDOWS_UTC = [
  { hour: 7, minute: 30 }, { hour: 8, minute: 30 },  // 08:00 结算前后
  { hour: 15, minute: 30 }, { hour: 16, minute: 30 }, // 16:00 结算前后
  { hour: 23, minute: 30 }, { hour: 0, minute: 30 },  // 00:00 结算前后
];

export function useAdaptivePolling(opts: {
  wsStatus: ConnectionStatus;
  priceChangePercent?: number;  // 最近 15 分钟 24h 变化率（来自 WS ticker）
  volatilityThreshold?: number; // 默认 1.0（即 1%）
}): { refetchInterval: number | false }
// WS 正常 → false
// WS 降级 + 活跃时段 → 10_000
// WS 降级 + 非活跃时段 → 60_000
```

### 5. Dashboard 集成方案

`DashboardPage` 的改造最小化：
- `usePortfolioSnapshot` 内部调用 `useMarketDataWS()`（不传 pair，仅获取 `connectionStatus`）和 `useAdaptivePolling()`，通过 `select` 选项在 React Query 缓存命中时用 WS 价格重算 `unrealized_pnl` 和 `equity`
  - 重算逻辑：对每个持仓调用 `getPrice(pos.pair.replace('/', ''))?.price` 获取实时价格
  - `equity` 重算：`cash + positions.reduce((s, p) => s + latestPrice * p.size, 0)`（仅 long；short 需取反，具体依业务逻辑）
  - WS 无数据时（`getPrice` 返回 undefined）保持 REST 原始值，降级透明
- `MetricCardsRow` 接收可选的 `connectionStatus?: ConnectionStatus`，在 WS 已连接时「总权益」卡片显示绿色实时标识
- `PositionsTable` 内部每行抽取为 `PositionRow`（`React.memo`），各自调用 `useMarketDataWS(pair)` 获取实时价格并重算本行 PnL

为防止整个 Dashboard 重渲染，`MetricCardsRow` 和 `PositionRow` 均用 `React.memo` 包裹；`PositionRow` 通过 `useSyncExternalStore` 精确订阅单一交易对，不会因其他交易对价格更新而重渲染。

### 6. MarketView 集成方案

`MarketSidebar` 改造：
- 当 `exchange === 'binance'` 时，调用 `useMarketDataWS(pair)` 订阅 WS
- WS 已连接时：`funding_rate` 和 `open_interest` 在 WS 推送后更新（若 ticker 流包含，否则保持 REST 轮询），REST 轮询频率降至 `useAdaptivePolling` 返回值
- 当 `exchange === 'okx'` 时，维持原有 30 秒固定轮询，不订阅 WS
- pair/exchange 切换时：`useEffect` 依赖数组包含 `[pair, exchange]`，确保旧订阅先 unsubscribe

**注**：Binance ticker 流不包含 `funding_rate` 和 `open_interest`（这些需订阅 `btcusdt@markPrice` 和 `btcusdt@openInterest` 流）。为控制复杂度，本 Feature 仅通过 ticker 流获取**实时价格**，`funding_rate`/`open_interest` 仍依赖自适应轮询 REST 降级。MarketSidebar 的 WS 集成体现在：显示当前 WS 连接状态、使用自适应间隔轮询 REST，以及在 WS 正常时完全停止 REST 轮询。

### 7. `WSStatusIndicator` — 连接状态组件

```tsx
// web/src/components/ws-status-indicator/ws-status-indicator.tsx
// Props: { status: ConnectionStatus; refetchInterval?: number }
// connected → 静默（不渲染任何元素）
// connecting → 灰色"连接中…"脉冲动画
// disconnected/degraded → 黄色徽章"轮询模式 (Xs)" + ARIA role="status"
// reconnecting → 动态"重连中…"
```

挂载位置：`TopBar` 中 `<ApiKeyBadge />` 右侧，条件渲染（`connected` 时不占位）。

### 8. 页面可见性处理（FR-020）

在 `MarketDataProvider` 中：
```typescript
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    // 设置 isPaused = true，onmessage 回调跳过状态更新
  } else {
    // 设置 isPaused = false，恢复分发
    // 触发一次 queryClient.invalidateQueries() 同步快照
  }
});
```

### 9. 消息节流（FR-019）

使用 `Map<string, ReturnType<typeof setTimeout>>` 实现每个交易对独立的 200ms 节流：
```typescript
// onmessage 回调内
const existing = throttleTimers.current.get(pair);
if (existing) clearTimeout(existing);
throttleTimers.current.set(pair, setTimeout(() => {
  priceMapRef.current[pair] = parsed;
  notifyListeners(pair);
  throttleTimers.current.delete(pair);
}, 200));
```

---

## 依赖变更

### npm 包变更

**无新增生产依赖**（使用浏览器原生 `WebSocket` API，满足 NFR-002）。

**可选开发依赖**（bundle size 验证，按需添加）：
```json
{
  "devDependencies": {
    "rollup-plugin-bundle-size": "^1.0.3"
  }
}
```

### Provider 栈变更（`main.tsx`）

```tsx
// 变更前
<QueryClientProvider client={queryClient}>
  <BrowserRouter>
    <ThemeProvider>
      <I18nProvider>
        ...
      </I18nProvider>
    </ThemeProvider>
  </BrowserRouter>
</QueryClientProvider>

// 变更后
<QueryClientProvider client={queryClient}>
  <BrowserRouter>
    <ThemeProvider>
      <I18nProvider>
        <MarketDataProvider>   ← 新增，在 I18nProvider 内层（需要 queryClient 访问）
          ...
        </MarketDataProvider>
      </I18nProvider>
    </ThemeProvider>
  </BrowserRouter>
</QueryClientProvider>
```

`MarketDataProvider` 放在 `I18nProvider` 内侧的理由：页面不可见时恢复时需要调用 `queryClient.invalidateQueries()`，而 `queryClient` 通过 `useQueryClient()` hook 获取，需处于 `QueryClientProvider` 内部。

---

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| Binance WS 被用户网络防火墙屏蔽 | 中 | 中（WS 功能完全失效） | FR-005 降级保底；WSStatusIndicator 告知用户；console 记录（不弹 toast） |
| React StrictMode 双重挂载导致 WS 连接泄漏 | 高（开发环境） | 低（生产无影响） | Provider `useEffect` 返回清理函数关闭 WS；`useRef` 持有 ws 实例 |
| 交易对切换竞态（旧 unsubscribe 与新 subscribe 乱序） | 低 | 中（数据混流） | `useEffect` 依赖 `[pair]` + 清理函数确保先 unsubscribe 再 subscribe；组合流重建而非增量修改 |
| bundle 体积超过 5 KB gzip 预算 | 低 | 中（NFR-002 违规） | 不引入外部库；Provider + hooks 代码量估算约 350 行，gzip 后约 2-3 KB |
| Binance ticker 推送速率过高引发性能问题 | 中 | 低（200ms 节流） | 200ms 节流 + `useSyncExternalStore` 精确订阅；生产前用 Lighthouse 验证 |
| `document.visibilitychange` 恢复后 REST 重复请求 | 低 | 低 | `invalidateQueries` 仅触发一次，staleTime=5s 防止短时间内重复请求 |
| 单元测试中无法使用真实 WebSocket | 必然 | 无 | `MarketDataContext` 暴露 `createWebSocket` 工厂函数参数，测试中注入 mock；符合 NFR-004 |
