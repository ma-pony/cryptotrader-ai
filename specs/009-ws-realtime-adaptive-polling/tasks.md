# 实施任务清单：WS 实时行情 + 自适应轮询

**Feature Branch**: `009-ws-realtime-adaptive-polling`
**依据**: `plan.md` + `spec.md` (2026-04-17 Draft)
**范围**: 纯前端，所有改动在 `web/` 目录内

**标记说明**：
- `[P]` 可与同组内其他 `[P]` 任务并行
- `[US1]~[US4]` 对应 spec.md 用户故事编号
- 预估工时为单人参考值（1-4 小时/任务）

---

## 阶段 0：基础类型与 Context 骨架

> **目标**：搭建 WS 数据层的类型契约与 Context 骨架，后续所有任务均依赖此基础。串行执行。

- [X] T001 [US1][US2] 创建 `web/src/contexts/market-data/market-data-context.ts`
  - 定义 `TickerData` 接口（字段：`pair: string`, `price: number`, `priceChangePercent: number`, `volume24h: number`, `ts: number`）
  - 定义 `ConnectionStatus` 联合类型（`'connecting' | 'connected' | 'disconnected' | 'degraded' | 'reconnecting'`）
  - 定义 `MarketDataContextValue` 接口（`connectionStatus`, `subscribe`, `unsubscribe`, `getPrice`, `subscribeToPrice`）
  - 创建 `MarketDataContext`（`React.createContext<MarketDataContextValue | null>(null)`）并导出
  - 创建 `web/src/contexts/market-data/index.ts` 统一导出
  - **预估工时**：1 小时

- [X] T002 [US1][US2] 创建 `web/src/contexts/market-data/market-data-provider.tsx` — Provider 框架
  - 实现 `subscriptions: Map<string, number>` 订阅计数器和 `priceMapRef: Ref<Record<string, TickerData>>` 价格存储
  - 实现 `subscribe(pair)` / `unsubscribe(pair)` 逻辑（计数器增减，降至 0 时自动取消订阅）
  - 实现 `subscribeToPrice(pair, cb)` 外部订阅通知机制（`listeners: Map<string, Set<() => void>>`）
  - 实现 `getPrice(pair)` 从 `priceMapRef.current` 读取（不触发 re-render）
  - 暴露 `createWebSocket?: (url: string) => WebSocket` prop 用于测试注入（NFR-004）
  - **此任务仅实现状态管理骨架，不包含 WS 连接逻辑（见 T003）**
  - **预估工时**：2 小时

---

## 阶段 1：WebSocket 连接管理

> **目标**：实现 WS 连接生命周期，含握手、消息处理、节流、重连。依赖 T001、T002。

- [X] T003 [US1][US2] 实现 `MarketDataProvider` 中的 WS 连接核心逻辑
  - 使用 `useEffect` 管理 WS 实例生命周期：`subscriptions` map 有条目时建立连接，清空时关闭
  - 目标 URL：`wss://stream.binance.com:9443/stream?streams=<pair1>@ticker/<pair2>@ticker`
  - 实现 `onopen`：设置 `connectionStatus → 'connecting'`，2 秒后设置为 `'connected'`（FR-017）
  - 实现 `onmessage`：从 `{ stream, data }` 结构解析 Binance 组合流消息；提取 `s`, `c`, `P`, `v`, `E` 字段；字符串转 number（FR-018）
  - 实现 200ms 节流：每个交易对独立 `setTimeout`，同窗口内覆盖，防止高频 re-render（FR-019）
  - 实现 `onerror` / `onclose`：启动 3 秒计时，超时后设置 `'degraded'`（FR-005）
  - 实现指数退避重连：初始 1 秒，最大 30 秒，最多 10 次，使用 `useRef` 存储计时器 ID（FR-006）
  - 所有 WS 事件使用 `console.info('[WSMarketData] <event>')` 格式记录（NFR-005）
  - 组件 unmount 时清理：关闭 WS、清除所有计时器
  - **预估工时**：4 小时

- [X] T004 [US2] 实现 `MarketDataProvider` 页面可见性处理
  - `document.addEventListener('visibilitychange', ...)` 在 `useEffect` 中注册，cleanup 时移除
  - 页面隐藏时：设置 `isPausedRef.current = true`，`onmessage` 回调中检查此 flag，暂停状态更新
  - 页面恢复时：设置 `isPausedRef.current = false`；调用 `queryClient.invalidateQueries({ queryKey: ['portfolio-snapshot'] })` 触发快照同步（FR-020）
  - 通过 `useQueryClient()` 获取 queryClient（Provider 处于 QueryClientProvider 内层）
  - **预估工时**：1 小时

---

## 阶段 2：自定义 Hooks

> **目标**：实现消费侧 hook 和自适应轮询 hook。T005 依赖 T001/T002，T006 无外部依赖，可并行。

- [X] T005 [P][US1][US2][US3] 创建 `web/src/hooks/use-market-data-ws.ts`
  - 消费 `MarketDataContext`（含 null 检查，未包裹时抛出清晰错误信息）
  - 接收 `pair?: string` 参数
  - `useEffect`：`pair` 有值时调用 `subscribe(pair)` + 返回 `() => unsubscribe(pair)` 清理（FR-015）
  - 使用 `useSyncExternalStore` 订阅 `subscribeToPrice`，精确响应单个交易对价格变更（NFR-001）
  - 返回：`{ connectionStatus, tickerData: TickerData | undefined }`
  - **预估工时**：1.5 小时

- [X] T006 [P][US2][US4] 创建 `web/src/hooks/use-adaptive-polling.ts`
  - 定义资金费率结算窗口常量：UTC 07:30-08:30、15:30-16:30、23:30-00:30（FR-009）
  - 实现 `isInFundingWindow(now: Date): boolean` 内部函数
  - 实现 `isActiveMarket(priceChangePercent: number | undefined, threshold: number): boolean`（24h 波动率绝对值 > threshold）
  - Hook 签名：`useAdaptivePolling({ wsStatus, priceChangePercent, volatilityThreshold? = 1.0 })`
  - 返回 `{ refetchInterval: number | false }`：
    - WS `connected` / `connecting` → `false`（FR-010，WS 正常时禁止 REST 轮询，FR-007）
    - WS `degraded` + 活跃时段 → `10_000`（FR-008）
    - WS `degraded` + 非活跃时段 → `60_000`（FR-008）
    - WS `disconnected` / `reconnecting` → `10_000`（保守默认）
  - **预估工时**：1.5 小时

---

## 阶段 3：UI 组件

> **目标**：实现 WS 状态指示器组件。依赖 T001（ConnectionStatus 类型）。

- [X] T007 [US3] 创建 `web/src/components/ws-status-indicator/ws-status-indicator.tsx`
  - Props：`{ status: ConnectionStatus; refetchInterval?: number }`
  - `connected` 状态：返回 `null`（静默，不占位，FR-016）
  - `connecting` 状态：灰色脉冲圆点 + 无文字（避免过于嘈杂）
  - `reconnecting` 状态：黄色旋转图标 + "重连中…" 文字
  - `degraded` / `disconnected` 状态：黄色三角警告图标 + "轮询模式"；若 `refetchInterval` prop 已提供，则追加间隔文字（如 `10s` / `60s`）（FR-016）
  - `aria-label` 含完整描述，无 `refetchInterval` 时为"数据延迟中，轮询模式"；有时为"数据延迟中，当前轮询间隔 10 秒"（SC-008）
  - 添加 `role="status"` 和 `aria-live="polite"` 满足 WCAG 2.1 AA
  - 创建 `web/src/components/ws-status-indicator/index.ts` 导出
  - **预估工时**：1.5 小时

- [X] T008 [US3] 修改 `web/src/components/layout/top-bar.tsx` — 插入状态指示器
  - 导入 `useMarketDataWS`（不传 pair，仅获取 `connectionStatus`）
  - 在 `<ApiKeyBadge />` 右侧插入 `<WSStatusIndicator status={connectionStatus} />`
  - **注意**：TopBar 无交易对上下文，不调用 `useAdaptivePolling`；`WSStatusIndicator` 在降级状态时仅显示"轮询模式"文字，不显示具体间隔秒数（间隔信息由 Dashboard / MarketSidebar 各自组件负责展示）
  - **预估工时**：0.5 小时

---

## 阶段 4：Provider 注入

> **目标**：将 MarketDataProvider 注入全局 Provider 栈。依赖 T003。

- [X] T009 [US1][US2][US3] 修改 `web/src/main.tsx` — 注入 MarketDataProvider
  - 导入 `MarketDataProvider`（`import { MarketDataProvider } from './contexts/market-data'`）
  - 当前 `main.tsx` 实际结构（`<I18nProvider>` 内含 `<ErrorBoundary>` + `<Toaster />`），插入位置为**在 `<ErrorBoundary>` 外层、`<I18nProvider>` 内层**：
    ```tsx
    <I18nProvider>
      <MarketDataProvider>   {/* ← 新增，包裹 ErrorBoundary */}
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
        <Toaster />
      </MarketDataProvider>
    </I18nProvider>
    ```
  - 最终完整 Provider 顺序：`QueryClientProvider → BrowserRouter → ThemeProvider → I18nProvider → MarketDataProvider → ErrorBoundary → App`
  - `MarketDataProvider` 处于 `QueryClientProvider` 内部，确保内部 `useQueryClient()` 调用可正常访问 `queryClient`（页面恢复时触发 `invalidateQueries`）
  - **预估工时**：0.5 小时

---

## 阶段 5：Dashboard 集成

> **目标**：Dashboard 核心指标实时刷新。依赖 T005、T006。

- [X] T010 [US1] 修改 `web/src/hooks/use-portfolio-snapshot.ts` — 整合 WS 实时 PnL 重算
  - 增加内部调用 `useMarketDataWS()` 获取 `connectionStatus`；通过 `useContext(MarketDataContext)` 获取 `getPrice` 函数引用（`getPrice` 从 `priceMapRef` 读取，不触发 re-render，符合 NFR-001）
  - 增加内部调用 `useAdaptivePolling({ wsStatus: connectionStatus, priceChangePercent: undefined })` 获取 `refetchInterval`（Portfolio hook 无单一交易对上下文，`priceChangePercent` 传 undefined，降级为保守轮询策略）
  - `useQuery` 的 `select` 选项：当 `connectionStatus === 'connected'` 时，遍历 `positions`，用 `getPrice(pos.pair.replace('/', ''))?.price` 重算 `unrealized_pnl = (latestPrice - avg_price) * size`（FR-011）；重算 `equity = cash + sum(positions.map(p => (getPrice(p.pair.replace('/', ''))?.price ?? p.avg_price) * p.size))`（FR-012）
  - 将 `refetchInterval` 替换为 `useAdaptivePolling` 的返回值（FR-007、FR-010）
  - WS 无数据时（`getPrice` 返回 undefined）保持原始 REST 数据，不做替换（降级透明，SC-002）
  - 返回值追加 `connectionStatus` 字段供 Dashboard 组件使用
  - **预估工时**：2 小时

- [X] T011 [US1] 修改 `web/src/pages/dashboard/components/metric-cards-row.tsx`
  - Props 接口增加 `connectionStatus?: ConnectionStatus`（可选，不破坏现有调用）
  - `总权益` MetricCard 在 WS connected 时显示特殊标识（如绿色小圆点），无障碍 `aria-label` 含"实时数据"
  - 使用 `React.memo` 包裹整个组件（NFR-001）
  - **预估工时**：1 小时

- [X] T012 [US1] 修改 `web/src/pages/dashboard/components/positions-table.tsx`
  - 每行 `<tr>` 抽取为独立的 `PositionRow` 子组件并用 `React.memo` 包裹（NFR-001）
  - `PositionRow` 通过 `useMarketDataWS(pos.pair.replace('/', ''))` 获取实时 `tickerData`
  - 实时 PnL 值优先使用 WS 计算结果（与 T010 的 `select` 重算保持一致）；WS 无数据时回退到 REST 值
  - **预估工时**：2 小时

- [X] T013 [US1] 修改 `web/src/pages/dashboard/index.tsx`
  - 从 `usePortfolioSnapshot()` 返回值中解构 `connectionStatus`
  - 将 `connectionStatus` 传递给 `<MetricCardsRow>`
  - **预估工时**：0.5 小时

---

## 阶段 6：MarketView 集成

> **目标**：MarketSidebar 自适应轮询 + WS 状态显示。依赖 T005、T006、T007。

- [X] T014 [US2] 修改 `web/src/pages/market/components/market-sidebar.tsx`
  - 仅在 `exchange === 'binance'` 时调用 `useMarketDataWS(pair.replace('/', ''))`
  - 调用 `useAdaptivePolling({ wsStatus: connectionStatus, priceChangePercent: tickerData?.priceChangePercent })`
  - `useQuery` 的 `refetchInterval` 替换为 `useAdaptivePolling` 返回的 `refetchInterval`（FR-010）
  - WS connected 时：在资金费率卡片标题旁显示微型 `WSStatusIndicator`（仅 `connected` 时为绿色圆点，其他状态静默）
  - `pair` / `exchange` 变更时（`useEffect` 依赖数组）：先 unsubscribe 旧 pair，再 subscribe 新 pair（FR-015）
  - exchange 切换为 `okx` 时：不订阅 WS，保持原有 30 秒固定轮询（spec 假设约束）
  - **预估工时**：2 小时

---

## 阶段 7：单元测试

> **目标**：核心 hooks 和 Provider 的单元测试覆盖率 ≥ 80%（SC-007）。T015、T016、T017 可并行。

- [X] T015 [P][US1][US2] 创建 `web/src/__tests__/hooks/use-market-data-ws.test.ts`
  - 创建 `MockWebSocket` 类（实现 `send`, `close`, `onopen`, `onmessage`, `onclose`, `onerror`）
  - 通过 `MarketDataProvider` 的 `createWebSocket` prop 注入 mock（NFR-004）
  - 测试用例：
    1. WS 连接成功后 `connectionStatus` 变为 `'connected'`
    2. `subscribe('BTCUSDT')` 后 `tickerData` 在收到 mock 消息后更新
    3. `unsubscribe('BTCUSDT')` 后不再更新 `tickerData`
    4. 200ms 节流：100ms 内推送 3 条消息，仅触发 1 次 React 状态更新
    5. WS `onclose` 触发后 3 秒内 `connectionStatus` 变为 `'degraded'`
    6. 重连成功后 `connectionStatus` 变回 `'connected'`
  - **预估工时**：3 小时

- [X] T016 [P][US4] 创建 `web/src/__tests__/hooks/use-adaptive-polling.test.ts`
  - 测试用例：
    1. WS `connected` 时返回 `false`（不轮询）
    2. WS `degraded` + 波动率 2% + 非结算窗口 → `10_000`
    3. WS `degraded` + 波动率 0.5% + 非结算窗口 → `60_000`
    4. WS `degraded` + 波动率 0.5% + 处于 UTC 07:45（结算窗口内）→ `10_000`
    5. WS `degraded` + 波动率 0.5% + 处于 UTC 09:00（窗口外）→ `60_000`
    6. `priceChangePercent` 为 `undefined`（WS 未收到数据）→ 默认非活跃 → `60_000`
  - 使用 `vi.useFakeTimers()` 控制系统时间（Vitest 内置）
  - **预估工时**：2 小时

- [X] T017 [P][US1][US2][US3] 创建 `web/src/__tests__/hooks/market-data-provider.test.tsx`
  - 使用 `MockWebSocket` 注入
  - 测试用例：
    1. 订阅计数器：同一 pair 订阅 2 次，取消 1 次后仍保持连接；取消第 2 次后自动发送 UNSUBSCRIBE
    2. 组合流 URL 构建：订阅 `BTCUSDT` 和 `ETHUSDT` 时 URL 包含两个 stream
    3. 价格字符串解析：mock 消息中 `"c": "43250.00"` 被解析为 `number` 43250
    4. 页面隐藏时消息不触发状态更新；页面恢复后恢复更新
    5. Provider unmount 时 WS `close()` 被调用
  - **预估工时**：3 小时

- [X] T018 [US3] 创建 `web/src/__tests__/components/ws-status-indicator.test.tsx`
  - **前置步骤**：先将 `vitest-axe` 添加到开发依赖：`pnpm add -D vitest-axe`（提供 `axe` matcher，与 Vitest 原生集成，无需额外配置；若已存在则跳过）
  - 测试各状态下的渲染输出
  - 使用 `vitest-axe`（`import { axe } from 'vitest-axe'`）验证 ARIA 合规性（SC-008）
  - 测试用例：
    1. `connected` → 不渲染任何 DOM 元素
    2. `degraded` + `refetchInterval=10000` → 显示文字含"10s"
    3. `degraded` + `refetchInterval=60000` → 显示文字含"60s"
    4. `reconnecting` → 显示"重连中"相关文字
    5. ARIA：`role="status"` 存在；`aria-label` 含有完整描述；axe-core 无违规
  - **预估工时**：2 小时

---

## 阶段 8：收尾与验收

> **目标**：整合验证、包体积检查、边界条件确认。依赖所有前序任务完成。

- [X] T019 [US1][US2][US3][US4] 端到端手动验收测试
  - 按 spec 验收场景逐一验证：
    - SC-001：Dashboard 总权益在 WS 推送 2 秒内更新
    - SC-002：WS 断线 3 秒内降级，数据不空白
    - SC-003：WS 重连后 5 秒内 REST 轮询停止
    - SC-004：MarketView 切换交易对，1 秒内旧订阅取消、新订阅建立
    - SC-005：非活跃时段 WS 降级时，Network Tab 验证轮询间隔 60 秒
  - 使用浏览器 DevTools → Network → WS 标签页验证订阅/取消消息
  - 使用 DevTools 模拟网络断开（Offline 模式）验证降级
  - **预估工时**：2 小时

- [X] T020 Bundle 体积验证（SC-006）
  - 运行 `pnpm build` 并检查 dist 目录
  - 使用 `gzip -c dist/assets/*.js | wc -c` 对比新增部分
  - 或在 `vite.config.ts` 中添加 `rollup-plugin-bundle-size` 输出报告
  - 确认新增 gzip 增量 ≤ 5 KB
  - **预估工时**：0.5 小时

- [X] T021 TypeScript 类型检查与 Lint
  - 运行 `pnpm typecheck`（`tsc --noEmit`）确认零类型错误
  - 运行 `pnpm lint` 确认零 ESLint 错误（禁止 noqa，零 lint 错误原则）
  - 确认无 `any` 类型逃逸（WS 消息解析使用 Zod schema 或手动类型守卫）
  - **预估工时**：1 小时

---

## 任务依赖图

```
T001 ──► T002 ──► T003 ──► T009
              │         └──► T004
              │
              ├──► T005 ──► T010 ──► T011 ──► T013
              │         └──► T012
              │         └──► T014
              │
              ├──► T006 ──► T010
              │         └──► T014
              │
              └──► T007 ──► T008
                        └──► T014

T015、T016、T017（可并行，依赖 T001~T006）
T018（依赖 T007）

T019、T020、T021（依赖所有前序任务）
```

---

## 用户故事覆盖矩阵

| 任务 | US1 Dashboard 实时 | US2 MarketView 实时 | US3 状态可见性 | US4 自适应轮询 |
|------|--------------------|--------------------|----------------|----------------|
| T001 | ✓ | ✓ | ✓ | |
| T002 | ✓ | ✓ | ✓ | |
| T003 | ✓ | ✓ | | |
| T004 | ✓ | ✓ | | |
| T005 | ✓ | ✓ | ✓ | |
| T006 | | ✓ | | ✓ |
| T007 | | | ✓ | |
| T008 | | | ✓ | |
| T009 | ✓ | ✓ | ✓ | ✓ |
| T010 | ✓ | | | ✓ |
| T011 | ✓ | | ✓ | |
| T012 | ✓ | | | |
| T013 | ✓ | | | |
| T014 | | ✓ | ✓ | ✓ |
| T015 | ✓ | ✓ | | |
| T016 | | | | ✓ |
| T017 | ✓ | ✓ | ✓ | |
| T018 | | | ✓ | |
| T019 | ✓ | ✓ | ✓ | ✓ |
| T020 | | | | |
| T021 | | | | |

**总计**：21 个任务，估计总工时约 32-36 小时
