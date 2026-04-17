# UI Routes Contract

**Feature**: `001-frontend-rewrite-langalpha-port`
**Date**: 2026-04-16

定义前端 8 个路由的契约：路径、组件、数据依赖、URL 参数、lazy 拆分、e2e 触点。

---

## 路由表

| 路径 | 组件目录 | 优先级 | Lazy chunk | 数据依赖（hook） | 关键 e2e |
|------|---------|--------|-----------|-----------------|---------|
| `/` | `pages/dashboard/` | P1 | ✅ | `usePortfolioSnapshot`, `useEquityCurve`, `useSchedulerStatus` | `dashboard.spec.ts` |
| `/decisions` | `pages/decisions/` | P1 | ✅ | `useDecisions(filter, pagination)`, `useDecisionDetail(commit_hash)` | `decisions.spec.ts` |
| `/backtest` | `pages/backtest/` | P1 | ✅ | `useBacktestRun(run_id)`, `useBacktestSessions`, `useBacktestSession(name)` | `backtest.spec.ts` |
| `/risk` | `pages/risk/` | P1 | ✅ | `useRiskStatus`, `useResetCircuitBreaker` | `risk.spec.ts` |
| `/metrics` | `pages/metrics/` | P1 | ✅ | `useMetricsSummary` + `useMetricsTrend`（IndexedDB） | `metrics.spec.ts` |
| `/chat` | `pages/chat/` | P2 | ✅ | `useChatMessages(sessionId)` | `chat.spec.ts`（P2） |
| `/market` | `pages/market/` | P2 | ✅ | `useMarketSnapshot(pair, exchange)` | `market.spec.ts`（P2） |
| `*` | `pages/not-found.tsx` | — | ❌（主 bundle） | — | — |

---

## 1. `/` Dashboard — FR-100~106

**组件契约**：
- `<DashboardPage>`：顶层
  - `<MetricCardsRow>` × 4（equity / cash / pnl_24h / drawdown）
  - `<EquityChart>`（lightweight-charts，range 切换器）
  - `<PositionsTable>`
  - `<SchedulerCard>`
- React Query：`refetchInterval: 10_000`（FR-105）
- ErrorBoundary：包整页，错误显示 fallback + 重试按钮

**URL 参数**：无
**Lazy 触发**：进入路由 → import dashboard chunk
**关键交互**：
- 切换 equity range → 触发 `useEquityCurve(range)` 重 fetch
- Redis 不可用 → SchedulerCard 显示警告（FR-104）

---

## 2. `/decisions` Decisions — FR-200~211

**组件契约**：
- `<DecisionsPage>`：split view
  - `<DecisionsFilterBar>`：pair 下拉 + 日期范围 + 分页
  - `<DecisionsTable>`：列表（点击行 → 选中）
  - `<DecisionDetailPanel>`：右侧详情（8 节）
    - `<NodeTimelinePipeline>` (FR-204)
    - `<AgentAnalysisGrid>` (FR-205) × 4 卡
    - `<ExperienceMemorySection>` (FR-206)
    - `<DebateSection>` (FR-207)
    - `<VerdictCard>` (FR-208)
    - `<RiskGateSection>` (FR-209)
    - `<ExecutionSection>` (FR-210)

**URL 参数**：
- `?id=<commit_hash>` — 选中决策（FR-211 同步）
- 浏览器前进/后退 → 切换选中

**Lazy 内层拆分**：
- `decision-detail/` 组件目录被 Backtest 详情复用，应在 `components/` 而非 `pages/`

---

## 3. `/backtest` Backtest — FR-300~307

**组件契约**：
- `<BacktestPage>`：Tabs
  - Tab 1 `<NewBacktestForm>`：react-hook-form + zod schema
  - Tab 2 `<HistoricalSessions>`：会话名下拉
- 提交后 `<BacktestProgressCard>`：进度条 + 取消按钮（FR-302）
- 完成后 `<BacktestResultView>`：
  - `<MetricCardsRow>` × 5（FR-304）
  - `<EquityChart>`（复用 dashboard 组件，FR-305）
  - `<DecisionTimelineTable>`（FR-306，仅 LLM 模式）
  - 点击单条 → 弹出 `<DecisionDetailPanel>`（复用 decisions 组件）

**URL 参数**：
- `?run_id=<id>` — 跟随当前任务
- `?session=<name>` — 加载历史会话

**轮询**：`refetchInterval: 5_000`，状态 `completed/failed/canceled` 时停止

---

## 4. `/risk` Risk — FR-400~405

**组件契约**：
- `<RiskPage>`
  - `<TradeCountCard>` × 2（hour / day）
  - `<CircuitBreakerCard>`：active 红色 + 倒计时；inactive 绿色（FR-402）
  - `<RiskThresholdsTable>`（FR-403）
  - `<ResetCircuitBreakerButton>`：confirm dialog（NFR-S-008）
- React Query：`refetchInterval: 5_000`
- Redis 不可达 → 整页 warning banner（FR-405）

**URL 参数**：无

---

## 5. `/metrics` Metrics — FR-500~504

**组件契约**：
- `<MetricsPage>`
  - `<CounterCardsGrid>`：4 卡（trades / orders_placed / orders_failed / risk_rejections）（FR-501）
  - `<LatencyPercentilesGrid>`：pipeline / execution × p50 / p95（FR-502）
  - `<MetricsTrendChart>`：4 曲线 lightweight-charts（FR-503）
- IndexedDB FIFO 60 样本（idb 库，hooks 内封装）
- React Query：`refetchInterval: 15_000`（FR-504）

**URL 参数**：无

---

## 6. `/chat` ChatAgent — FR-600~604（P2）

**组件契约**：
- `<ChatPage>`：三栏布局
  - 左 `<SessionList>`
  - 中 `<MessageStream>`（含 `<InlineWidget>` 嵌入）
  - 右 `<ChatInput>`（多行 + Cmd+Enter + 模型下拉）
- `useChatMessages` hook：≤ 500 行硬限（NFR-M-007）
- `<InlineWidget>`：iframe sandbox（NFR-S-003）

**URL 参数**：
- `?session=<id>` — 当前会话

---

## 7. `/market` MarketView — FR-700~702（P2）

**组件契约**：
- `<MarketPage>`
  - `<TradingViewWidget>`（iframe，懒加载 NFR-P-010）
  - 侧栏 `<FundingRatePanel>` / `<OpenInterestPanel>` / `<LiquidationsHeatmap>`
  - `<ExchangeSwitcher>`：Binance / OKX

**URL 参数**：
- `?pair=BTC-USDT` — 当前币对
- `?exchange=binance` — 交易所

---

## 8. 全局 Provider 栈

```tsx
// main.tsx
<QueryClientProvider client={queryClient}>
  <BrowserRouter>
    <ThemeProvider>
      <I18nProvider>
        <App />
        <Toaster />  {/* 全局 toast */}
      </I18nProvider>
    </ThemeProvider>
  </BrowserRouter>
</QueryClientProvider>
```

**约束**：
- ThemeProvider 注入 CSS 变量（暗/亮，符合 WCAG AA 对比度 NFR-A-002）
- I18nProvider 默认 zh-CN，运行时切换 en-US 不刷新（SC-007）

---

## 9. 共享布局

**`<AppShell>`**：
- 左侧 `<Sidebar>`：导航（图标 + i18n 标签）
- 顶部 `<Topbar>`：主题切换 / 语言切换 / API 状态指示
- 主内容区：`<Outlet>`（react-router）

**键盘导航**：
- `Tab` 顺序：Sidebar → Topbar → Main
- 每个交互元素 `aria-label`（NFR-A-003）
- `Cmd/Ctrl + K`：全局命令面板（P3，本期不做）

---

## 10. ErrorBoundary 与 Suspense 边界

```tsx
<ErrorBoundary fallback={<GlobalErrorFallback />}>
  <Suspense fallback={<RouteSkeleton />}>
    <Routes>
      <Route path="/" element={<DashboardPage />} />
      ...
    </Routes>
  </Suspense>
</ErrorBoundary>
```

**约束**：
- 每个页面级路由有 `<RouteSkeleton>`（避免空白闪烁）
- `<GlobalErrorFallback>` 显示 trace_id（若 OTel 可用）+ 重试按钮
- console.error 同时上报到预留 hook（NFR-O-001，本期不发 Sentry）

---

## 11. e2e 触点（Playwright）

每页 1 个 spec 文件，覆盖 happy path + 关键 edge：

| spec | 关键 case |
|------|----------|
| `dashboard.spec.ts` | 4 卡渲染 / 10s 自动刷新 / Redis down 警告 |
| `decisions.spec.ts` | 筛选 / 分页 / 选中同步 URL / 详情 8 节渲染 |
| `backtest.spec.ts` | 提交 → 轮询 → 结果 / 取消任务 / 加载历史会话 |
| `risk.spec.ts` | ACTIVE 卡 / confirm dialog / reset 成功 |
| `metrics.spec.ts` | 计数器 / 延迟 / 趋势图 60 样本 FIFO |
| `streamlit-removal.spec.ts` | 4 条 ripgrep 命令 0 命中（FR-915） |
| `chat.spec.ts`（P2） | 5 种 SSE 事件 / iframe sandbox |
| `market.spec.ts`（P2） | TradingView 加载 / 资金费率渲染 |

---

## 12. 路由懒加载示例

```tsx
const DashboardPage = lazy(() => import('./pages/dashboard'));
const DecisionsPage = lazy(() => import('./pages/decisions'));
// ...

<Routes>
  <Route path="/" element={<DashboardPage />} />
  <Route path="/decisions" element={<DecisionsPage />} />
  ...
</Routes>
```

每个 page 编译为独立 chunk；主 bundle 仅含 Provider + Layout + 公共组件 ≤ 80KB（详见 [research.md §10](../research.md)）。
