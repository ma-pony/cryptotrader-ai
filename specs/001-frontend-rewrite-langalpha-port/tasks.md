# Tasks: 前端重写 — LangAlpha 移植 + Crypto 化

**Input**: Design documents from `/specs/001-frontend-rewrite-langalpha-port/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: 已请求（spec NFR-M-002/003 + plan TDD 偏好 + A-9 少 mock 走真实 e2e）。包含 Vitest 单元测试 + Playwright e2e + pytest 后端测试。

**Organization**: 11 个 phase 串行（来自 quickstart.md §4 + spec §11）。Phase 1-2 共享基础；Phase 3-7 是 5 个 P1 用户故事；Phase 8 是 Streamlit 物理删除（硬门槛）；Phase 9-10 是 P2；Phase 11 收尾。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无未完成依赖）
- **[Story]**: 归属用户故事（US1~US7）
- 所有路径为绝对/相对仓库根的明确路径

## Path Conventions

- **前端**: `web/` 子目录（新增），自包含 `src/` `tests/` `public/`
- **后端**: `src/api/routes/`（既有）+ `src/cryptotrader/`（既有，本特性原则上不动业务逻辑）
- **后端测试**: `tests/`（既有）

---

## Phase 1: Setup（Shared Infrastructure，FR-001~011 脚手架）

**Purpose**: 建立 `web/` 子目录脚手架，安装核心技术栈，配置 lint/format/test/build 工具链。

- [X] T001 创建 `web/` 子目录与基础文件结构：`web/{src,tests,public}` + `web/.gitignore` + `web/README.md`（占位）
- [X] T002 在 `web/` 内 `pnpm init` 并写入 `web/package.json`：name `cryptotrader-web`，private true，type module，scripts（`dev`/`build`/`preview`/`lint`/`typecheck`/`test`/`test:e2e`）
- [X] T003 安装核心依赖（在 `web/` 内执行）：`pnpm add react@^19.2 react-dom@^19.2 react-router@^7 @tanstack/react-query@^5 zustand@^4 i18next@^23 react-i18next@^14 zod@^3 react-hook-form@^7 @hookform/resolvers@^3 lightweight-charts@^4 idb@^8 lucide-react@latest class-variance-authority clsx tailwind-merge react-markdown rehype-sanitize`
- [X] T004 [P] 安装 Radix UI 原子（在 `web/` 内）：`pnpm add @radix-ui/react-dialog @radix-ui/react-dropdown-menu @radix-ui/react-popover @radix-ui/react-scroll-area @radix-ui/react-tabs @radix-ui/react-tooltip @radix-ui/react-separator @radix-ui/react-toast`
- [X] T005 [P] 安装 dev 依赖（在 `web/` 内）：`pnpm add -D vite@^7 @vitejs/plugin-react@^5 typescript@^5.9 @types/react @types/react-dom @types/node tailwindcss@^3 postcss autoprefixer eslint @eslint/js typescript-eslint eslint-plugin-react-hooks eslint-plugin-i18next prettier vitest @vitest/coverage-v8 @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom @playwright/test web-vitals jiti`（注：`@vitejs/plugin-react@^5` 与 vite 7 对齐；`eslint-plugin-import` 暂移除待 ESLint 10 兼容；`jiti` 给 ESLint 加载 TS config）
- [X] T006 [P] 写入 `web/tsconfig.json`：strict + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes` + `target ES2022` + `module ESNext` + `moduleResolution bundler` + path alias `@/* → ./src/*`（NFR-M-001）
- [X] T007 [P] 写入 `web/vite.config.ts`：React plugin + 路径 alias + dev server proxy 转发 `/api` 到 `http://localhost:8003`（FR-009，对齐 `arena serve` 默认端口）+ build target `es2022` + `manualChunks`（拆分 lightweight-charts / react-markdown 重型依赖）
- [X] T008 [P] 写入 `web/tailwind.config.ts` + `web/postcss.config.cjs` + `web/src/styles/globals.css`：CSS variables 暗/亮主题（FR-005）+ Tailwind base/components/utilities + WCAG AA 对比度色板（NFR-A-002）
- [X] T009 [P] 写入 `web/eslint.config.ts`（flat config）：`@typescript-eslint/recommended-type-checked` + `react-hooks/recommended` + `eslint-plugin-i18next`，0 警告策略（NFR-M-002/003）
- [X] T010 [P] 写入 `web/prettier.config.js`：line-length 120（NFR-M-004）
- [X] T011 [P] 写入 `web/playwright.config.ts`：base URL `http://localhost:5173`，chromium project，retries 0（dev）/2（CI），webServer 启动 `pnpm dev` + 等待 docker compose 后端
- [X] T012 [P] 写入 `web/vitest.config.ts`：jsdom 环境 + setupFiles（`./tests/unit/setup.ts` 引入 `@testing-library/jest-dom`）+ coverage v8
- [X] T013 [P] 写入 `web/.env.example`：`VITE_API_BASE_URL=http://localhost:8003`、`VITE_API_KEY=`、`VITE_OTLP_UI_ENDPOINT=`（FR-009）
- [X] T014 [P] 写入 `web/index.html`：`<html lang="zh-CN">` + favicon + meta theme-color（暗/亮）+ root div + main.tsx 入口
- [X] T015 [P] 写入 `web/Dockerfile`（multi-stage：node:20-alpine 构建 → nginx:alpine 托管）+ `web/nginx.conf`（SPA fallback + security headers NFR-S-006：`X-Content-Type-Options nosniff`、`X-Frame-Options DENY`、`Referrer-Policy no-referrer`、`CSP` 连接源白名单）
- [X] T016 写入 `web/src/main.tsx`：最小 hello world（验证 `pnpm dev` 跑通），后续 Phase 2 替换为 Provider 栈
- [X] T017 验证 Phase 1 DoD：`pnpm dev` HTTP 200；`pnpm typecheck` 0 错；`pnpm lint` 0 警告；`pnpm build` 成功，主 bundle 60 KB gzipped（远低于 NFR-P-002 ≤ 300 KB）

**Checkpoint**: `pnpm dev` 浏览器 hello world；lint/typecheck 0 错 0 警告；Phase 1 闭合。

---

## Phase 2: Foundational（Provider 栈 / 路由 / Layout / 主题 / i18n / API client / SSE，BLOCKS all user stories）

**Purpose**: 构建所有 7 路由共用的基础设施（Provider 栈、AppShell 布局、API client、streamFetch SSE 客户端、UI 原子、错误处理、i18n 框架）。**这些任务不属于任何单个用户故事，但所有用户故事都依赖**。

⚠️ **CRITICAL**: 本 phase 完成前不得开始 Phase 3+。

### 2.1 类型与 schema 基线

- [X] T018 [P] 写入 `web/src/types/api.ts`：导出所有 data-model.md §1-§9 的 TypeScript interface（Portfolio / Position / EquityCurve / DecisionCommit / AgentAnalysis / DebateRound / Verdict / RiskGate / Execution / NodeTimelineEntry / ExperienceMemoryRef / ExperienceRule / SchedulerStatus / RiskStatus / CircuitBreakerStatus / RiskThresholds / MetricsSummary / BacktestParams / BacktestRun / BacktestResult / BacktestSession / ChatMessage / MarketDataPoint / ApiError / PaginatedResponse）
- [X] T019 [P] 写入 `web/src/types/api.schema.ts`：与 T018 对齐的 zod schema，文件名 `*.schema.ts`（NFR-M-010）

### 2.2 工具与 client

- [X] T020 [P] 写入 `web/src/lib/env.ts`：类型化 `import.meta.env`（VITE_API_BASE_URL / VITE_API_KEY / VITE_OTLP_UI_ENDPOINT），首屏 fail-fast 校验（zod）
- [X] T021 [P] 写入 `web/src/lib/api-client.ts`：fetch 包装，统一注入 `X-API-Key` header（NFR-S-001），统一 ApiError 解析（contracts/http-endpoints.md §9），AbortController 支持
- [X] T022 [P] 写入 `web/src/lib/stream-fetch.ts`：移植自 LangAlpha 的 streamFetch（fetch + ReadableStream）；处理 401/429/413/404/500/503（contracts/sse-events.md §4）；JSON.parse patch（NaN/Infinity）；debug 模式（NFR-O-004）
- [X] T023 [P] 写入 `web/src/lib/query-client.ts`：QueryClient 默认配置（staleTime 5s、retry 1、refetchOnWindowFocus false）+ MutationCache 错误统一 toast
- [X] T024 [P] 写入 `web/src/lib/cn.ts`：Tailwind className merger（`clsx` + `tailwind-merge`）
- [X] T025 [P] 写入 `web/src/lib/format.ts`：Intl.DateTimeFormat 本地化（DG-2 zh-CN/en-US 时间格式）+ 金额千分位 `formatCurrency` + PnL 涨跌色辅助 `pnlClass`

### 2.3 i18n（FR-006）

- [X] T026 [P] 写入 `web/src/locales/zh-CN/common.json` + `dashboard.json` + `decisions.json` + `backtest.json` + `risk.json` + `metrics.json` + `chat.json` + `market.json`（按 namespace 拆分；中文先行）
- [X] T027 [P] 写入 `web/src/locales/en-US/common.json` 等 8 个 namespace 占位翻译（缺失键回退 zh-CN）
- [X] T028 写入 `web/src/lib/i18n.ts`：i18next 初始化（默认 zh-CN，回退 zh-CN，namespace 分文件，detection 优先 localStorage）

### 2.4 状态（Zustand）

- [X] T029 [P] 写入 `web/src/stores/use-ui-store.ts`：theme（dark/light/system）、locale（zh-CN/en-US）、sidebarCollapsed（NFR-M-008 按领域拆分）
- [X] T030 [P] 写入 `web/src/stores/use-settings-store.ts`：`apiKey?`（仅运行时内存，不持久化 NFR-S-005）+ otlp endpoint
- [X] T031 [P] 写入 `web/src/stores/use-chat-store.ts`：会话列表（P2 用，本 phase 仅占位 actions）

### 2.5 UI 原子组件（FR-010）

- [X] T032 [P] 写入 `web/src/components/ui/button.tsx`：CVA variants（primary/secondary/ghost/destructive）+ aria-label 必填（NFR-A-003）
- [X] T033 [P] 写入 `web/src/components/ui/card.tsx`：Card / CardHeader / CardTitle / CardContent / CardFooter
- [X] T034 [P] 写入 `web/src/components/ui/dialog.tsx`：Radix Dialog 包装 + ConfirmDialog（NFR-S-008 二次确认通用组件）
- [X] T035 [P] 写入 `web/src/components/ui/dropdown-menu.tsx`：Radix DropdownMenu 包装
- [X] T036 [P] 写入 `web/src/components/ui/popover.tsx`：Radix Popover 包装
- [X] T037 [P] 写入 `web/src/components/ui/scroll-area.tsx`：Radix ScrollArea 包装
- [X] T038 [P] 写入 `web/src/components/ui/tabs.tsx`：Radix Tabs 包装
- [X] T039 [P] 写入 `web/src/components/ui/badge.tsx`：CVA variants（success/warning/destructive/outline；做多/做空/已成交/已拒绝/触发中 DG-1 中文）
- [X] T040 [P] 写入 `web/src/components/ui/separator.tsx`：Radix Separator 包装
- [X] T041 [P] 写入 `web/src/components/ui/tooltip.tsx`：Radix Tooltip 包装
- [X] T042 [P] 写入 `web/src/components/ui/skeleton.tsx`：加载占位（FR-106 / NFR-A）
- [X] T043 [P] 写入 `web/src/components/ui/toast.tsx` + `toaster.tsx`：Radix Toast 包装 + 全局 Toaster 容器
- [X] T044 [P] 写入 `web/src/components/ui/error-state.tsx`：统一错误占位（图标 + i18n 文案 + 重试按钮）

### 2.6 Provider 栈与布局

- [X] T045 写入 `web/src/components/providers/theme-provider.tsx`：CSS variables 注入（暗/亮，跟随 prefers-color-scheme，可手动切换并持久化到 localStorage，FR-005）；首屏 inline script 防 FOUC（EC-6）
- [X] T046 写入 `web/src/components/providers/i18n-provider.tsx`：包装 react-i18next Provider；运行时切换不刷新（SC-007）
- [X] T047 写入 `web/src/components/layout/app-shell.tsx`：Sidebar（左侧导航 8 路由，i18n 标签 + lucide 图标）+ TopBar（主题切换 / 语言切换 / API 状态指示器）+ Main（`<Outlet />`）；屏宽 < 1024px 显示"桌面专用"提示（EC-8 / NFR-C-002）
- [X] T048 写入 `web/src/components/error-boundary.tsx`：全局 ErrorBoundary（NFR-O-001）+ trace_id 显示 + 重试按钮
- [X] T049 写入 `web/src/components/route-skeleton.tsx`：路由级 Suspense fallback（避免空白闪烁，contracts/ui-routes.md §10）
- [X] T050 重写 `web/src/main.tsx`：Provider 栈组装顺序（FR-004）`QueryClientProvider → BrowserRouter → ThemeProvider → I18nProvider → App + Toaster`
- [X] T051 写入 `web/src/App.tsx`：8 路由 + lazy imports（contracts/ui-routes.md §12 lazy 拆分）；ErrorBoundary 包 Suspense 包 Routes；7 路由全部占位 `<RoutePlaceholder />` + `*` 走 NotFound
- [X] T052 写入 `web/src/pages/not-found.tsx`：404 页（不 lazy 走主 bundle）

### 2.7 共享组件：图表、决策详情、Web Vitals

- [X] T053 [P] 写入 `web/src/components/charts/equity-chart.tsx`：lightweight-charts 封装（折线图 + range 切换 + 暗/亮主题感知）；> 5k 点降级蜡烛图聚合（EC-11）
- [X] T054 [P] 写入 `web/src/components/charts/trend-chart.tsx`：lightweight-charts 封装（多曲线，4 条 P50/P95，给 Metrics 用）
- [X] T055 [P] 写入 `web/src/lib/web-vitals.ts`：dev 模式 console 输出 LCP/INP/CLS（NFR-O-003）；在 main.tsx 调用 `reportWebVitals`

### 2.8 后端 P1 endpoint 全部就绪（contracts/http-endpoints.md §10 优先级表）

> Phase 2 必须把 5 个 P1 页面用到的 endpoint 全部就绪，前端集成才能并行。后端遵循 A-7：先 pytest 写测试，再实现 endpoint，再 curl 验通。

- [X] T056 [P] 写入后端测试 `tests/test_api_portfolio_snapshot.py`：覆盖 `GET /api/portfolio/snapshot`（FR-800）happy path + 503（交易所不可达）
- [X] T057 [P] 写入后端测试 `tests/test_api_portfolio_equity_curve.py`：覆盖 `GET /api/portfolio/equity-curve?range=24h|7d|30d|all`（FR-801）+ 400 非法 range
- [X] T058 [P] 写入后端测试 `tests/test_api_decisions_list.py`：覆盖 `GET /api/decisions?pair=&from=&to=&page=&size=`（FR-803）含 pagination
- [X] T059 [P] 写入后端测试 `tests/test_api_decisions_detail.py`：覆盖 `GET /api/decisions/{commit_hash}`（FR-804）+ 404
- [X] T060 [P] 写入后端测试 `tests/test_api_backtest_run.py`：覆盖 `POST /api/backtest/run`（FR-805）202 + 400 参数校验
- [X] T061 [P] 写入后端测试 `tests/test_api_backtest_status.py`：覆盖 `GET /api/backtest/runs/{run_id}` running/completed + `DELETE` 取消（FR-302/805）+ 409 已结束
- [X] T062 [P] 写入后端测试 `tests/test_api_backtest_sessions.py`：覆盖 `GET /api/backtest/sessions` 与 `GET /api/backtest/sessions/{name}`（FR-806）+ 404
- [X] T063 [P] 写入后端测试 `tests/test_api_risk.py`：覆盖 `GET /api/risk/status` + `POST /api/risk/circuit-breaker/reset`（FR-807）+ 409 inactive + 503 Redis 不可达
- [X] T064 在 `src/api/routes/portfolio.py` 增 `GET /api/portfolio/snapshot` 与 `GET /api/portfolio/equity-curve`（FR-800/801）；返回 data-model Portfolio / EquityCurve；run T056/T057 → green
- [X] T065 新建 `src/api/routes/decisions.py` 实现 `GET /api/decisions` + `GET /api/decisions/{commit_hash}`（FR-803/804）；从既有 journal store 读取 + 关联 ExperienceMemory；在 `src/api/main.py` 注册 router；run T058/T059 → green
- [X] T066 新建 `src/api/routes/backtest.py` 实现 `POST /api/backtest/run` + `GET /api/backtest/runs/{run_id}` + `DELETE /api/backtest/runs/{run_id}` + `GET /api/backtest/sessions` + `GET /api/backtest/sessions/{name}`（FR-805/806）；后台任务用既有 `BacktestEngine` + TaskRegistry；在 `src/api/main.py` 注册 router；run T060/T061/T062 → green
- [X] T067 新建 `src/api/routes/risk.py` 实现 `GET /api/risk/status` + `POST /api/risk/circuit-breaker/reset`（FR-807）；复用既有 risk state；从 portfolio.py 迁移既有 `/api/risk/status`；在 `src/api/main.py` 注册并移除 portfolio.py 中重复路由；run T063 → green
- [X] T068 验证 `GET /api/scheduler/status`（FR-802）与 `GET /api/metrics/summary`（FR-808）已满足 contracts；如缺字段补齐 + 加 pytest
- [X] T069 在 `src/api/main.py` 配置 dev 模式 CORS 允许 `http://localhost:5173`（FR-009 / Q3 排查）
- [X] T070 docker compose 起全栈后用 quickstart §2.2 的 4 条 curl 命令验通所有 P1 endpoint（A-7）
- [X] T071 `uv run pytest -q` 全绿；`uv run pytest --cov=src --cov-report=term-missing` 覆盖率 ≥ 70%（quickstart §5.3）

### 2.9 路由占位与 e2e 框架

- [X] T072 在 `web/src/pages/` 下为 7 个业务路由建立占位 `index.tsx`（dashboard / decisions / backtest / risk / metrics / chat / market）+ 标题 + i18n key（quickstart §4 Phase 1 DoD）
- [X] T073 [P] 写入 `web/tests/e2e/streamlit-removal.spec.ts`：执行 `rg -i streamlit ...` 4 条命令（FR-915）；本期先 expect-fail（streamlit 还在），Phase 8 后会变绿
- [X] T074 [P] 写入 `web/tests/unit/setup.ts`：vitest setup（@testing-library/jest-dom 注入 + i18n test mode + matchMedia mock）

**Checkpoint**: `pnpm dev` 看到 7 占位路由 + 主题/语言切换有效；`pnpm typecheck && pnpm lint && pnpm test` 全绿；后端 pytest 全绿且 P1 所有 endpoint 可 curl 验通。**所有 5 个 P1 用户故事可并行启动**。

---

## Phase 3: User Story 1 — 自动交易监控（Dashboard，P1 🎯 MVP）

**Goal**: 替换 Streamlit Overview。运维者打开 `/` 看到 4 个 metric 卡 + 权益曲线 + 持仓表 + 调度器卡，10 秒自动刷新。

**Independent Test**: 在已运行后端环境下访问 `http://localhost:5173/`，2 秒内看到完整内容；停留 10 秒后数字自动刷新。

### 测试（先写）

- [X] T075 [P] [US1] 写入 `web/tests/e2e/dashboard.spec.ts`（Playwright）：4 卡渲染 / 10s 自动刷新 / Redis down 警告 / range 切换（acceptance scenario 1-4）
- [X] T076 [P] [US1] 写入 `web/src/pages/dashboard/__tests__/dashboard.test.tsx`（Vitest + RTL）：组件渲染、loading skeleton、error fallback

### 实现

- [X] T077 [P] [US1] 写入 `web/src/hooks/use-portfolio-snapshot.ts`：React Query hook；`refetchInterval: 10_000`（FR-105）；zod schema 解析；返回 `Portfolio`
- [X] T078 [P] [US1] 写入 `web/src/hooks/use-equity-curve.ts`：React Query hook；接收 `range: '24h'|'7d'|'30d'|'all'`；缓存键 `[equity-curve, range]`
- [X] T079 [P] [US1] 写入 `web/src/hooks/use-scheduler-status.ts`：React Query hook；`refetchInterval: 10_000`；处理 `redis_available: false`（FR-104）
- [X] T080 [US1] 写入 `web/src/pages/dashboard/components/metric-card.tsx`：单个指标卡（label / value / delta / trend icon ↑↓ NFR-A-002）
- [X] T081 [US1] 写入 `web/src/pages/dashboard/components/metric-cards-row.tsx`：4 卡布局（总权益 / 可用现金 / 24h PnL / 当前回撤 FR-101）；金额走 `formatCurrency`（DG-2）
- [X] T082 [US1] 写入 `web/src/pages/dashboard/components/positions-table.tsx`：持仓表（FR-103，按未实现 PnL 排序，做多/做空 Badge DG-1）
- [X] T083 [US1] 写入 `web/src/pages/dashboard/components/scheduler-card.tsx`：调度器状态卡（FR-104，下次触发币对 + 倒计时；Redis 不可用 ErrorState 警告）
- [X] T084 [US1] 写入 `web/src/pages/dashboard/components/equity-chart-section.tsx`：复用 `EquityChart` + range 切换器（24h/7d/30d/all FR-102）
- [X] T085 [US1] 写入 `web/src/pages/dashboard/index.tsx`：组装 `MetricCardsRow` + `EquityChartSection` + `PositionsTable` + `SchedulerCard`；Suspense + ErrorBoundary 包整页；FR-106 加载 skeleton + 错误 toast
- [X] T086 [US1] 在 `web/src/locales/{zh-CN,en-US}/dashboard.json` 补齐所有文案 key
- [X] T087 [US1] 跑 `pnpm test web/src/pages/dashboard` 单测全绿；docker compose 起全栈 → `pnpm test:e2e --grep dashboard` 全绿（DoD：quickstart §4 Phase 3）

**Checkpoint**: User Story 1 端到端可用，独立可演示（MVP）。

---

## Phase 4: User Story 2 — 决策复盘（Decisions，P1）

**Goal**: 替换 Streamlit Live Decisions。研究员按币对/日期筛选历史决策，点击行展开右侧详情面板（8 节）；URL 同步选中。

**Independent Test**: 访问 `/decisions`，按 BTC/USDT 筛选，点击第一行，详情面板 8 节全部有内容。

### 测试（先写）

- [X] T088 [P] [US2] 写入 `web/tests/e2e/decisions.spec.ts`：筛选 / 分页 / 选中同步 URL / 详情 8 节渲染 / trace_id 链接（acceptance 1-4）
- [X] T089 [P] [US2] 写入 `web/src/pages/decisions/__tests__/decisions.test.tsx`：组件单测

### 实现

- [X] T090 [P] [US2] 写入 `web/src/hooks/use-decisions.ts`：React Query hook 列表（接收 filter + pagination；keepPreviousData）
- [X] T091 [P] [US2] 写入 `web/src/hooks/use-decision-detail.ts`：React Query hook 详情（接收 commit_hash；enabled 条件）
- [X] T092 [US2] 写入 `web/src/components/decision-detail/node-timeline-pipeline.tsx`：水平 stepper（节点序列 + 耗时，FR-204）
- [X] T093 [US2] 写入 `web/src/components/decision-detail/agent-analysis-grid.tsx`：4 卡（NewsAgent/MacroAgent/SentimentAgent/TechnicalAgent，FR-205，含 react-markdown + rehype-sanitize NFR-S-004）
- [X] T094 [US2] 写入 `web/src/components/decision-detail/experience-memory-section.tsx`：3 折叠组（success_patterns / forbidden_zones / strategic_insights，FR-206）
- [X] T095 [US2] 写入 `web/src/components/decision-detail/debate-section.tsx`：Bull/Bear 双轮气泡消息流（FR-207）
- [X] T096 [US2] 写入 `web/src/components/decision-detail/verdict-card.tsx`：action/size/confidence/reasoning + weighted-downgrade 徽章（FR-208）
- [X] T097 [US2] 写入 `web/src/components/decision-detail/risk-gate-section.tsx`：通过 ✓ / 拒绝 ✗ 序列 + 拒绝原因（FR-209）
- [X] T098 [US2] 写入 `web/src/components/decision-detail/execution-section.tsx`：order id / status / fill price / 手续费 / 滑点（FR-210）
- [X] T099 [US2] 写入 `web/src/components/decision-detail/decision-detail-panel.tsx`：组装 8 个 section + 头部（时间戳 / 币对 / 价格 / trace_id Jaeger 链接 NFR-O-005）
- [X] T100 [US2] 写入 `web/src/pages/decisions/components/decisions-filter-bar.tsx`：币对下拉 + 日期范围 + 分页器（FR-201）
- [X] T101 [US2] 写入 `web/src/pages/decisions/components/decisions-table.tsx`：列表行（FR-202）+ 选中态 + 点击触发 onSelect
- [X] T102 [US2] 写入 `web/src/pages/decisions/index.tsx`：split view 组装；`useSearchParams` 同步 `?id=`（FR-211 + acceptance 3）
- [X] T103 [US2] 在 `web/src/locales/{zh-CN,en-US}/decisions.json` 补齐文案
- [X] T104 [US2] 跑 `pnpm test:e2e --grep decisions` 全绿（DoD：quickstart §4 Phase 4）

**Checkpoint**: User Story 1 + 2 都独立可用。

---

## Phase 5: User Story 3 — 回测新会话（Backtest，P1）

**Goal**: 新建回测表单提交 → 5s 轮询 → 完成后展示 5 项 metric + 权益曲线 + 决策时间线；可加载历史会话。

**Independent Test**: 提交合法表单 → 看到进度卡 → 完成后展示结果；保存的会话可重新加载。

### 测试（先写）

- [X] T105 [P] [US3] 写入 `web/tests/e2e/backtest.spec.ts`：提交→轮询→结果 / 取消任务 / 加载历史会话 / pure-rules 模式隐藏 Agent 区（acceptance 1-5）
- [X] T106 [P] [US3] 写入 `web/src/pages/backtest/__tests__/backtest.test.tsx`：表单 zod 校验单测

### 实现

- [X] T107 [P] [US3] 写入 `web/src/hooks/use-backtest-run.ts`：React Query hook；`refetchInterval: 5_000`；状态 completed/failed/canceled 时停止
- [X] T108 [P] [US3] 写入 `web/src/hooks/use-backtest-sessions.ts` + `use-backtest-session.ts`：列表 + 详情
- [X] T109 [P] [US3] 写入 `web/src/hooks/use-cancel-backtest.ts`：useMutation 包 DELETE，confirm dialog 二次确认（NFR-S-008 / EC-4）
- [X] T110 [US3] 写入 `web/src/pages/backtest/components/new-backtest-form.tsx`：react-hook-form + zod schema（开始/结束日期 / 币对 / 初始资金 ≥ 100 / 模式 rules|llm / 可选 session_name FR-301）
- [X] T111 [US3] 写入 `web/src/pages/backtest/components/backtest-progress-card.tsx`：进度条 + 已运行时长 + 取消按钮（FR-302 / EC-4）
- [X] T112 [US3] 写入 `web/src/pages/backtest/components/historical-sessions.tsx`：会话名下拉 + 加载（FR-303）
- [X] T113 [US3] 写入 `web/src/pages/backtest/components/decision-timeline-table.tsx`：决策时间线（仅 LLM 模式 FR-306）；点击行调用 `<DecisionDetailPanel>`（复用 Phase 4 组件）
- [X] T114 [US3] 写入 `web/src/pages/backtest/components/backtest-result-view.tsx`：5 metric 卡（FR-304）+ EquityChart 复用（FR-305）+ DecisionTimelineTable；pure-rules 时隐藏 Agent 区（FR-307）
- [X] T115 [US3] 写入 `web/src/pages/backtest/index.tsx`：Tabs（新建 / 历史）+ 进度卡 + 结果视图；`?run_id=` / `?session=` URL 参数同步
- [X] T116 [US3] 在 `web/src/locales/{zh-CN,en-US}/backtest.json` 补齐文案
- [X] T117 [US3] 跑 `pnpm test:e2e --grep backtest` 全绿（DoD：quickstart §4 Phase 5）

**Checkpoint**: 3 个 P1 用户故事独立可用。

---

## Phase 6: User Story 4 — 风控状态管理（Risk，P1）

**Goal**: 风控页显示交易计数 / 断路器卡（ACTIVE 红 / INACTIVE 绿 + 倒计时）/ 阈值；Reset 按钮带 confirm dialog。

**Independent Test**: ACTIVE 时显示红卡 + 倒计时 → 点击 Reset → confirm → 提交 → toast + 卡转绿。

### 测试（先写）

- [X] T118 [P] [US4] 写入 `web/tests/e2e/risk.spec.ts`：ACTIVE 卡渲染 / confirm dialog / reset 成功 / Redis 不可达 warning（acceptance 1-4）
- [X] T119 [P] [US4] 写入 `web/src/pages/risk/__tests__/risk.test.tsx`：组件单测（含倒计时逻辑）

### 实现

- [X] T120 [P] [US4] 写入 `web/src/hooks/use-risk-status.ts`：React Query hook；`refetchInterval: 5_000`
- [X] T121 [P] [US4] 写入 `web/src/hooks/use-reset-circuit-breaker.ts`：useMutation；onSuccess invalidate risk-status；onError toast
- [X] T122 [US4] 写入 `web/src/pages/risk/components/trade-count-card.tsx`：小时/日交易计数（FR-401）
- [X] T123 [US4] 写入 `web/src/pages/risk/components/circuit-breaker-card.tsx`：active 红色 + 倒计时；inactive 绿色（FR-402）；倒计时 hook 内每秒 setState
- [X] T124 [US4] 写入 `web/src/pages/risk/components/risk-thresholds-table.tsx`：阈值表（FR-403）
- [X] T125 [US4] 写入 `web/src/pages/risk/components/reset-circuit-breaker-button.tsx`：复用 `<ConfirmDialog>` 二次确认（NFR-S-008）+ 触发 mutation
- [X] T126 [US4] 写入 `web/src/pages/risk/index.tsx`：组装 + Redis 不可达整页 warning banner（FR-405 / EC-2）
- [X] T127 [US4] 在 `web/src/locales/{zh-CN,en-US}/risk.json` 补齐文案
- [X] T128 [US4] 跑 `pnpm test:e2e --grep risk` 全绿（DoD：quickstart §4 Phase 6）

**Checkpoint**: 4 个 P1 用户故事独立可用。

---

## Phase 7: User Story 5 — 系统指标观测（Metrics，P1）

**Goal**: 指标页显示 Prometheus 计数器 + p50/p95 延迟 + 4 条延迟趋势曲线（IndexedDB FIFO 60 样本）。

**Independent Test**: 等 30 秒趋势曲线至少 2 个采样点；刷新浏览器历史样本仍存在。

### 测试（先写）

- [X] T129 [P] [US5] 写入 `web/tests/e2e/metrics.spec.ts`：计数器 / 延迟 / 趋势图 60 样本 FIFO（acceptance 1-4）
- [X] T130 [P] [US5] 写入 `web/src/pages/metrics/__tests__/metrics.test.tsx`：FIFO 上限单测 + IndexedDB 持久化单测

### 实现

- [X] T131 [P] [US5] 写入 `web/src/lib/metrics-store-idb.ts`：基于 `idb` 的 FIFO 60 样本存储（put / getAll / pruneToLast 60）
- [X] T132 [P] [US5] 写入 `web/src/hooks/use-metrics-summary.ts`：React Query hook；`refetchInterval: 15_000`（FR-504）
- [X] T133 [P] [US5] 写入 `web/src/hooks/use-metrics-trend.ts`：组合 query + IndexedDB；onSuccess 写入 IDB；首屏从 IDB 恢复历史样本（acceptance 4）
- [X] T134 [US5] 写入 `web/src/pages/metrics/components/counter-cards-grid.tsx`：4 卡（trades / orders_placed / orders_failed / risk_rejections FR-501）
- [X] T135 [US5] 写入 `web/src/pages/metrics/components/latency-percentiles-grid.tsx`：4 卡 pipeline/execution × P50/P95（FR-502）
- [X] T136 [US5] 写入 `web/src/pages/metrics/components/metrics-trend-chart.tsx`：复用 `<TrendChart>` 4 曲线（FR-503）
- [X] T137 [US5] 写入 `web/src/pages/metrics/index.tsx`：组装 3 区
- [X] T138 [US5] 在 `web/src/locales/{zh-CN,en-US}/metrics.json` 补齐文案
- [X] T139 [US5] 跑 `pnpm test:e2e --grep metrics` 全绿（DoD：quickstart §4 Phase 7）

**Checkpoint**: 5 个 P1 用户故事全部独立可用。**MVP 完整**。

---

## Phase 8: e2e 全绿 + Streamlit 一次性删除（FR-900~915，PR 合并硬门槛）

**Purpose**: 所有 P1 e2e 通过后，把 Streamlit 整套基础设施物理删除（A-8 单 PR 一次性合并，前置条件：5 个 P1 e2e 全绿）。**4 条 ripgrep 命令必须 0 命中**才允许合并。

⚠️ **CRITICAL**: 本 phase 启动前必须确认 Phase 3-7 所有 e2e 全部绿；任何一个 P1 故事不可用则阻塞本 phase。

### 8.1 e2e 终态校验

- [X] T140 docker compose up -d 起全栈，跑 `cd web && pnpm test:e2e` 5 个 P1 spec 全部绿（dashboard / decisions / backtest / risk / metrics）
- [X] T141 跑 `pnpm lint && pnpm typecheck && pnpm test && pnpm build` 全绿，主 bundle gzipped ≤ 300 KB（NFR-P-002）
- [X] T142 跑 Lighthouse Performance ≥ 90（quickstart §6.2）

### 8.2 物理删除（按 FR-900~913 顺序）

- [X] T143 删除 `src/dashboard/` 整个目录（含 `__init__.py` / `app.py` / `components.py` / `data_loader.py` / `_pages/` 全部子文件 FR-900）
- [X] T144 删除 streamlit 相关测试：`tests/test_dashboard*.py` + `tests/test_live_decisions_page.py` + 任何 import streamlit 的测试文件（FR-901）
- [X] T145 [P] 编辑 `pyproject.toml`：移除 `streamlit>=1.55` 依赖（FR-902）；执行 `uv lock --upgrade` 重新生成 lock；确认 lock 中无 streamlit / altair / pydeck / watchdog（若仅 streamlit 引入）
- [X] T146 [P] 编辑 `docker-compose.yml`：移除 `dashboard` service 定义（端口 8501、build context、env_file、volumes、depends_on FR-903）
- [X] T147 [P] 编辑 `src/cli/main.py`：移除 `dashboard` 命令；新增 `arena web` 命令（启动 vite preview 或打印部署说明 FR-904）
- [X] T148 [P] 编辑 `docker-compose.yml`：新增 `web` service，build context `./web`，nginx 暴露 5173 → 80，depends_on api（FR-905）
- [X] T149 [P] 编辑根 `README.md`：移除 Streamlit 启动说明 / 截图 / 端口 8501 引用；新增 Web Frontend 启动章节（FR-906）
- [X] T150 [P] 删除 streamlit 相关 Dockerfile 步骤、`.dockerignore` 例外项（FR-908）
- [X] T151 [P] 删除 GitHub Actions / CI 配置中针对 dashboard 的 job 或 step（FR-909）
- [X] T152 [P] 删除 `docs/**/*.md` / `CLAUDE.md` / `.kiro/**` / `brainstorm/**` / `CHANGELOG.md` 中描述当前架构的 streamlit 引用段落（FR-910）
- [X] T153 [P] 删除 `.claude/settings.local.json` 等本地配置中针对 streamlit 进程的允许列表条目（FR-911）
- [X] T154 [P] 删除 `scripts/` 目录中启动 streamlit 的 shell 脚本（FR-912）
- [X] T155 [P] 删除 `.kiro/steering/*` 中 streamlit dashboard 架构描述条目（FR-913）

### 8.3 终态校验（PR 合并硬门槛 FR-915）

- [X] T156 跑 `rg -i streamlit src/ tests/ scripts/` → 必须 0 命中（注释也不允许）
- [X] T157 跑 `rg -i streamlit pyproject.toml docker-compose.yml Dockerfile` → 必须 0 命中
- [X] T158 跑 `rg -i 'src/dashboard' src/ tests/ docs/` → 必须 0 命中
- [X] T159 跑 `rg -i ':8501' .` → 必须 0 命中（除 brainstorm/ 历史会话文件外，由 streamlit-removal.spec.ts 配置 ignore）
- [X] T160 把 T156-T159 加入 CI `streamlit-removal-gate` job（quickstart §7）；任一非零命中 fail
- [X] T161 重跑 `web/tests/e2e/streamlit-removal.spec.ts`（T073）现在应转为 GREEN
- [X] T162 重跑 `uv run pytest -q` 全绿（streamlit 测试文件已删，覆盖率 ≥ 70%）
- [X] T163 重跑 docker compose up -d 验证全栈起来（postgres + redis + api + scheduler + web）；首次访问 `http://localhost:5173/` 看到 Dashboard（SC-009）
- [X] T164 单 PR 提交所有 Phase 8 改动；标题含 `BREAKING: 弃用 Streamlit dashboard`；reviewer focus on FR-915 校验输出

**Checkpoint**: Streamlit 痕迹清零（SC-008）；Web Frontend 完全替代；MVP 阶段闭合。

---

## Phase 9: User Story 6 — Ad-hoc AI 对话分析（ChatAgent，P2）

**Goal**: SSE 流式多代理对话 + inline widget 沙盒 iframe；`useChatMessages` ≤ 500 行硬限。

**Independent Test**: 访问 `/chat` 发送一条消息，观察至少 5 类 SSE 事件 + iframe 安全渲染。

### 测试（先写）

- [X] T165 [P] [US6] 写入 `web/tests/e2e/chat.spec.ts`：5 种 SSE 事件 / iframe sandbox / 断连重连（acceptance 1-3）
- [X] T166 [P] [US6] 写入 `web/src/pages/chat/__tests__/use-chat-messages.test.ts`：5 事件类型解析单测 + AbortController 取消单测

### 后端

- [X] T167 [P] [US6] 写入后端测试 `tests/test_api_chat_stream.py`：覆盖 `POST /api/chat/stream`（FR-809）SSE 5 事件 + done + error
- [X] T168 新建 `src/api/routes/chat.py` 实现 `POST /api/chat/stream`：触发 `build_trading_graph` 或专用 chat graph，输出适配为 5 种 SSE 事件 + done；支持 AbortController（contracts/sse-events.md §7）；run T167 → green

### 前端

- [X] T169 [P] [US6] 写入 `web/src/hooks/use-chat-messages.ts`：≤ 500 行硬限（NFR-M-007）；处理 5 种事件 + done + error；支持 AbortController；持久化到 IndexedDB（FR-604）
- [X] T170 [US6] 写入 `web/src/components/inline-widget/inline-widget.tsx`：iframe `sandbox="allow-scripts"`（**不带** allow-same-origin NFR-S-003）；注入 24+ CSS 主题变量；JSON.parse NaN/Infinity patch（FR-603）
- [X] T171 [US6] 写入 `web/src/pages/chat/components/session-list.tsx`：左栏会话列表（zustand chat store）
- [X] T172 [US6] 写入 `web/src/pages/chat/components/message-stream.tsx`：消息流渲染 + 内联 `<InlineWidget>`
- [X] T173 [US6] 写入 `web/src/pages/chat/components/chat-input.tsx`：多行输入 + Cmd+Enter 提交 + 模型选择下拉（FR-601）
- [X] T174 [US6] 写入 `web/src/pages/chat/index.tsx`：三栏布局（FR-600）+ `?session=` URL 参数同步
- [X] T175 [US6] 在 `web/src/locales/{zh-CN,en-US}/chat.json` 补齐文案
- [X] T176 [US6] 跑 `pnpm test:e2e --grep chat` 全绿（DoD：quickstart §4 Phase 9）；确认 `useChatMessages` 行数 `wc -l web/src/hooks/use-chat-messages.ts` ≤ 500（NFR-M-007 硬限）

**Checkpoint**: ChatAgent P2 端到端可用。

---

## Phase 10: User Story 7 — 加密货币市场看板（MarketView，P2）

**Goal**: TradingView Widget + 资金费率/OI/清算热图侧栏；Binance/OKX 切换。

**Independent Test**: 访问 `/market` 输入 ETH/USDT，TradingView 渲染图表 + 侧栏指标。

### 测试（先写）

- [X] T177 [P] [US7] 写入 `web/tests/e2e/market.spec.ts`：TradingView 加载 / 资金费率渲染 / 交易所切换 / 降级到 lightweight-charts（acceptance 1-3 / EC-12）
- [X] T178 [P] [US7] 写入 `web/src/pages/market/__tests__/market.test.tsx`：组件单测

### 后端

- [X] T179 [P] [US7] 写入后端测试 `tests/test_api_market.py`：覆盖 `GET /api/market/{pair}/funding-rate` + `/open-interest` + `/liquidations`（FR-810）
- [X] T180 新建 `src/api/routes/market.py` 实现 3 个 endpoint；复用既有 binance/okx market data clients；URL safe pair（`-` 替 `/`）；run T179 → green

### 前端

- [X] T181 [P] [US7] 写入 `web/src/hooks/use-market-snapshot.ts`：组合 funding-rate / open-interest / liquidations 3 query
- [X] T182 [US7] 写入 `web/src/pages/market/components/tradingview-widget.tsx`：TradingView Advanced Chart Widget（iframe 懒加载 NFR-P-010）；CSP 拦截降级到 `<EquityChart>`（EC-12）
- [X] T183 [US7] 写入 `web/src/pages/market/components/funding-rate-panel.tsx`：24h 资金费率（FR-701）
- [X] T184 [US7] 写入 `web/src/pages/market/components/open-interest-panel.tsx`：未平仓量（FR-701）
- [X] T185 [US7] 写入 `web/src/pages/market/components/liquidations-heatmap.tsx`：清算热图（FR-701）
- [X] T186 [US7] 写入 `web/src/pages/market/components/exchange-switcher.tsx`：Binance / OKX（FR-702）
- [X] T187 [US7] 写入 `web/src/pages/market/index.tsx`：组装；`?pair=` / `?exchange=` URL 同步
- [X] T188 [US7] 在 `web/src/locales/{zh-CN,en-US}/market.json` 补齐文案
- [X] T189 [US7] 跑 `pnpm test:e2e --grep market` 全绿（DoD：quickstart §4 Phase 10）

**Checkpoint**: 全部 7 个用户故事独立可用。

---

## Phase 11: 部署文档与 Polish（NFR-D / Cross-cutting）

**Purpose**: 文档收尾 + 性能/安全 polish + 跨故事关注点。

- [X] T190 [P] 写入 `web/README.md`（NFR-D-001）：开发启动 / 环境变量 / 构建 / Docker 部署 / 故障排查（quickstart §8 FAQ）
- [X] T191 [P] 写入 `docs/frontend-architecture.md`（NFR-D-003）：Provider 栈图 / 路由表 / 数据合约引用 / 状态管理图（Zustand + React Query 分工）
- [X] T192 [P] 更新根 `README.md` 的 Web Frontend 章节（NFR-D-002）：单条 `docker compose up -d` 拉起全栈说明
- [X] T193 [P] 更新 `MEMORY.md` 与 `architecture-review.md`：反映新前端架构（FR-914）
- [X] T194 跑 Lighthouse Performance ≥ 90 + Accessibility ≥ 90（NFR-A 全套：键盘 / 对比度 / aria-label）
- [X] T195 [P] 验证主 bundle gzipped ≤ 300 KB（`pnpm build && pnpm exec vite-bundle-visualizer`，NFR-P-002 / SC-009）
- [X] T196 [P] 验证权益曲线 1k 点渲染 ≤ 200 ms（Performance.now() 标记，NFR-P-004）
- [X] T197 [P] 验证 Metrics 趋势图 60 样本 4 曲线渲染 ≤ 150 ms（NFR-P-007）
- [X] T198 [P] 验证决策详情 P95 ≤ 800 ms（network panel + e2e Performance API，NFR-P-005）
- [X] T199 安全 review：所有 fetch 调用走 api-client.ts（X-API-Key NFR-S-001）；所有 markdown 走 rehype-sanitize（NFR-S-004）；所有 iframe sandbox 不带 allow-same-origin（NFR-S-003）；nginx security headers 验证（NFR-S-006）
- [X] T200 i18n 完整度：跑 `pnpm exec i18next-parser` 检测缺失键；en-US 翻译补齐
- [X] T201 a11y 全量：每个交互元素 aria-label（NFR-A-003）；键盘 Tab 顺序覆盖 7 路由（NFR-A-001）
- [X] T202 跑 quickstart §10 完整命令链验证：`/speckit-tasks` 已生成（本任务）；下一步触发 `/spex:review-plan` 生成 REVIEW-PLAN.md
- [X] T203 跑全套 CI：`pnpm lint && pnpm typecheck && pnpm test && pnpm test:e2e && pnpm build` 全绿；后端 `uv run pytest --cov=src --cov-report=term-missing` 覆盖率 ≥ 70%
- [X] T204 单条命令验证：`docker compose up -d`；浏览器访问 `http://localhost:5173/` 看到 Dashboard（SC-009 终态）

**Checkpoint**: 全部 SC-001~012 验收通过；交付完成。

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1（Setup）**：无依赖，立即开始
- **Phase 2（Foundational）**：依赖 Phase 1；**BLOCKS Phase 3-10 全部用户故事**
- **Phase 3（US1 Dashboard P1）**：依赖 Phase 2
- **Phase 4（US2 Decisions P1）**：依赖 Phase 2；**T099 决策详情组件被 Phase 5 复用**
- **Phase 5（US3 Backtest P1）**：依赖 Phase 2 + Phase 4 的 `<DecisionDetailPanel>`（T099）+ Phase 3 的 `<EquityChart>`（T053 共享组件）
- **Phase 6（US4 Risk P1）**：依赖 Phase 2
- **Phase 7（US5 Metrics P1）**：依赖 Phase 2 + `<TrendChart>`（T054 共享组件）
- **Phase 8（Streamlit Removal）**：**硬依赖 Phase 3-7 全部 e2e 绿**（A-8 + FR-907）
- **Phase 9（US6 ChatAgent P2）**：依赖 Phase 2；可与 Phase 8 并行（独立分支）
- **Phase 10（US7 MarketView P2）**：依赖 Phase 2；可与 Phase 8/9 并行
- **Phase 11（Polish）**：依赖所有期望的故事完成

### User Story Dependencies

- **US1（Dashboard）**：纯独立
- **US2（Decisions）**：纯独立；产出的 `decision-detail/*` 组件供 US3 复用（解耦：先 US2 后 US5）
- **US3（Backtest）**：依赖 US2 的 `<DecisionDetailPanel>`（FR-306 决策时间线点击展开）+ US1 的 `<EquityChart>`（FR-305 复用）
- **US4（Risk）**：纯独立
- **US5（Metrics）**：纯独立
- **US6（Chat P2）**：依赖 US2 的 `<VerdictCard>` 复用（contracts/sse-events.md §2.5）
- **US7（Market P2）**：纯独立

### Within Each User Story

- 测试先写（标记 `[ ]`，确保 fail）→ Hooks → Components → Pages → e2e 绿（DoD）
- 后端 endpoint 先 pytest 红 → 实现 → curl 验通 → 前端 hook 才能写（A-7）

### Parallel Opportunities

- **Phase 1**：T004-T015 全部 `[P]`（不同文件）
- **Phase 2.1-2.7**：`[P]` 标注的任务可全并行（types / lib / locales / stores / UI 原子）
- **Phase 2.8 后端测试**：T056-T063 全部 `[P]`（不同测试文件）
- **Phase 3-7（5 个 P1 故事）**：Phase 2 完成后可并行（teams trait 派发）；唯一约束是 US3 等 US1 + US2 共享组件
- **Phase 9-10（P2）**：与 Phase 8 完全独立可并行
- **Phase 11**：T190-T201 大量 `[P]`

---

## Parallel Example: Phase 2 Foundational

```bash
# T018 + T019 + T020 + T021 + T022 + T023 + T024 + T025（lib + types 全并行）
Task: "写入 web/src/types/api.ts 全部 data-model interface"
Task: "写入 web/src/types/api.schema.ts zod schema"
Task: "写入 web/src/lib/env.ts 类型化环境变量"
Task: "写入 web/src/lib/api-client.ts fetch 包装"
Task: "写入 web/src/lib/stream-fetch.ts SSE 客户端"
Task: "写入 web/src/lib/query-client.ts React Query 配置"
Task: "写入 web/src/lib/cn.ts className 工具"
Task: "写入 web/src/lib/format.ts 本地化格式化"

# T032-T044（13 个 UI 原子全并行）
Task: "写入 web/src/components/ui/button.tsx"
Task: "写入 web/src/components/ui/card.tsx"
Task: "写入 web/src/components/ui/dialog.tsx"
... （13 个组件并行）
```

## Parallel Example: 5 P1 用户故事并行（Phase 2 完成后）

```bash
# 5 个并行流（teams trait 派发）
Stream A (US1 Dashboard):  T075→T087
Stream B (US2 Decisions):  T088→T104
Stream C (US3 Backtest):   T105→T117（注意：等 Stream B 的 T099 完成）
Stream D (US4 Risk):       T118→T128
Stream E (US5 Metrics):    T129→T139
```

---

## Implementation Strategy

### MVP First（5 个 P1 用户故事 + Streamlit 删除）

1. **Phase 1（Setup）**：1 天，单人；DoD：`pnpm dev` hello world
2. **Phase 2（Foundational）**：3-4 天，单人或 2 人；DoD：7 路由占位 + P1 endpoints 全绿
3. **Phase 3-7（5 个 P1 故事）**：3-5 人并行 7-10 天；DoD：5 spec.ts 全绿
4. **Phase 8（Streamlit 删除）**：单 PR 1-2 天；DoD：FR-915 4 条 ripgrep 0 命中
5. **STOP and VALIDATE**：单条 `docker compose up -d` 拉起全栈，5 个页面验通；MVP 完整 ✅

### Incremental Delivery（P2 增量）

6. **Phase 9（ChatAgent P2）**：3-5 天；可独立合 PR
7. **Phase 10（MarketView P2）**：2-3 天；可独立合 PR
8. **Phase 11（Polish）**：2-3 天；docs + perf + a11y 收尾

### Parallel Team Strategy

- 1 人后端：Phase 2.8（后端 P1 endpoints）+ Phase 9.后端 + Phase 10.后端
- 1 人基础设施：Phase 1 + Phase 2.1-2.7（types/lib/UI 原子/Provider/布局）
- 3 人页面：Phase 3-7 各取 1-2 个故事
- 1 人 QA：写 e2e + 维护 streamlit-removal-gate CI

---

## Notes

- `[P]` 标记仅在不同文件、无未完成依赖时使用
- `[Story]` 标签仅 Phase 3-10 的故事任务有；Phase 1/2/8/11 不带
- 每个用户故事的 e2e spec 是该 phase 的 DoD（quickstart §4）
- A-7 严格执行：后端 endpoint 必须先 pytest 写测试 → 实现 → curl 验通 → 前端才写 hook
- Phase 8 是单 PR 一次性合并（A-8）；任何子任务漏做导致 4 条 rg 命令非 0 命中即阻塞
- `useChatMessages` ≤ 500 行（NFR-M-007）是硬限；超过用 `lib/chat-utils.ts` 拆出（quickstart §8 Q4）
- 主 bundle ≤ 300 KB（NFR-P-002）：违反时检查 lazy 拆分（quickstart §8 Q5）；常见污染源：lightweight-charts / react-markdown 未 lazy
- 提交节奏：每完成一个 phase 至少一个 commit；Phase 8 必须独立 PR
