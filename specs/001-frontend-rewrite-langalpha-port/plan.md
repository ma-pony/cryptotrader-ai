# Implementation Plan: 前端重写 — LangAlpha 移植 + Crypto 化

**Branch**: `001-frontend-rewrite-langalpha-port` | **Date**: 2026-04-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-frontend-rewrite-langalpha-port/spec.md`

## Summary

弃用 Streamlit dashboard 技术栈，引入位于 `web/` 子目录的 React 19 + Vite 7 + TypeScript 5.9 SPA，**1:1 复刻** Streamlit 5 大业务页面（Dashboard / Decisions / Backtest / Risk / Metrics）+ 2 个 P2 新增页面（ChatAgent / MarketView），并把所有股票语义迁移为加密货币语义。后端 FastAPI 按 spec FR-800~810 增量补齐 endpoint；Streamlit 在最终一次性 PR 中物理删除（含代码、配置、依赖、Docker service、CLI 命令、文档），以 4 条 ripgrep 命令 0 命中作为合并硬门槛。技术路径：**前端先行、后端按需补**；测试遵循项目"少 mock"约束，e2e 走真实 docker compose 全栈。

## Technical Context

### 前端（new — `web/` 子目录）

- **Language/Version**: TypeScript 5.9 (strict mode + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`)
- **Runtime**: Node.js ≥ 20 (LTS), pnpm ≥ 10.18
- **Build**: Vite 7（dev server HMR；prod 走 `vite build` → nginx 托管）
- **UI**: React 19.2 + React Router v7 + Tailwind CSS 3 + Radix UI + class-variance-authority + lucide-react
- **State**: React Query 5（远程数据 + 缓存）+ Zustand（UI/Settings/Chat 本地状态）
- **i18n**: i18next + react-i18next（zh-CN 默认，en-US 可选）
- **图表**: lightweight-charts 4（权益/趋势曲线）+ TradingView Advanced Chart Widget（行情）
- **SSE 客户端**: 移植自 LangAlpha 的 `streamFetch`（fetch + ReadableStream，处理 429/413/404，注入 `X-API-Key`）
- **Markdown**: react-markdown + rehype-sanitize（禁原始 HTML）
- **本地存储**: idb（IndexedDB；用于 Metrics 60-sample FIFO + Chat 会话持久化）
- **表单**: react-hook-form + zod
- **测试**: Vitest（单元）+ React Testing Library（组件）+ Playwright（e2e；走真实 docker compose 全栈）
- **Lint/Format**: ESLint（`@typescript-eslint/recommended-type-checked` + `react-hooks/recommended` + `eslint-plugin-i18next` + `eslint-plugin-import`，0 错 0 警告）+ Prettier（line-length 120）+ pre-commit hook

### 后端（既有 — `src/api/`，按需扩展）

- **Language/Version**: Python 3.12+, FastAPI ≥ 0.135, Pydantic ≥ 2.12
- **现有 endpoints（已确认）**:
  - `POST /api/analyze` (analyze.py)
  - `GET /api/health` (health.py)
  - `GET /api/log`, `GET /api/{hash}` (journal.py)
  - `GET /metrics`, `GET /api/metrics/summary` (metrics.py)
  - `GET /api/portfolio`, `GET /api/risk/status` (portfolio.py)
  - `GET /api/scheduler/status` (scheduler.py)
- **需新增 endpoints**: 详见 [research.md §5](./research.md) 的 FR→endpoint 矩阵
- **存储**: Postgres（决策/经验记忆/回测会话）+ Redis（风控状态/调度器状态）+ SQLite（市场数据缓存 `~/.cryptotrader/market_data.db`）
- **测试**: pytest + pytest-asyncio + httpx；覆盖率门槛 ≥ 70%

### 部署/基础设施

- **Target Platform**: Linux server（生产）+ macOS/Linux dev；浏览器 Chrome/Edge ≥ 120 / Firefox ≥ 121 / Safari ≥ 17
- **Project Type**: web（frontend SPA + backend API + scheduled worker）
- **Container**: Docker Compose 单一编排（postgres + redis + api + scheduler + **web**），完全替代当前 dashboard service
- **Orchestration**: 单条 `docker compose up -d` 拉起全栈（SC-009 硬指标）

### 性能目标（来自 NFR-P）

- 首屏 TTI（3G Fast）≤ 2 秒
- 主 bundle gzipped ≤ 300 KB
- 路由切换 P50 ≤ 100 ms（lazy chunk 已缓存）
- 权益曲线 1000 点 ≤ 200 ms
- 决策详情数据加载 P95 ≤ 800 ms
- 指标趋势图（60 样本 × 4 曲线）≤ 150 ms
- 每页面独立 lazy chunk

### 约束（来自 NFR-S/A/M/D）

- 安全：所有 API 调用注入 `X-API-Key`；InlineWidget iframe `sandbox="allow-scripts"` 不带 `allow-same-origin`；Markdown 走 `rehype-sanitize`；nginx 注入 `X-Content-Type-Options: nosniff` / `X-Frame-Options: DENY` / `Referrer-Policy: no-referrer` / `CSP`
- 可访问性：键盘导航 + WCAG AA 对比度 + `aria-label` + 红绿色盲双通道（PnL ↑↓ 箭头）
- 可维护性：TS strict + 0 lint 警告 + `useChatMessages` ≤ 500 行 + 单组件 ≤ 300 行 + Zustand 按领域拆分 + 禁 `any`
- 文档：`web/README.md` + 根 `README.md` Web 章节 + `docs/frontend-architecture.md`

### Scale/Scope

- 8 路由（5 P1 + 2 P2 + `/`）, ~50 React 组件, ~12 后端 endpoint（含已有），~14 Zustand actions
- 单租户单用户（A-2）
- 决策列表 1k+ 行（分页 20/页）；权益曲线 1000 点；指标 4 曲线 × 60 样本

## Constitution Check

> **状态**：`.specify/memory/constitution.md` 仍是占位模板，未配置项目级原则。本节按项目既有事实硬约束（来自 `CLAUDE.md` + 长期记忆 + ruff/lint 规则）替代评估。

| 门槛 | 项目硬约束 | 本特性合规判定 |
|------|------------|----------------|
| Library-First / 单一职责 | 前端独立子目录 `web/`，与后端 `src/` 完全解耦；不污染 Python 包 | ✅ |
| Test-First | 项目偏好 TDD；前端用 Vitest + Playwright；后端 pytest + 覆盖率 ≥ 70% | ✅（每页面有先写 e2e 的纪律见 [tasks.md](./tasks.md) 待生成） |
| Integration Testing | "Mock 越少越好"；e2e 走真实 docker compose 全栈 | ✅（A-9） |
| Observability | 既有 OpenTelemetry + Prometheus；前端集成 web-vitals + trace_id 链接 | ✅ |
| Simplicity / YAGNI | 不做 SSR / PWA / 移动端 / 多用户 / 错误上报后端 | ✅（A-3 / A-4） |
| Versioning | 前端走 SemVer；本期 v0.1.0 起步；Streamlit 删除标 v1.0.0 | ✅ |
| 模块边界 (TID251) | 前后端分离天然合规；后端 endpoints 仍受 ruff TID251 + 现有 per-file-ignores 约束 | ✅ |
| 0 lint 警告 | ruff 0 警告（已有），ESLint 0 警告（NFR-M-003） | ✅ |
| Docker only 偏好 | postgres / redis / api / web 全部走 docker compose；不要求 Homebrew | ✅（D-2） |

**Gate Result**: PASS — 无需在 Complexity Tracking 中记录违规。

## Project Structure

### Documentation (this feature)

```text
specs/001-frontend-rewrite-langalpha-port/
├── plan.md                      # 本文件 (/speckit.plan 输出)
├── spec.md                      # 已存在
├── review_brief.md              # 已存在（审阅简报）
├── checklists/
│   └── requirements.md          # 已存在（spec 自检清单）
├── research.md                  # Phase 0 输出
├── data-model.md                # Phase 1 输出
├── quickstart.md                # Phase 1 输出
├── contracts/                   # Phase 1 输出
│   ├── README.md
│   ├── http-endpoints.md        # FastAPI endpoint 合约（FR-800~810）
│   ├── sse-events.md            # ChatAgent SSE 事件合约（FR-602）
│   └── ui-routes.md             # 前端路由表 + 组件契约
└── tasks.md                     # /speckit-tasks 输出（NOT 由本命令生成）
```

### Source Code (repository root)

```text
# 前端（新增）
web/
├── package.json                 # pnpm workspace，React 19.2 / Vite 7 / TS 5.9
├── pnpm-lock.yaml
├── tsconfig.json                # strict + noUncheckedIndexedAccess + exactOptionalPropertyTypes
├── vite.config.ts
├── tailwind.config.ts
├── postcss.config.cjs
├── eslint.config.ts             # flat config
├── prettier.config.js
├── playwright.config.ts
├── index.html
├── public/
│   └── favicon.svg
├── src/
│   ├── main.tsx                 # Provider 栈：QueryClient → BrowserRouter → ThemeProvider → I18nProvider → App + Toaster
│   ├── App.tsx                  # 顶层路由 + Suspense + ErrorBoundary
│   ├── pages/                   # 每页一个目录（含 index.tsx + 子组件 + tests）
│   │   ├── dashboard/           # FR-100~106，P1
│   │   ├── decisions/           # FR-200~211，P1
│   │   ├── backtest/            # FR-300~307，P1
│   │   ├── risk/                # FR-400~405，P1
│   │   ├── metrics/             # FR-500~504，P1
│   │   ├── chat/                # FR-600~604，P2
│   │   └── market/              # FR-700~702，P2
│   ├── components/              # 跨页面复用组件
│   │   ├── charts/              # EquityChart / TrendChart（lightweight-charts 封装）
│   │   ├── decision-detail/     # 8 节详情组件，被 Decisions 与 Backtest 复用
│   │   ├── inline-widget/       # ChatAgent iframe sandbox 移植
│   │   └── ui/                  # Radix 包装（Button / Dialog / Toast / Skeleton 等）
│   ├── lib/
│   │   ├── api-client.ts        # fetch 包装 + X-API-Key 注入
│   │   ├── stream-fetch.ts      # 移植自 LangAlpha
│   │   ├── query-client.ts      # React Query 配置
│   │   └── env.ts               # import.meta.env 类型化
│   ├── hooks/                   # 自定义 hook（usePortfolio / useDecisions / useChatMessages 等）
│   ├── stores/                  # Zustand：useUIStore / useChatStore / useSettingsStore
│   ├── locales/
│   │   ├── zh-CN.json           # 默认
│   │   └── en-US.json
│   ├── types/                   # 公共 TS 类型 + zod schema（*.schema.ts）
│   └── styles/
│       └── globals.css          # Tailwind + CSS variables（暗/亮主题）
├── tests/
│   ├── unit/                    # Vitest（hooks / utils / pure components）
│   └── e2e/                     # Playwright（5 P1 happy paths + 关键 edge）
│       ├── dashboard.spec.ts
│       ├── decisions.spec.ts
│       ├── backtest.spec.ts
│       ├── risk.spec.ts
│       ├── metrics.spec.ts
│       └── streamlit-removal.spec.ts  # FR-915 ripgrep 校验
├── README.md                    # NFR-D-001
├── Dockerfile                   # multi-stage：node 构建 → nginx 托管
└── nginx.conf                   # security headers (NFR-S-006) + SPA fallback

# 后端（按需扩展现有 routes/）
src/api/routes/
├── analyze.py                   # 已有
├── health.py                    # 已有
├── journal.py                   # 已有 → 改造/新增 decisions.py（FR-803~804）
├── metrics.py                   # 已有 → 满足 FR-808
├── portfolio.py                 # 已有 → 扩展 FR-800~801（equity-curve）
├── scheduler.py                 # 已有 → 满足 FR-802
├── risk.py                      # 新增（FR-807，断路器 reset）
├── backtest.py                  # 新增（FR-805~806，run + sessions）
├── chat.py                      # 新增 P2（FR-809，SSE）
└── market.py                    # 新增 P2（FR-810，funding/OI/liquidations）

# 文档与编排
docs/
└── frontend-architecture.md     # NFR-D-003

docker-compose.yml               # 移除 dashboard service，新增 web service（FR-903 + FR-905）
README.md                        # 移除 Streamlit 章节，新增 Web Frontend 章节（FR-906）

# 删除范围（FR-900~915，单 PR 一次性）
src/dashboard/                   # 整个目录
tests/test_dashboard*.py
tests/test_live_decisions_page.py
pyproject.toml                   # 移 streamlit 依赖
docker-compose.yml               # 移 dashboard service
src/cli/main.py                  # 移 dashboard 命令，新增 arena web
```

**Structure Decision**: 选用 web 应用结构（option 2）—— 前端独立子目录 `web/` 配合既有后端 `src/api/`。前后端在 docker compose 中作为独立 service 编排。前端 `web/` 子目录下自包含 `tests/` + `Dockerfile` + `nginx.conf`，避免污染 Python 包根。后端 endpoint 在既有 `src/api/routes/` 下按文件粒度增量补齐（每个新 router 单独文件，便于 review 与回滚）。

## Complexity Tracking

> 无违规需记录。本特性虽涉及 ~50 组件 + 11 实施阶段，但每项都是 spec FR 直接派生；技术栈选择直接对齐 LangAlpha（已验证），无需额外架构论证。

## Implementation Notes (post-merge, 2026-04-29)

实施阶段发现并记录的 5 条隐含架构契约，详见 [research.md §15](./research.md#15-实现阶段架构注记-2026-04-29)：

1. **§15.1 Trace registry** — `tracing._node_trace_registry` 配合 `journal._resolve_node_trace()`，让 graph 内部的 `record_trade` 节点能读到由 graph 外部 runner 累积的 node_trace。
2. **§15.2 LangGraph state 序列化** — 移除 `MemorySaver()`；`EventBus` / `RedisStateManager` 通过 `chat/runtime_registry.py` (session_id-keyed) 传递，避免 msgpack 失败。
3. **§15.3 Py 3.10 asyncio.TimeoutError 兼容性** — SSE keepalive、scheduler timeout 必须 `except asyncio.TimeoutError`（不是 `TimeoutError`）。
4. **§15.4 Risk check 必须用 return delta** — 见 [data-model.md#RiskGate](./data-model.md#riskgate) PROD-I3 契约：`CheckResult.scale_adjustment` 提议，gate 聚合 (min)，节点 return delta 写回。
5. **§15.5 技术指标无原生依赖** — `agents/_indicators.py` (纯 pandas/numpy) 替代 `pandas_ta` (依赖 talib，arm64 不兼容)。

**对应 spec.md 修订**（见 [research.md changelog](./research.md#changelog)）：
- NFR-S-001 (fail-closed AUTH_MODE)
- NFR-S-002 (hidden sourcemap + forbid baked VITE_API_KEY)
- NFR-S-005 / FR-007 (in-memory key only)
- NFR-S-006 (CORS allowlist + Redis-backed rate limiter)
- data-model RiskGate (新增 scale_adjustment)
