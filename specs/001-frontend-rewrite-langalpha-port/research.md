# Phase 0 — Research & Decisions

**Feature**: `001-frontend-rewrite-langalpha-port`
**Date**: 2026-04-16
**Status**: All NEEDS CLARIFICATION resolved

本文档汇总 Phase 0 调研结论：每条决策含 **决策 / 理由 / 已考虑替代方案**。spec 中所有未决项已解决（spec 自检显示 0 个 [NEEDS CLARIFICATION]）。

---

## 1. 前端基础栈

### 1.1 React 19 vs React 18

**决策**：React 19.2（latest stable）
**理由**：
- LangAlpha 已用 React 19；移植路径最短
- 新 hooks（`use`, `useOptimistic`, `useFormStatus`）契合 SSE 流式与表单乐观更新
- `useTransition` 改进降低长列表（Decisions 1k+ 行）卡顿
**替代**：React 18（更稳）— 但失去 LangAlpha 移植优势，且与 spec 明确技术约束冲突（DG-2）

### 1.2 Vite 7 vs Webpack/Next.js

**决策**：Vite 7
**理由**：
- 与 spec 显式约束一致；与 LangAlpha 一致
- HMR 速度对 50+ 组件项目至关重要
- `vite build` 产物可直接 nginx 托管，无需 Node SSR 运行时（A-3 明确不做 SSR）
**替代**：Next.js（SSR/API routes）— A-3 明确拒绝；Webpack 5 — DX 倒退

### 1.3 TypeScript strict 选项

**决策**：strict + `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`（NFR-M-001）
**理由**：
- 项目长期标准是"0 lint 警告"；前端等价于"0 ts 错误 + 严格模式"
- `noUncheckedIndexedAccess` 避免 `arr[0]` 隐含 `undefined` bug，对决策列表分页关键
- `exactOptionalPropertyTypes` 与 zod schema 的可选字段语义对齐
**替代**：仅 strict — 与项目零警告文化不符

---

## 2. 状态管理：React Query + Zustand

### 2.1 为何不用 Redux Toolkit？

**决策**：远程数据 React Query 5；UI/Settings/Chat 本地状态 Zustand（NFR-M-008 拆分 store）
**理由**：
- 项目数据 95% 是远程 API 状态（portfolio / decisions / backtest），React Query 的缓存 + 重试 + refetchInterval 直接覆盖 FR-105（10s 自动刷新）/ FR-302（5s 轮询）
- 真正的本地状态只有：主题、语言、当前选中的决策 hash、Chat 会话——Zustand 5KB 足够
- LangAlpha 使用同栈，移植路径最短
**替代**：Redux Toolkit + RTK Query — 样板代码 ~2x；Jotai — 与项目"少模式"原则不符

### 2.2 Zustand store 拆分

**决策**：3 个独立 store（NFR-M-008）
- `useUIStore`：theme / sidebar / modal
- `useSettingsStore`：locale / API key（不持久化敏感）
- `useChatStore`：当前会话 / 消息流（P2，IndexedDB 持久化）
**理由**：领域拆分避免 monolithic store；每 store 有清晰所有权
**替代**：单一 store — 违反 NFR-M-008

---

## 3. SSE 客户端：streamFetch（移植自 LangAlpha）

### 3.1 为何不用 EventSource？

**决策**：移植 LangAlpha 的 `streamFetch`（fetch + ReadableStream）
**理由**：
- EventSource 不支持 POST、不支持自定义 header（无法注入 `X-API-Key`，违反 NFR-S-001）
- EventSource 对 4xx 错误的处理 opaque，无法区分 429/413/404
- streamFetch 已在 LangAlpha 验证多代理流式场景的稳定性
**替代**：原生 EventSource — 安全性失败；自实现 SSE — 重复造轮子

### 3.2 streamFetch 关键能力（移植清单）

- POST + JSON body
- 注入任意 header（含 `X-API-Key` / `X-Trace-Id`）
- 显式处理 429（rate limit，exponential backoff）/ 413（payload too large）/ 404（端点不存在）
- 支持 `AbortSignal` 取消（FR-302 回测取消）
- `debug: true` 开关打印所有 chunk（NFR-O-004）

---

## 4. 图表库：lightweight-charts 4 + TradingView Widget

### 4.1 权益/趋势曲线

**决策**：lightweight-charts 4
**理由**：
- 1000 点权益曲线渲染目标 ≤ 200 ms（NFR-P-004），lightweight-charts 实测可达
- 体积小（~30KB gzipped），符合主 bundle ≤ 300KB 约束（NFR-P-002）
- Tradeable charts 视觉一致性
**替代**：Recharts / Chart.js — 性能不达标；ECharts — 体积过大（~400KB）

### 4.2 加密货币行情图

**决策**：TradingView Advanced Chart Widget（外嵌 iframe）
**理由**：
- 免费版功能足够（蜡烛 + 指标 + 多周期）；A-12 明确假设可用
- 自实现成本巨大；Binance 自带的 chart 体验也是 TradingView
**替代**：纯 lightweight-charts — 缺少专业指标；自研 — 不在范围
**降级方案**：CSP / 网络受限时切换到纯 lightweight-charts（A-12 兜底）

---

## 5. 后端 endpoint 现状盘点 → FR 缺口矩阵

### 5.1 现有 endpoints（已确认存在）

| 文件 | 路由 | 用途 |
|------|------|------|
| `analyze.py` | `POST /api/analyze` | 触发 ad-hoc 决策分析 |
| `health.py` | `GET /api/health` | 健康检查 |
| `journal.py` | `GET /api/log`, `GET /api/{hash}` | 决策日志查询（旧风格，需改造） |
| `metrics.py` | `GET /metrics`, `GET /api/metrics/summary` | Prometheus 原始 + 聚合 |
| `portfolio.py` | `GET /api/portfolio`, `GET /api/risk/status` | 投资组合 + 风控状态查询 |
| `scheduler.py` | `GET /api/scheduler/status` | 调度器状态 |

### 5.2 FR → endpoint 映射 + 状态

| FR | 端点 | 现状 | 行动 |
|----|------|------|------|
| FR-800 | `GET /api/portfolio/snapshot` | 已有 `/api/portfolio` | **改造**：补 `pnl_24h` + `drawdown` 字段，路由可改名或保留 |
| FR-801 | `GET /api/portfolio/equity-curve?range=` | ❌ 不存在 | **新增** |
| FR-802 | `GET /api/scheduler/status` | ✅ 已有 | 复用 |
| FR-803 | `GET /api/decisions?pair=&from=&to=&page=&size=` | 部分（journal.py `/api/log`） | **新增 `decisions.py`**，弃用旧 `/api/log` 路由 |
| FR-804 | `GET /api/decisions/{commit_hash}` | 部分（journal.py `/api/{hash}`） | **新增**，含 ExperienceMemory join |
| FR-805 | `POST /api/backtest/run` + `GET /api/backtest/runs/{run_id}` | ❌ 不存在 | **新增 `backtest.py`**（异步 task + run table） |
| FR-806 | `GET /api/backtest/sessions`, `/{name}` | ❌ 不存在 | **新增**（基于 `backtest/session.py` 已有 store） |
| FR-807 | `GET /api/risk/status`, `POST /api/risk/circuit-breaker/reset` | 部分（portfolio.py 有 risk/status） | **新增 `risk.py`**，迁移 `risk/status` 路由，加 reset 操作 |
| FR-808 | `GET /api/metrics/summary` | ✅ 已有 | 复用 |
| FR-809（P2） | `POST /api/chat/stream`（SSE） | ❌ | **新增 `chat.py`** |
| FR-810（P2） | `GET /api/market/{pair}/funding-rate` 等 | ❌ | **新增 `market.py`** |

### 5.3 路由命名一致性整改

**决策**：所有前端面向的资源路由统一为 `/api/<resource>` 形态；旧 `/api/log` 与 `/metrics`（无 `/api` 前缀）保持向后兼容直到 Streamlit 删除 PR 一并清理
**理由**：减少前端 zod schema 维护负担；避免 SSE / REST 混用前缀

---

## 6. i18n 策略

**决策**：i18next + react-i18next；zh-CN 为默认 + 主写文件；en-US 翻译可后续补全；缺失键回退到 zh-CN
**理由**：A-13 明确假设；DG-1/2/3 给出术语与格式细则；用户为中文优先
**关键约束**：
- 翻译键命名 `<scope>.<element>.<state>`（例：`dashboard.metric.equity.label`）
- 业务术语保持 spec DG-1 约定（决策/回测/做多/做空 等）
- 时间格式走 `Intl.DateTimeFormat(locale)`，不自实现
- ESLint `eslint-plugin-i18next` 强制不允许字符串字面量（仅 placeholder）

---

## 7. 测试策略：少 mock + 真实 docker compose

**决策**：
- **单元（Vitest）**：纯函数 / hook 工具 / 无网络组件 — 占比 ~40%
- **组件（RTL）**：复杂交互组件（DecisionDetail、BacktestForm、CircuitBreakerCard）— 占比 ~30%；网络层用 `msw` 拦截（限定 schema 测试，不替代 e2e）
- **E2E（Playwright）**：5 P1 happy path + 关键 edge（Redis down、断路器 ACTIVE、回测取消）+ FR-915 streamlit 删除校验 — 占比 ~30%
- **后端 endpoint**：pytest + httpx，真实 docker compose 起 postgres + redis；覆盖率 ≥ 70%（与项目门槛一致）

**理由**：项目长期偏好 [Minimize mocks](memory:feedback-minimize-mocks.md)；E2E 用真实栈避免"mock 通过、生产失败"
**替代**：纯组件级 mock — 与项目纪律冲突

---

## 8. Streamlit 删除策略

### 8.1 单 PR 一次性删除

**决策**：所有 Streamlit 痕迹在最后一个 PR 一次性删除（D-11 + FR-907）
**理由**：
- 半路灰度会留下两套 UI 入口、两套配置，运维心智负担大
- 一次性删除可由 `rg` 命令机器校验（FR-915）
- 前置条件已明确：5 个 P1 e2e 全绿
**替代**：渐进灰度 — 拒绝（D-11）

### 8.2 删除范围 checklist

依据 FR-900~914，删除范围必须覆盖：
1. 代码：`src/dashboard/` 整目录
2. 测试：`tests/test_dashboard*.py` / `test_live_decisions_page.py`
3. 依赖：`pyproject.toml` 移 `streamlit>=1.55` + `uv lock --upgrade` 重新生成 lock 文件
4. 编排：`docker-compose.yml` 移 `dashboard` service（含端口 8501）
5. CLI：`src/cli/main.py` 移 `dashboard` 子命令，新增 `arena web`
6. Docker：所有 Dockerfile 步骤、`.dockerignore` 例外
7. CI：GitHub Actions 中 dashboard 相关 job/step
8. 文档：`docs/`、`CLAUDE.md`、`.kiro/steering/`、`brainstorm/`、`CHANGELOG.md`
9. 配置：`.claude/settings.local.json` 中 streamlit 进程允许列表
10. 脚本：`scripts/` 中启动 streamlit 的脚本

### 8.3 终态校验（FR-915）

```bash
rg -i streamlit src/ tests/ scripts/                                  # 0 命中
rg -i streamlit pyproject.toml docker-compose.yml Dockerfile           # 0 命中
rg -i 'src/dashboard' src/ tests/ docs/                                # 0 命中
rg -i ':8501' .                                                        # 0 命中（除历史 brainstorm 文件外）
```

CI 中加一个 `streamlit-removal-gate` job 跑这 4 条命令，任何一条非零命中则 fail，作为 PR 合并硬门槛。

---

## 9. ChatAgent InlineWidget（P2 关键风险）

### 9.1 iframe sandbox 严格度

**决策**：`<iframe sandbox="allow-scripts">`（不带 `allow-same-origin`）（NFR-S-003）
**理由**：
- 渲染的 widget 内容来自 LLM 生成，潜在 XSS / 数据外泄
- `allow-scripts` 单独使用时，iframe 与父页面跨源，无法读 cookies / localStorage / postMessage 受限
- LangAlpha InlineWidget 已采用同方案
**替代**：`allow-scripts allow-same-origin` — 安全风险极高，拒绝

### 9.2 useChatMessages ≤ 500 行硬限

**决策**：移植时**只保留 5 个事件类型**（`message_chunk` / `tool_call` / `tool_result` / `inline_widget` / `verdict`），删去 LangAlpha 的多模型路由 / 多用户 / collaboration 等不需要的复杂度（NFR-M-007）
**理由**：LangAlpha 原版 186KB 单文件是反面教材；本期单租户单会话场景可大幅简化
**校验**：`wc -l web/src/hooks/useChatMessages.ts` 必须 ≤ 500

---

## 10. 性能预算分配（NFR-P-002 主 bundle ≤ 300KB gzipped）

### 10.1 估算

| 模块 | gzipped 估算 | 备注 |
|------|--------------|------|
| react + react-dom 19 | ~45 KB | base |
| react-router 7 | ~10 KB | |
| @tanstack/react-query 5 | ~15 KB | |
| zustand | ~5 KB | |
| i18next + react-i18next | ~15 KB | |
| radix-ui（仅用到的 primitive） | ~25 KB | tree-shake |
| tailwindcss runtime | 0 KB | 编译时 |
| zod | ~12 KB | |
| react-hook-form | ~8 KB | |
| lucide-react（按需） | ~10 KB | tree-shake |
| 应用代码（非 lazy） | ~80 KB | Provider + Layout + 公共组件 |
| **主 bundle 合计** | **~225 KB** | < 300 KB ✅ |

### 10.2 lazy 拆分

每个 page 独立 chunk（`React.lazy`）；以下重型依赖必须 lazy：
- `lightweight-charts`（~30 KB）→ 按需挂载在 Dashboard / Backtest / Metrics
- TradingView Widget（外部 iframe）→ 进入视口才挂载（NFR-P-010）
- `react-markdown` + `rehype-sanitize`（~40 KB）→ 仅 Decisions / Chat 加载
- `idb`（~5 KB）→ Metrics / Chat 按需

---

## 11. 部署：Docker Compose web service

### 11.1 multi-stage Dockerfile

```dockerfile
# build
FROM node:20-alpine AS build
WORKDIR /app
COPY web/package.json web/pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

# serve
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

### 11.2 nginx security headers（NFR-S-006）

```nginx
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header Referrer-Policy "no-referrer" always;
add_header Content-Security-Policy "default-src 'self'; connect-src 'self' http://api:8003 https://www.tradingview.com; frame-src https://www.tradingview.com; style-src 'self' 'unsafe-inline'; script-src 'self'" always;

location / {
  try_files $uri /index.html;  # SPA fallback
}
```

### 11.3 docker-compose 集成

```yaml
services:
  web:
    build:
      context: .
      dockerfile: web/Dockerfile
    ports:
      - "5173:80"
    depends_on:
      - api
```

替代旧 `dashboard` service；同时 `dashboard` service 在 FR-900~915 PR 中物理删除。

---

## 12. 已考虑但拒绝的方案汇总

| 方案 | 拒绝原因 |
|------|---------|
| Next.js（App Router + RSC） | A-3 明确不做 SSR；项目无 SEO 需求 |
| PWA + Service Worker | A-3 明确不做离线 |
| 移动端响应式 | NFR-C-002 明确 ≥ 1280px |
| Sentry / Datadog RUM | A-4 明确不实装错误上报后端，仅预留 hook |
| OAuth / 多用户登录 | A-2 明确单租户复用 X-API-Key |
| Vue / Svelte | LangAlpha 是 React，移植成本最低 |
| Storybook | NFR-M-005 强调"扁平结构"，本期不引入 |
| GraphQL | 后端是 REST + SSE，无 BFF 需求 |

---

## 13. 风险与缓解（来自 spec 风险矩阵 R-1 ~ R-12 + 本研究新增）

| 风险 | 缓解 |
|------|------|
| Streamlit 残留漏删 | FR-915 4 条 `rg` 命令 + CI gate |
| 主 bundle 超 300 KB | §10 预算 + bundle-visualizer + lazy chunk + CI 阈值检查 |
| TradingView CSP 拒绝 | A-12 兜底降级到 lightweight-charts |
| LangAlpha 移植代码引入股票残留 | code review + `rg -i 'sec\|edgar\|earnings\|10-?[KQ]\|stock' web/` 在 e2e CI 中执行 |
| useChatMessages 失控膨胀 | pre-commit hook 检查 `wc -l ≤ 500` |
| 后端 endpoint 缺失阻塞前端 | A-7 + 每页面开发前 `curl` 验通 + Phase 2 集中补齐 |
| docker compose 启动顺序 | depends_on + healthcheck（postgres / redis 先就绪） |
| spex teams 子代理写文件冲突（R-11） | worktrees trait 隔离；任务清单分文件粒度 |

---

## 14. 决策版本

| 字段 | 值 |
|------|----|
| 决策时间 | 2026-04-16 |
| 适用 spec | `001-frontend-rewrite-langalpha-port` |
| 上次评审 | brainstorm Section 1-7 + spec self-review 19.5/20 |
| 待跟踪未决项 | 0 |

---

**Phase 0 出口**：所有 NEEDS CLARIFICATION 已解决，可进入 Phase 1（数据模型 + 接口合约）。

---

## 15. 实现阶段架构注记 (2026-04-29)

实现过程中发现的几个隐含架构契约，记录以便未来维护：

### 15.1 节点 trace 收集 — trace_id-keyed 内存 registry
**问题**：FR-204 要求决策详情渲染节点 Pipeline，需要 `node_timeline + latency_breakdown` 写入 journal commit。LangGraph 的 `record_trade` 节点在 graph 内部运行，但 trace 由 graph 外部的 runner（`run_graph_traced` 或 `analysis_runner`）逐 chunk 累积。无法通过 state delta 把累积中的列表传给后续节点（reducer 会覆盖）。

**方案**：`tracing.py` 提供 `_node_trace_registry: dict[trace_id, list[entry]]`，runner 在 `register/append/unregister` 三段式中维护；`journal.py:_resolve_node_trace(state)` 优先读 `state["data"]["node_trace"]`，回退到 `trace_get(state.metadata.trace_id)`。`finally` 清理 registry，避免内存泄漏。

### 15.2 LangGraph state 序列化禁忌
**问题**：state 中放 `EventBus` / `RedisStateManager` 等运行时对象，会让 MemorySaver checkpointer msgpack 失败 → SSE 末端 `stream_error`。

**方案**：(a) 移除非必要的 checkpointer（`build_trading_graph` 不再使用 `MemorySaver()`）；(b) 运行时对象通过 `chat/runtime_registry.py` (session_id-keyed) 传递，`state.metadata` 只存可序列化基本类型；(c) 节点用 `get_event_bus(session_id)` 从 registry 取。

### 15.3 Py 3.10 `asyncio.TimeoutError` 不是 `TimeoutError`
**问题**：Python 3.10 下 `asyncio.TimeoutError is not builtins.TimeoutError`。`except TimeoutError` 不捕获 `asyncio.wait_for` 的超时 → SSE keepalive 不触发 → 30s 后强行断连。

**方案**：所有 `await asyncio.wait_for(...)` 的超时捕获必须显式写 `except asyncio.TimeoutError`。Py 3.11+ 这两个等价，但代码必须兼容 3.10。

### 15.4 Risk check 必须用 return delta 提议 scale 调整
见 [data-model.md#RiskGate](./data-model.md#riskgate) 中 PROD-I3 契约：检查不得原地 mutate `verdict.position_scale`，必须通过 `CheckResult.scale_adjustment` 返回；`risk_check` 节点聚合 (取 min) 后通过 LangGraph return delta 写回。

### 15.5 技术指标无原生依赖
**问题**：`pandas_ta` 在调用 `rsi/macd/atr` 时尝试 `from talib import ...`，arm64 Mac 上 talib x86_64 .so dlopen 失败 → tech_agent 静默 fallback 到 mock。

**方案**：`agents/_indicators.py` 提供纯 pandas/numpy 实现的 6 个 indicator (rsi, macd, sma, bbands, atr, obv)，`tech.py` 用 `from cryptotrader.agents import _indicators as ta` 替代 `import pandas_ta`。`pyproject.toml` 移除 `pandas-ta` 依赖。

---

## Changelog
- 2026-04-29: NFR-S-001 — 鉴权默认 fail-closed (AUTH_MODE env, `secrets.compare_digest`)
- 2026-04-29: NFR-S-002 — 显式禁用生产 sourcemap (hidden 模式) + forbid `VITE_API_KEY` at build time
- 2026-04-29: NFR-S-005 — 明确 `useSettingsStore.apiKey` 仅 dev 模式 hydrate
- 2026-04-29: NFR-S-006 — CORS allowlist 强制；Redis-backed rate limiter (multi-process)
- 2026-04-29: FR-007 — 澄清 X-API-Key 仅来自 in-memory store；SSE keepalive 用 `asyncio.TimeoutError`
- 2026-04-29: data-model RiskGate — 新增 `scale_adjustment` 字段 + PROD-I3 契约
- 2026-04-29: research §15 — 实现阶段架构注记 (5 条)
