# Quickstart — 本特性开发与验证

**Feature**: `001-frontend-rewrite-langalpha-port`
**Date**: 2026-04-16
**Audience**: 加入此 feature 的开发者 / reviewer

本指南给出从 0 到首次跑通本特性所需的全部命令。前置条件：项目已 clone、当前分支已切到 `001-frontend-rewrite-langalpha-port`。

---

## 1. 前置依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Docker Desktop | latest | postgres + redis + api + web 全栈 |
| Node.js | ≥ 20 LTS | 前端开发 |
| pnpm | ≥ 10.18 | 前端包管理（**禁止 npm/yarn**） |
| Python | ≥ 3.12 | 后端开发 |
| uv | latest | Python 包管理 |
| ripgrep (`rg`) | latest | FR-915 终态校验 |

```bash
# 验证
docker --version && node --version && pnpm --version && python3 --version && uv --version && rg --version
```

---

## 2. 初次启动（后端先跑通）

### 2.1 拉起后端栈

```bash
# 在项目根目录
docker compose up -d postgres redis
uv sync
uv run arena serve   # 在 http://localhost:8003
```

### 2.2 验通已有 endpoint

```bash
curl -s http://localhost:8003/api/health | jq
curl -s http://localhost:8003/api/portfolio | jq
curl -s http://localhost:8003/api/scheduler/status | jq
```

预期看到 JSON 响应（不需要 X-API-Key 时为开发模式）。

---

## 3. 前端开发环境

### 3.1 创建 web/ 子目录（Phase 0 任务）

```bash
mkdir -p web && cd web
pnpm init
pnpm add react@^19.2 react-dom@^19.2
pnpm add -D vite@^7 @vitejs/plugin-react typescript@^5.9 @types/react @types/react-dom
# ... （完整依赖见 plan.md Technical Context）
```

### 3.2 启动 dev server

```bash
cd web
pnpm install
pnpm dev   # 在 http://localhost:5173
```

### 3.3 接入后端

`.env.local`：
```
VITE_API_BASE_URL=http://localhost:8003
VITE_API_KEY=                # dev 模式留空
```

打开 `http://localhost:5173/`，应能看到 Dashboard（实现完成后）。

---

## 4. 实施阶段顺序（来自 spec §11 阶段）

按顺序串行：

| Phase | 内容 | DoD |
|-------|------|-----|
| 0 | 脚手架（pnpm/Vite/TS strict/Tailwind/i18n/React Query/Zustand/streamFetch） | `pnpm dev` 看到 hello world |
| 1 | 框架（Provider 栈 / 路由 / AppShell / 主题 / 国际化） | 7 路由皆有占位页 |
| 2 | 后端 endpoints 补齐（FR-800~808） | pytest 全绿 + curl 验通 |
| 3 | Dashboard（FR-100~106） | e2e dashboard.spec.ts 全绿 |
| 4 | Decisions（FR-200~211） | e2e decisions.spec.ts 全绿 |
| 5 | Backtest（FR-300~307） | e2e backtest.spec.ts 全绿 |
| 6 | Risk（FR-400~405） | e2e risk.spec.ts 全绿 |
| 7 | Metrics（FR-500~504） | e2e metrics.spec.ts 全绿 |
| **8** | **e2e 全绿 + Streamlit 一次性删除（FR-900~915）** | **4 条 rg 命令 0 命中 + Lighthouse ≥ 90 + bundle ≤ 300KB** |
| 9 | ChatAgent（P2，FR-600~604） | iframe sandbox 验通 + ≤ 500 行 |
| 10 | MarketView（P2，FR-700~702） | TradingView 渲染验通 |
| 11 | 部署文档（NFR-D-001~004） | docker compose up -d 一条命令拉起全栈 |

---

## 5. 测试

### 5.1 前端单元 + 组件

```bash
cd web
pnpm test            # Vitest
pnpm test:coverage   # 覆盖率报告
```

### 5.2 前端 e2e（Playwright）

```bash
# 起全栈
docker compose up -d
cd web
pnpm exec playwright install chromium
pnpm test:e2e        # 跑 5 个 P1 spec
pnpm test:e2e --grep streamlit-removal  # 只跑 FR-915 校验
```

### 5.3 后端

```bash
uv run pytest -q
uv run pytest --cov=src --cov-report=term-missing  # 覆盖率 ≥ 70%
```

### 5.4 端到端 lint

```bash
# 前端
cd web
pnpm lint            # 0 警告
pnpm typecheck       # 0 错误

# 后端
ruff check .
ruff format --check .
```

---

## 6. 性能验证

### 6.1 主 bundle 大小（NFR-P-002 ≤ 300 KB gzipped）

```bash
cd web
pnpm build
pnpm exec vite-bundle-visualizer
# 或 CI 中：cat dist/stats.json | jq '.bundleSizes.main'
```

### 6.2 Lighthouse Performance ≥ 90

```bash
docker compose up -d
npx lighthouse http://localhost:5173/ \
  --output=json --output-path=./lighthouse-report.json \
  --preset=desktop --quiet
jq '.categories.performance.score' lighthouse-report.json
# 期望 ≥ 0.90
```

---

## 7. Streamlit 删除终态校验（FR-915，PR 合并硬门槛）

```bash
# 4 条命令，全部必须 0 命中（除注释中的明确历史引用）
rg -i streamlit src/ tests/ scripts/                                  # 必须为空
rg -i streamlit pyproject.toml docker-compose.yml Dockerfile           # 必须为空
rg -i 'src/dashboard' src/ tests/ docs/                                # 必须为空
rg -i ':8501' .                                                        # 必须为空（除 brainstorm/ 外）
```

CI 中 `streamlit-removal-gate` job 跑这 4 条；任一非零命中则 fail。

---

## 8. 常见问题排查

### Q1: pnpm install 报 peer dep 警告
A: 检查 React 19 兼容性，部分包用 `pnpm install --shamefully-hoist`。

### Q2: 前端连不上后端
A: 检查 `VITE_API_BASE_URL` + 后端 CORS 配置。开发模式 FastAPI 默认允许 `localhost:5173`。

### Q3: Playwright 卡在第一个测试
A: 确认 docker compose 全栈起来（`docker compose ps`），api / web 两个服务都 healthy。

### Q4: `useChatMessages` 超过 500 行
A: 拆出工具函数到 `lib/chat-utils.ts`；删去 LangAlpha 不需要的多模型路由 / collaboration 逻辑（NFR-M-007）。

### Q5: Lighthouse 分数 < 90
A: 检查 `bundle-visualizer`，看是否有未 lazy 的重型依赖（lightweight-charts、react-markdown）。

---

## 9. 工件清单（Phase 1 完成时）

- ✅ `spec.md`
- ✅ `plan.md`
- ✅ `research.md`
- ✅ `data-model.md`
- ✅ `contracts/{README,http-endpoints,sse-events,ui-routes}.md`
- ✅ `quickstart.md`（本文件）
- ✅ `review_brief.md`
- ✅ `checklists/requirements.md`
- ⏳ `tasks.md`（下一步 `/speckit-tasks` 生成）
- ⏳ `REVIEW-PLAN.md`（`/spex:review-plan` 生成）

---

## 10. 下一步

```bash
/speckit-tasks   # 生成 tasks.md
# 然后
/spex:review-plan   # 生成 REVIEW-PLAN.md
# 全部就绪后
/speckit-implement   # 进入实施
```
