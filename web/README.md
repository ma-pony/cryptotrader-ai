# cryptotrader-web

CryptoTrader-AI 前端 — React 19 + Vite 8 + TypeScript 5.9 + Tailwind 3 SPA。

提供监控/复盘/操作三类视图。

## 快速开始

```bash
pnpm install
pnpm dev   # http://localhost:5173
```

后端默认地址 `http://localhost:8003`（通过 `arena serve` 启动）。可在 `.env.local` 中覆盖：

```bash
# .env.local — dev only, NEVER commit, NEVER use in production builds
VITE_API_BASE_URL=http://localhost:8003
VITE_API_KEY=dev-test-key   # 仅 dev 模式启动 hydrate 进 useSettingsStore
```

## 脚本

- `pnpm dev` — Vite 开发服务器
- `pnpm build` — 生产构建（**会拒绝 `VITE_API_KEY` 非空**，参见 NFR-S-002）
- `pnpm preview` — 本地预览生产构建
- `pnpm lint` — ESLint 0 警告策略
- `pnpm typecheck` — `tsc --noEmit`
- `pnpm test` — Vitest 单元/组件测试
- `pnpm test:e2e` — Playwright 端到端测试（需先 `docker compose up -d`）

## 关键约束

- **API key 仅运行时输入**：生产 bundle 永不打入 key（`vite.config.ts` 里 `forbid-baked-api-key` 插件强制）。用户在 Settings UI 输 key，存 `useSettingsStore`（in-memory only）。
- **生产 sourcemap = `hidden`**：`.map` 文件可上传到私有 error tracker，但 JS bundle 不引用它，防止源码泄漏。
- **TypeScript strict 模式**：`exactOptionalPropertyTypes` + `noUncheckedIndexedAccess` + `verbatimModuleSyntax` 全开。
- **`useChatMessages` ≤ 500 行硬限**（NFR-M-007，目前 ~280 行）。
- **主 bundle ≤ 300 KB gzipped**（NFR-P-002，目前 ~220 KB）。

详细架构见仓库根 [`docs/frontend-architecture.md`](../docs/frontend-architecture.md)。
