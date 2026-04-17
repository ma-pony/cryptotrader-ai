# cryptotrader-web

CryptoTrader-AI 前端 — React 19 + Vite 7 + TypeScript 5.9 + Tailwind 3 SPA。

提供监控/复盘/操作三类视图。

## 快速开始

```bash
pnpm install
pnpm dev   # http://localhost:5173
```

后端默认地址 `http://localhost:8003`（通过 `arena serve` 启动）。可在 `.env.local` 中覆盖。

## 脚本

- `pnpm dev` — Vite 开发服务器
- `pnpm build` — 生产构建
- `pnpm preview` — 本地预览生产构建
- `pnpm lint` — ESLint 0 警告策略
- `pnpm typecheck` — `tsc --noEmit`
- `pnpm test` — Vitest 单元/组件测试
- `pnpm test:e2e` — Playwright 端到端测试（需先 `docker compose up -d`）

详细文档见仓库根 `docs/frontend-architecture.md`（Phase 11 添加）。
