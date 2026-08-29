# Frontend Architecture — CryptoTrader AI

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Framework | React | 19.x |
| Build | Vite | 8.x |
| Language | TypeScript | 5.9 (strict) |
| Styling | Tailwind CSS | 3.x + CSS variables |
| Components | Radix UI | Primitives |
| State (server) | React Query | 5.x |
| State (client) | Zustand | 5.x |
| i18n | react-i18next | 15.x |
| Charts | lightweight-charts | 4.x |
| Charts (market) | TradingView Widget | CDN |
| Markdown | react-markdown + rehype-sanitize | 10.x / 6.x |

## Directory Structure

```
web/
├── index.html
├── src/
│   ├── main.tsx               # App entry
│   ├── App.tsx                # Routes (lazy-loaded)
│   ├── lib/
│   │   ├── api-client.ts      # Typed fetch with Zod validation
│   │   ├── stream-fetch.ts    # SSE client (POST-based, X-API-Key)
│   │   ├── i18n.ts            # i18next bootstrap
│   │   ├── cn.ts              # tailwind-merge + clsx
│   │   ├── env.ts             # Vite env validation
│   │   ├── format.ts          # Number/date formatting
│   │   └── metrics-history.ts # IndexedDB latency caching
│   ├── types/
│   │   ├── api.schema.ts      # Zod schemas for all API responses
│   │   └── api.ts             # z.output<> type exports
│   ├── stores/
│   │   ├── use-settings-store.ts  # API key, theme, locale
│   │   ├── use-chat-store.ts      # Chat sessions (Zustand)
│   │   └── use-ui-store.ts        # Sidebar, theme persistence
│   ├── hooks/
│   │   ├── use-portfolio.ts       # Portfolio + equity curve
│   │   ├── use-scheduler-status.ts
│   │   ├── use-decisions.ts       # Paginated decisions
│   │   ├── use-backtest.ts        # Run + poll + sessions
│   │   ├── use-risk-status.ts     # Risk + circuit breaker
│   │   ├── use-metrics-summary.ts # Counters + latency
│   │   └── use-chat-messages.ts   # SSE streaming (≤500 LOC)
│   ├── components/
│   │   ├── ui/                # Radix primitives (Card, Button, Badge, etc.)
│   │   ├── layout/            # AppShell, Sidebar, TopBar
│   │   ├── decision-detail/   # 8-section detail panel
│   │   └── inline-widget/     # Sandboxed iframe widget
│   ├── pages/
│   │   ├── dashboard/         # P1: Portfolio overview (4 metric cards: 总权益 · 可用现金 · 胜率·成交 · 总收益)
│   │   ├── decisions/         # P1: Trading decision explorer (PnL column + reject_reason from risk_gate / execution_status)
│   │   ├── backtest/          # P1: Backtest runner
│   │   ├── risk/              # P1: Risk status + circuit breaker
│   │   ├── metrics/           # P1: Prometheus metrics viewer
│   │   ├── chat/              # P2: Multi-agent chat (SSE)
│   │   └── market/            # P2: TradingView + funding/OI
│   └── locales/
│       ├── zh-CN/             # Default locale (8 namespaces)
│       └── en-US/             # Alternative locale
└── vite.config.ts
```

## Key Patterns

### API Client

All HTTP calls go through `apiClient` which validates responses with Zod schemas. The generic uses `z.ZodTypeAny` with `z.output<S>` return type to correctly resolve `.default()` fields under `exactOptionalPropertyTypes`.

### SSE Streaming

`streamFetch()` uses `fetch()` + `ReadableStream` to consume SSE from POST endpoints. This is used instead of `EventSource` because we need custom headers (`X-API-Key`). NaN/Infinity values are sanitized before JSON.parse.

### State Management

- **Server state**: React Query with `refetchInterval` for polling (10s dashboard, 5s backtest/risk, 30s market).
- **Client state**: Zustand for theme, locale, API key, chat sessions.
- **URL state**: React Router params for decisions/:commitId and chat/:sessionId.

### i18n

Namespace-per-page pattern. zh-CN is the default locale; en-US is the alternative. Locale persists to localStorage.

### Theming

CSS HSL variables with `data-theme="dark|light"` on `<html>`. Semantic tokens (`--success`, `--destructive`, `--warning`) for consistent status coloring. Theme FOUC prevented by inline script in index.html.

## Bundle Budget

- Main bundle (app code): < 100 KB gzipped
- Total gzipped (including vendor): < 300 KB
- Vendor chunk (React, React Query, Zustand, Radix, i18next): ~180 KB gzipped
- Charts chunk (lightweight-charts): ~50 KB gzipped

## Backend API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/portfolio/snapshot` | GET | Portfolio summary (incl. `total_return`, `total_return_pct`, `avg_trade_pnl`) |
| `/api/portfolio/equity-curve?range=24h\|7d\|30d\|all` | GET | Equity curve data |
| `/api/scheduler/status` | GET | Scheduler state (incl. `pair_statuses[]` with last_action / risk_passed / last_error / trace_id) |
| `/api/decisions?pair=&from=&to=&page=&size=` | GET | Paginated decisions |
| `/api/decisions/:id` | GET | Decision detail (incl. `node_timeline` + `latency_breakdown` + `debate_turns`) |
| `/api/backtest/run` | POST | Start backtest (returns `run_id`) |
| `/api/backtest/runs/:id` | GET | Poll backtest status + progress |
| `/api/backtest/sessions` | GET | Historical sessions |
| `/api/risk/status` | GET | Risk + circuit breaker (live Redis ping) |
| `/api/risk/circuit-breaker/reset` | POST | Reset circuit breaker |
| `/api/metrics/summary` | GET | Prometheus counters + p50/p95 latencies |
| `/api/chat/stream` | POST | SSE chat stream (keepalive every 30s) |
| `/api/hitl/pending` | GET | Pending human-approval requests |
| `/api/hitl/:id/respond` | POST | Approve / reject |
| `/api/market/:pair` | GET | Funding rate / OI / liquidations |
| `/api/market/:pair/ohlcv?timeframe=&limit=` | GET | OHLCV bars |
| `/api/memory/skills?agent=&limit=` | GET | Trilogy evolution skills (spec 019) |
| `/api/memory/patterns?agent=&maturity=&limit=` | GET | Evolved PatternRecord list (spec 022 — closes spec 021 T021) |
| `/api/events/heartbeat?since=&types=&limit=&cursor=` | GET | Pull-mode event stream (verdict / trade / rejection / evolution) — spec 022 |
| `/skill/<name>` | GET | External Anthropic SKILL.md fetch (cryptotrader / verdict-feed / market-intel / evolution-insights / execution-replay) |

## Auth & API Key Handling

**运行时配置模型**：

- 鉴权策略和 API 密钥属于数据库 `runtime_config`，由后端在请求时读取；前端不从环境文件取得密钥。
- API/SSE 客户端使用同源相对路径；本地 Vite 开发服务器代理到 API，不需要 `VITE_*` 配置。
- 平台凭据只在平台连接表单提交时发送到后端加密存储，前端后续只取得“是否已配置”的脱敏状态。
- 运行时未初始化时，路由直接显示初始化向导；配置 revision、冲突和校验错误由配置页面展示。

**Sourcemaps** (NFR-S-002): Production build uses `sourcemap: 'hidden'` — `.map` files emitted but bundles don't reference them.

**CORS** (NFR-S-006): Backend uses explicit `allow_methods` / `allow_headers` allowlists (no wildcards), required by `allow_credentials=true`.
