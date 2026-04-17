# Implementation Checkpoint — 2026-04-17

## Status

| Phase | Tasks | Done | Notes |
|-------|-------|------|-------|
| Phase 1 — Setup | 17 | 17 | scaffold green: dev/typecheck/lint/build all pass |
| Phase 2 — Foundational (frontend) | 41 | 41 (T018-T055, T072-T074) | typecheck 0 / lint 0 / 4 unit tests pass / build 167 KB gzipped |
| Phase 2 — Foundational (backend P1 endpoints) | 16 (T056-T071) | 0 | NEXT |
| Phase 3-7 — P1 user stories | 65 | 0 | Dashboard / Decisions / Backtest / Risk / Metrics |
| Phase 8 — e2e + Streamlit removal | 25 | 0 | one-shot PR with `rg -i streamlit` 0 hits gate |
| Phase 9-11 — P2 + Polish | 40 | 0 | ChatAgent / MarketView / docs |
| **Total** | **204** | **28** | **~14 %** |

## Frontend foundational deliverables (this session)

```
web/
  src/
    App.tsx                                lazy 8 routes + ErrorBoundary
    main.tsx                               Provider stack: QueryClient → Theme → I18n → ErrorBoundary → BrowserRouter → App + Toaster
    components/
      error-boundary.tsx                   trace_id + retry
      route-skeleton.tsx                   Suspense fallback
      providers/
        theme-provider.tsx                 system / light / dark + matchMedia + colorScheme
        i18n-provider.tsx                  react-i18next bridge
      layout/
        app-shell.tsx                      Sidebar + DesktopOnlyBanner + TopBar + <Outlet/>
        sidebar.tsx                        7 NavLinks + collapse
        top-bar.tsx                        theme/locale dropdowns + API key badge
        desktop-only-banner.tsx            < 1024px banner (EC-8)
      ui/                                  14 atoms (button/card/dialog/dropdown/popover/scroll-area/tabs/badge/separator/tooltip/skeleton/toast/toaster/error-state)
      charts/
        equity-chart.tsx                   lightweight-charts area/line + > 5k aggregation (EC-11)
        trend-chart.tsx                    multi-line for Metrics
    lib/
      env.ts                               zod-validated import.meta.env
      api-client.ts                        ApiError + zod-validated fetch wrapper, X-API-Key inject
      stream-fetch.ts                      LangAlpha streamFetch port (NaN/Infinity patch)
      query-client.ts                      React Query defaults
      i18n.ts                              16 namespace × locale resources
      cn.ts / format.ts / web-vitals.ts
    locales/{zh-CN,en-US}/{common,dashboard,decisions,backtest,risk,metrics,chat,market}.json
    pages/                                 dashboard / decisions / backtest / risk / metrics / chat / market / not-found (placeholders)
    stores/                                use-ui-store / use-settings-store / use-chat-store
    types/                                 api.ts + api.schema.ts (zod mirrors of data-model §1-§9)
  tests/
    unit/setup.ts                          jest-dom + matchMedia + ResizeObserver mocks
    unit/format.test.ts                    4 sample tests
    e2e/streamlit-removal.spec.ts          FR-915 expect-fail until Phase 8
  index.html                               FOUC-prevention bootstrap script
```

## Validation

```
$ pnpm typecheck   →  0 errors
$ pnpm lint        →  0 problems
$ pnpm test        →  Test Files 1 passed (1) | Tests 4 passed (4)
$ pnpm build       →  vendor 138 KB + radix 16 KB + index 12 KB = ~167 KB gzipped (well under 300 KB NFR-P-002)
```

## Resume here

Next stop is **Phase 2 §2.8 — Backend P1 endpoints (T056-T071)**: pytest + FastAPI routers for portfolio snapshot/equity-curve, decisions list/detail, backtest run/status/sessions, risk status/circuit-breaker reset, plus dev CORS for `:5173`. Target: full pytest green ≥ 70 % coverage, then proceed to Phase 3 (US1 Dashboard P1).
