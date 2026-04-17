# HTTP Endpoints Contract

**Feature**: `001-frontend-rewrite-langalpha-port`
**Date**: 2026-04-16
**Source FRs**: FR-800 ~ FR-810
**Auth**: 所有 endpoint 走 `X-API-Key` header（dev 模式可空）

> **响应数据形态参考** [data-model.md](../data-model.md)。本文档只罗列路径/参数/状态码与示例。

---

## 1. Portfolio 域（Dashboard 用）

### GET `/api/portfolio/snapshot` — FR-800

**说明**：返回当前投资组合快照（替代既有 `/api/portfolio`，补 `pnl_24h` + `drawdown`）。

**Query 参数**：无
**Response 200**：`Portfolio`
**示例**：
```json
{
  "equity": 10523.45,
  "cash": 2103.10,
  "positions": [
    {
      "pair": "BTC/USDT",
      "side": "long",
      "size": 0.05,
      "avg_price": 65000.0,
      "unrealized_pnl": 152.30,
      "unrealized_pnl_pct": 0.0469,
      "opened_at": "2026-04-16T10:00:00Z"
    }
  ],
  "pnl_24h": 215.80,
  "pnl_24h_pct": 0.0209,
  "drawdown": 0.012,
  "updated_at": "2026-04-16T13:30:00Z"
}
```
**错误**：
- `503 ApiError` — 交易所 API 不可达

---

### GET `/api/portfolio/equity-curve` — FR-801

**Query 参数**：
- `range`: `24h` | `7d` | `30d` | `all`（必填）

**Response 200**：`EquityCurve`
**示例**：
```json
{
  "range": "24h",
  "points": [
    { "ts": "2026-04-15T13:30:00Z", "equity": 10307.65 },
    { "ts": "2026-04-15T14:00:00Z", "equity": 10312.20 }
  ]
}
```
**错误**：
- `400` — `range` 非法值

---

## 2. Scheduler 域（Dashboard 用）

### GET `/api/scheduler/status` — FR-802

**已存在**。返回 `SchedulerStatus`。
```json
{
  "enabled": true,
  "next_pair": "BTC/USDT",
  "next_run_at": "2026-04-16T13:35:00Z",
  "redis_available": true
}
```

---

## 3. Decisions 域（Decisions 页 + Backtest 详情用）

### GET `/api/decisions` — FR-803

**Query 参数**：
- `pair`?: string — 例 `BTC/USDT`
- `from`?: string — ISO 8601 起始
- `to`?: string — ISO 8601 截止
- `page`?: int = 1
- `size`?: int = 20

**Response 200**：`PaginatedResponse<DecisionCommit>`
**示例**：
```json
{
  "items": [
    {
      "commit_hash": "a1b2c3d",
      "ts": "2026-04-16T13:00:00Z",
      "pair": "BTC/USDT",
      "price": 65000.0,
      "verdict": { "action": "long", "size": 0.5, "confidence": 0.72, "reasoning": "...", "source": "ai" },
      "is_filled": true,
      "trace_id": "abc123"
    }
  ],
  "total": 142,
  "page": 1,
  "size": 20,
  "has_next": true
}
```
**注**：列表响应中的 `DecisionCommit` 仅含字段子集（节省带宽），完整结构走 `/{commit_hash}` endpoint。

---

### GET `/api/decisions/{commit_hash}` — FR-804

**Response 200**：`DecisionCommit`（完整结构，含 `agent_analyses`/`debate_rounds`/`risk_gate`/`execution`/`node_timeline`/`experience_memory_ref`）
**错误**：
- `404` — commit_hash 不存在

---

## 4. Backtest 域

### POST `/api/backtest/run` — FR-805

**Request Body**：`BacktestParams`
```json
{
  "start": "2026-01-01",
  "end": "2026-04-01",
  "pair": "BTC/USDT",
  "initial_capital": 10000,
  "mode": "rules",
  "session_name": "q1-rules-baseline"
}
```
**Response 202**：
```json
{ "run_id": "run_a1b2c3" }
```
**错误**：
- `400` — 参数校验失败（日期反向 / capital < 100 / mode 非法）

---

### GET `/api/backtest/runs/{run_id}` — FR-805

**Response 200**：`BacktestRun`
**示例（运行中）**：
```json
{
  "run_id": "run_a1b2c3",
  "params": { "...": "..." },
  "status": "running",
  "progress": 0.42,
  "started_at": "2026-04-16T13:00:00Z"
}
```
**示例（完成）**：
```json
{
  "run_id": "run_a1b2c3",
  "status": "completed",
  "progress": 1.0,
  "started_at": "2026-04-16T13:00:00Z",
  "finished_at": "2026-04-16T13:08:42Z",
  "result": {
    "metrics": { "total_return_pct": 0.085, "sharpe": 1.42, "max_drawdown_pct": 0.12, "win_rate": 0.61, "trades_count": 38 },
    "equity_curve": [{ "ts": "...", "equity": 10000 }],
    "decisions": []
  }
}
```

---

### DELETE `/api/backtest/runs/{run_id}` — FR-302（取消）

**Response 200**：
```json
{ "canceled": true }
```
**错误**：
- `409` — 任务已结束（completed/failed/canceled），不可取消

---

### GET `/api/backtest/sessions` — FR-806

**Response 200**：
```json
{ "sessions": ["q1-rules-baseline", "q2-llm-aggressive"] }
```

---

### GET `/api/backtest/sessions/{name}` — FR-806

**Response 200**：`BacktestSession`
**错误**：
- `404` — 会话名不存在

---

## 5. Risk 域

### GET `/api/risk/status` — FR-807

**说明**：替代既有 `portfolio.py` 中的 `/api/risk/status`（迁移到独立 `risk.py`）。

**Response 200**：`RiskStatus`
**示例**：
```json
{
  "trade_count_hour": 3,
  "trade_count_day": 12,
  "circuit_breaker": {
    "state": "active",
    "triggered_at": "2026-04-16T11:00:00Z",
    "expires_at": "2026-04-17T11:00:00Z",
    "reason": "daily_loss_exceeded"
  },
  "thresholds": {
    "max_position_pct": 0.30,
    "max_daily_loss_pct": 0.05,
    "max_stop_loss_pct": 0.02,
    "max_trades_per_hour": 10,
    "max_trades_per_day": 50,
    "post_loss_cooldown_seconds": 1800
  },
  "redis_available": true
}
```
**Redis 不可达**：返回 200，`redis_available: false`，计数字段为 `null`

---

### POST `/api/risk/circuit-breaker/reset` — FR-807 / FR-404

**Request Body**：无（前端用 confirm dialog 二次确认 NFR-S-008）
**Response 200**：
```json
{ "success": true, "message": "断路器已重置" }
```
**错误**：
- `409` — 断路器当前 `inactive`（无需重置）
- `503` — Redis 不可达

---

## 6. Metrics 域

### GET `/api/metrics/summary` — FR-808（已存在）

**Response 200**：`MetricsSummary`
**示例**：
```json
{
  "counters": {
    "trades_total": 142,
    "orders_placed": 138,
    "orders_failed": 4,
    "risk_rejections": 12,
    "debate_skipped_total": 23
  },
  "percentiles": {
    "pipeline_p50_ms": 1250,
    "pipeline_p95_ms": 4800,
    "execution_p50_ms": 320,
    "execution_p95_ms": 880
  },
  "collected_at": "2026-04-16T13:30:00Z"
}
```

---

### GET `/metrics`（已存在）

Prometheus 原始抓取端点。前端不直接使用；通过 `/api/metrics/summary` 取聚合。

---

## 7. Chat 域（P2）

### POST `/api/chat/stream` — FR-809

**Content-Type**: `application/json` 请求；`text/event-stream` 响应
**Request Body**：
```json
{
  "session_id": "sess_a1b2",
  "model": "gpt-4o-mini",
  "messages": [
    { "role": "user", "content": "分析下 BTC/USDT 当前态势" }
  ]
}
```
**Response**：SSE 流 — 详见 [sse-events.md](./sse-events.md)

---

## 8. Market 域（P2）

### GET `/api/market/{pair}/funding-rate` — FR-810

**Path 参数**：`pair`，例 `BTC-USDT`（URL safe，斜杠用 `-`）
**Query 参数**：
- `exchange`?: `binance` | `okx`（默认 binance）
- `range`?: `24h` | `7d`（默认 24h）

**Response 200**：
```json
{
  "pair": "BTC/USDT",
  "exchange": "binance",
  "points": [
    { "ts": "2026-04-16T08:00:00Z", "rate": 0.00012, "predicted_rate": 0.00015 }
  ]
}
```

---

### GET `/api/market/{pair}/open-interest` — FR-810

**Response 200**：
```json
{
  "pair": "BTC/USDT",
  "exchange": "binance",
  "points": [
    { "ts": "2026-04-16T08:00:00Z", "oi_usd": 12300000000 }
  ]
}
```

---

### GET `/api/market/{pair}/liquidations` — FR-810

**Response 200**：
```json
{
  "pair": "BTC/USDT",
  "exchange": "binance",
  "events": [
    { "ts": "2026-04-16T08:15:00Z", "side": "long", "size_usd": 250000, "price": 64900 }
  ]
}
```

---

## 9. 错误响应统一格式

所有非 2xx 响应：
```json
{
  "detail": "断路器当前未触发，无需重置",
  "code": "CIRCUIT_BREAKER_INACTIVE",
  "trace_id": "abc123"
}
```

| HTTP | 含义 |
|------|------|
| 400 | 客户端参数错误 |
| 401 | 缺少或错误的 X-API-Key |
| 404 | 资源不存在 |
| 409 | 状态冲突（任务已结束、断路器未触发） |
| 429 | 速率限制（streamFetch 退避） |
| 500 | 后端内部错误 |
| 503 | 上游依赖（Redis / 交易所）不可达 |

---

## 10. 实施优先级

| 优先级 | endpoint |
|--------|---------|
| **P1（页面集成前必须就绪）** | FR-800/801/802/803/804/805/806/807/808 |
| **P2（页面 9-10）** | FR-809/810 |

每个 endpoint 在 Phase 2 实施时遵循 A-7 纪律：
1. 先 pytest 写测试
2. 实现 endpoint
3. 通过 `docker compose up -d` + `curl` 验通
4. 前端写 zod schema + React Query hook
