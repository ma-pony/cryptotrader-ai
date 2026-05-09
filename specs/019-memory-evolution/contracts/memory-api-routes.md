# Contract：Memory API Routes

**模块路径**：`src/api/routes/memory.py`（本 spec 新增）

**注册位置**：`src/api/main.py` MUST `app.include_router(memory.router, prefix="/api/memory", tags=["memory"])`

## 4 个 Endpoints

### 1. `GET /api/memory/rules`

**Query 参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `agent` | str | ✗ | tech / chain / news / macro；缺省返回所有 |
| `status` | str | ✗ | observed / probationary / active / deprecated / archived；缺省返回所有 |

**Response 200**：

```json
{
  "items": [
    {
      "name": "high_funding_fade",
      "agent": "macro",
      "description": "极端 funding rate 反向操作",
      "maturity": "active",
      "importance": 0.85,
      "access_count": 23,
      "last_accessed_at": "2026-05-09T12:34:56Z",
      "pnl_track": {
        "successes": 12,
        "losses": 3,
        "total_pnl": 1450.5
      },
      "regime_tags": ["choppy", "high_funding"],
      "fundamental_failure_streak": 0,
      "version": 4,
      "manually_edited": false
    }
  ],
  "total": 1
}
```

**Response 400**：参数验证失败（如 status 不是合法 Maturity 字面量）

**Response 404**：agent 名不存在

### 2. `GET /api/memory/cases`

**Query 参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `from` | ISO8601 | ✗ | 起始时间；缺省 7 天前 |
| `to` | ISO8601 | ✗ | 结束时间；缺省现在 |
| `agent` | str | ✗ | 过滤 agent_analyses 含该 agent 的 case |

**Response 200**：

```json
{
  "items": [
    {
      "cycle_id": "0aef65c79ca4e2cd",
      "timestamp": "2026-05-07T14:02:41Z",
      "pair": "SOL/USDT:USDT",
      "verdict_action": "short",
      "final_pnl": -45.5,
      "trade_execution": {
        "entry_price": 88.45,
        "stop_loss": 90.0,
        "take_profit": 84.0,
        "actual_exit_price": 89.95,
        "fill_status": "stopped_out",
        "hit_sl": true
      },
      "ive_classification": {
        "failure_type": "implementation",
        "confidence": 0.78,
        "reasoning": "进场价偏高 + 停损撞了短期反弹"
      }
    }
  ],
  "total": 1
}
```

### 3. `GET /api/memory/transitions`

**Query 参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `since` | ISO8601 | ✗ | 起始时间；缺省 24h 前 |

**Response 200**：

```json
{
  "items": [
    {
      "rule_id": "macro::high_funding_fade",
      "agent_id": "macro",
      "old_state": "probationary",
      "new_state": "active",
      "triggered_by": "time_elapsed",
      "timestamp": "2026-05-09T11:30:00Z"
    }
  ],
  "total": 1
}
```

### 4. `GET /api/memory/archived`

**Query 参数**：无

**Response 200**：

```json
{
  "items": [
    {
      "name": "btc_dominance_extreme",
      "agent": "macro",
      "archived_at": "2026-05-08T09:15:00Z",
      "fundamental_failure_streak": 3,
      "final_pnl_track": {
        "successes": 5,
        "losses": 8,
        "total_pnl": -320.0
      },
      "archive_reason": "3 连续 fundamental 失败"
    }
  ],
  "total": 1
}
```

## 错误处理

- 所有 endpoint 在 IO 异常时返回 500 + structured error JSON：
  ```json
  {"error": "memory_io_error", "detail": "..."}
  ```
- query 参数解析错误返回 400 + `{"error": "invalid_query", "detail": "..."}`
- agent_id 不存在返回 404 + `{"error": "agent_not_found"}`

## 鉴权

沿用 spec 015 既有 API 鉴权机制（X-API-Key header）。

## Caching

Response 含 `Cache-Control: max-age=30` for `/rules` / `/cases` / `/transitions`；`/archived` 含 `max-age=300`（变化频率低）。

## 单测要求

参考 spec.md SC-Z12：`tests/test_api_memory.py` ≥ 6 用例 PASS：
1. `GET /api/memory/rules?agent=tech` 返回 200 + JSON
2. `GET /api/memory/cases?agent=macro` 返回近期 case
3. `GET /api/memory/transitions?since=...` 返回 events
4. `GET /api/memory/archived` 返回 archived list
5. 错误参数返回 400（如 `status=invalid`）
6. agent_id 不存在返回 404

## 集成 Frontend

`web/src/pages/memory/queries.ts` 4 个 React Query hooks 调用对应 endpoint：

```ts
useMemoryRules({ agent, status }, { staleTime: 30000 })
useMemoryCases({ from, to, agent }, { staleTime: 60000 })
useRecentTransitions({ since }, { staleTime: 30000 })
useArchivedRules({}, { staleTime: 300000 })
```

复用 spec 014/15 既有 React Query / api client 模式（如 `web/src/lib/api.ts`）。
