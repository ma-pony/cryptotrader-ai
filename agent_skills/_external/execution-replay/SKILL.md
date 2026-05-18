---
name: execution-replay
description: CryptoTrader AI 执行回放技能。提供 journal 事件流查询和 OCO 算法订单状态查询，外部 agent 可追踪每笔交易的完整执行生命周期，用于风控审计、PnL 归因和异常检测。
version: "1.0"
---
# Execution Replay — 执行回放

CryptoTrader AI 的每个交易决策触发完整的执行链路，包含仓位开平、SL/TP 设置
和 OCO 算法订单管理。本技能描述如何查询执行事件流和订单状态。

## Journal 事件流

每次交易周期的关键事件均写入 journal，包含决策时间、执行结果和 PnL 信息。

```bash
# 获取最近 20 条 journal 事件（默认）
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/journal/events"

# 按时间范围过滤
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/journal/events?since=2026-05-17T00:00:00Z&limit=50"

# 按交易对过滤
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/journal/events?pair=BTC%2FUSDT%3AUSDT"

# 按事件类型过滤（open / close / update / error）
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/journal/events?event_type=close"
```

响应示例：

```json
{
  "items": [
    {
      "event_id": "evt_abc123",
      "cycle_id": "cycle_xyz789",
      "pair": "BTC/USDT:USDT",
      "event_type": "close",
      "action": "long",
      "entry_price": 65200.0,
      "exit_price": 67800.0,
      "pnl_usdt": 312.5,
      "pnl_pct": 3.99,
      "sl_triggered": false,
      "tp_triggered": true,
      "timestamp": "2026-05-17T16:45:00Z"
    }
  ],
  "total": 1
}
```

## OCO 算法订单状态查询

CryptoTrader AI 使用 OKX 服务端 SL/TP（OCO 算法订单）管理止损止盈。

```bash
# 查询所有活跃 OCO 订单状态
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/execution/oco-status"

# 查询指定交易对的 OCO 订单
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/execution/oco-status?pair=BTC%2FUSDT%3AUSDT"
```

响应示例：

```json
{
  "items": [
    {
      "pair": "BTC/USDT:USDT",
      "algo_id": "alg_20260517_001",
      "side": "long",
      "entry_price": 65200.0,
      "sl_price": 64000.0,
      "tp_price": 68500.0,
      "sl_order_id": "slord_001",
      "tp_order_id": "tpord_001",
      "status": "active",
      "created_at": "2026-05-17T10:30:00Z"
    }
  ],
  "total": 1
}
```

## 执行状态码说明

| status | 含义 |
|---|---|
| `active` | 持仓中，SL/TP 订单挂起 |
| `tp_hit` | 止盈已触发，仓位已平 |
| `sl_hit` | 止损已触发，仓位已平 |
| `manually_closed` | 人工干预平仓 |
| `cancelled` | 订单已取消（风控拒绝等） |
| `error` | 执行异常，需人工检查 |

## PnL 归因查询

```bash
# 获取今日 PnL 汇总（按交易对）
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/journal/pnl-summary?period=today"

# 获取近 7 天 PnL 曲线
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/journal/pnl-summary?period=7d"
```

## API 规范引用

- Journal 事件完整字段定义：`docs/api/execution.yaml`
- OCO 订单 schema：`docs/api/execution.yaml`（`OcoStatusItem` schema）
- 认证方式：参见 `GET /skill/cryptotrader` 中的认证说明
