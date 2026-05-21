---
name: verdict-feed
description: CryptoTrader AI 交易决策流技能。提供最新 verdict 查询、实时 SSE 订阅及心跳监控，外部 agent 可用于跟踪交易信号并在决策事件发生时触发下游动作。
version: "1.0"
---
# Verdict Feed — 交易决策流

CryptoTrader AI 的核心输出是经过 4 agent 分析 + Debate Gate 协议过滤的
`verdict`（交易决策）。本技能描述如何查询历史 verdict 及订阅实时决策流。

## 用例场景

- 外部风控系统监控每笔交易决策的方向和置信度
- 告警 bot 在 verdict action 为 `long`/`short` 时发送 Telegram/Slack 通知
- 回测工具拉取历史 verdict 序列用于策略对比
- 可视化 dashboard 实时展示决策流时间线

## 查询最近 Verdict

```bash
# 最近 10 条 verdict（默认）
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/verdicts/recent"

# 指定条数 + 指定交易对
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/verdicts/recent?limit=20&pair=BTC%2FUSDT%3AUSDT"
```

响应示例：

```json
{
  "items": [
    {
      "cycle_id": "abc123",
      "pair": "BTC/USDT:USDT",
      "action": "long",
      "confidence": 0.72,
      "sl_price": 64800.0,
      "tp_price": 68500.0,
      "reasoning": "多 agent 一致看多，链上资金费率低位...",
      "timestamp": "2026-05-17T14:32:00Z"
    }
  ],
  "total": 1
}
```

## 实时 SSE 订阅

```bash
# Server-Sent Events 实时决策流
curl -H "X-API-Key: $API_KEY" \
     -H "Accept: text/event-stream" \
     "http://localhost:8000/api/events/stream?topic=verdict"
```

每条 SSE 事件格式：

```
event: verdict
data: {"cycle_id":"xyz789","pair":"ETH/USDT:USDT","action":"short","confidence":0.68}
```

## 心跳订阅模式

外部 agent 推荐采用心跳轮询避免长连接超时：

```python
import httpx, time

POLL_INTERVAL = 30  # 秒
last_seen_id = None

while True:
    params = {"limit": 5}
    if last_seen_id:
        params["after_id"] = last_seen_id
    resp = httpx.get(
        "http://localhost:8000/api/verdicts/recent",
        headers={"X-API-Key": API_KEY},
        params=params,
    )
    items = resp.json()["items"]
    if items:
        last_seen_id = items[0]["cycle_id"]
        # 处理新 verdict ...
    time.sleep(POLL_INTERVAL)
```

## API 规范引用

- 完整 verdict 字段定义：`docs/api/verdict.yaml`
- SSE 事件格式定义：`docs/api/events.yaml`
- 认证方式：参见 `GET /skill/cryptotrader` 中的认证说明
