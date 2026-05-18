---
name: market-intel
description: CryptoTrader AI 市场情报技能。提供 4 个专业 agent（技术分析/链上分析/新闻情感/宏观分析）的最新分析结果及完整市场快照数据，外部 agent 可获取结构化市场判断用于决策辅助。
version: "1.0"
---
# Market Intel — 市场情报

CryptoTrader AI 的 4 个专业 agent 并行分析市场，各自输出方向判断和置信度。
本技能描述如何获取这些分析结果及底层市场快照数据。

## 4 Agent 分析架构

| Agent | ID | 分析维度 |
|---|---|---|
| 技术分析 | `tech` | 价格行为、均线、动量指标 |
| 链上分析 | `chain` | 资金费率、持仓量、交易所净流量 |
| 新闻情感 | `news` | 新闻标题、监管事件、社交情绪 |
| 宏观分析 | `macro` | 美联储政策、美元强度、ETF 资金流 |

## 获取市场快照

```bash
# BTC/USDT 永续合约快照
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/snapshot/BTC%2FUSDT%3AUSDT"

# ETH/USDT 快照
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/snapshot/ETH%2FUSDT%3AUSDT"
```

响应示例（节选）：

```json
{
  "pair": "BTC/USDT:USDT",
  "timestamp": "2026-05-17T14:00:00Z",
  "ticker": {"last": 67500.0, "volume_24h": 2800000000},
  "funding_rate": 0.00012,
  "fear_greed_index": 62,
  "technical": {"rsi_14": 58.3, "ema_20": 66800.0},
  "onchain": {"open_interest": 18500000000, "exchange_netflow": -1200},
  "macro": {"dxy": 103.8, "fed_rate": 5.25}
}
```

## 获取 Agent 分析结果

```bash
# 获取 tech agent 最近 5 次分析结果
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/agents/tech/results/recent?limit=5"

# 获取 chain agent 最近分析
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/agents/chain/results/recent"

# 获取 news agent 最近分析
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/agents/news/results/recent"

# 获取 macro agent 最近分析
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/agents/macro/results/recent"
```

响应示例：

```json
{
  "items": [
    {
      "cycle_id": "abc123",
      "agent": "tech",
      "pair": "BTC/USDT:USDT",
      "direction": "bullish",
      "confidence": 0.71,
      "sufficiency": "high",
      "reasoning": "RSI 从超卖区反弹，EMA20 形成支撑...",
      "timestamp": "2026-05-17T14:00:05Z"
    }
  ],
  "total": 1
}
```

## Agent 共识查询

4 个 agent 的方向判断共同输入 Debate Gate。外部系统可读取多个 agent 结果
并自行计算共识强度：

```python
directions = [r["direction"] for r in all_agent_results]
consensus = max(set(directions), key=directions.count)
consensus_ratio = directions.count(consensus) / len(directions)
```

## API 规范引用

- 快照字段完整定义：`docs/api/market.yaml`
- Agent 结果字段定义：`docs/api/market.yaml`（`AgentResult` schema）
- 认证方式：参见 `GET /skill/cryptotrader` 中的认证说明
