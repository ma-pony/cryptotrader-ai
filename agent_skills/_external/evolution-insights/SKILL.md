---
name: evolution-insights
description: CryptoTrader AI Trilogy 进化系统产出技能。提供 skill 提案列表、pattern 统计及 skill 详情查询，外部 agent 可追踪系统自我进化状态并获取最新策略模式数据。
version: "1.0"
---
# Evolution Insights — Trilogy 进化系统产出

CryptoTrader AI 包含一个持续自我进化的 Trilogy 系统，通过 Memory Evolution、
Skill Evolution 和 Evolution Daemon 三层架构自动提炼交易经验。本技能描述
如何查询进化系统的产出数据。

## Trilogy 架构概览

```
CaseRecord (每次交易周期)
    ↓ Memory Evolution — 蒸馏 PatternRecord
PatternRecord (策略模式，FSM 成熟度管理)
    ↓ Skill Evolution — LLM 提案 SkillRecord
SkillRecord / SKILL.md (agent 注入的技能，git 跟踪)
```

## 查询 Skill 列表

```bash
# 获取所有 internal skill 的摘要列表
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/memory/skills"

# 按 agent 过滤
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/memory/skills?agent=tech"
```

响应示例（`SkillRecord` 列表）：

```json
{
  "items": [
    {
      "name": "tech-analysis",
      "scope": "agent:tech",
      "version": "1.0",
      "description": "Technical analysis skill for interpreting price action...",
      "importance": 0.8,
      "confidence": 0.75,
      "access_count": 414,
      "last_accessed_at": "2026-05-13T02:46:27Z",
      "manually_edited": false,
      "regime_tags": []
    }
  ],
  "total": 5
}
```

## 查询 Pattern 列表

```bash
# 获取所有已知 pattern 统计（active + probationary）
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/memory/patterns"

# 按 agent + 成熟度过滤
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/memory/patterns?agent=tech&maturity=active"
```

`PatternRecord` schema 示例：

```json
{
  "name": "funding_squeeze_long",
  "agent": "tech",
  "description": "资金费率从极负值快速回升时的看多形态",
  "maturity": "active",
  "regime_tags": ["negative_funding"],
  "pnl_track": {
    "cases": 23,
    "win_rate": 0.61,
    "avg_pnl": 127.4,
    "last_active": "2026-05-16T08:00:00Z"
  },
  "created": "2026-04-01T00:00:00Z"
}
```

## 查询 Skill 提案

Evolution Daemon 每天自动生成 skill 优化提案（`.draft` 文件），等待人工审核。

```bash
# 获取待审核的 skill 提案列表
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/memory/skill-proposals"
```

响应示例：

```json
{
  "items": [
    {
      "name": "tech-analysis",
      "draft_path": "agent_skills/_internal/tech-analysis/SKILL.md.draft",
      "created_at": "2026-05-17T03:00:00Z",
      "llm_inferred_metadata": {
        "regime_tags": ["high_funding"],
        "triggers_keywords": ["rsi", "funding", "squeeze"],
        "importance": 0.85,
        "confidence": 0.78
      },
      "llm_call_failed": false,
      "user_saved": false
    }
  ],
  "total": 1
}
```

## Skill Access 统计

```bash
# 获取 skill 调用频率统计（用于观测 agent 使用偏好）
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/memory/skill-access"
```

## API 规范引用

- Skill / Pattern 完整 schema：`docs/api/memory.yaml`
- `SkillRecord` / `PatternRecord` Pydantic 定义：`src/api/routes/memory.py`
- 认证方式：参见 `GET /skill/cryptotrader` 中的认证说明
