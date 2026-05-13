---
agent_id: example
description: 最小合法示例 agent，用于单元测试 fixture
sections:
  - system_prompt
  - user_tail
  - available_skills
  - output_schema
budget: 4000
priority:
  system_prompt: 1
  output_schema: 1
  snapshot: 2
  portfolio: 3
  user_tail: 4
  available_skills: 6
---

## system_prompt

你是一个示例分析 agent，用于单元测试。

职责：输出固定格式的 JSON 分析结果。

## user_tail

请基于上述数据，输出符合 output_schema 的 JSON。

## available_skills

（运行时由 SkillProvider 注入）

## output_schema

```json
{
  "direction": "bullish|bearish|neutral",
  "confidence": 0.0,
  "reasoning": "分析理由"
}
```
