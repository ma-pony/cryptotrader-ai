---
name: high_funding_fade
agent: macro
description: 极端正向资金费率时做空（均值回归）
maturity: observed
version: 1
manually_edited: false
regime_tags:
  - high_funding
  - overheated
pnl_track:
  cases: 3
  wins: 2
  win_rate: 0.667
  avg_pnl: 30.0
  last_active: '2026-01-15'
source_cycles:
  - cycle_xyz
created: '2026-01-15T00:00:00+00:00'
---
## Rule

当资金费率超过 0.1% 时，多头杠杆过高，做空胜率较好。

## Conditions
- 资金费率 > 0.1%
- 持仓量高于历史均值 120%

## Invalidation
- 资金费率回落至 0.05% 以下
- 市场进入趋势性上涨
