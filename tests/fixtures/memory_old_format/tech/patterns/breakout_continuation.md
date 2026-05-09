---
name: breakout_continuation
agent: tech
description: 价格突破关键阻力后持续上涨的形态
maturity: active
version: 2
manually_edited: false
regime_tags:
  - trending
pnl_track:
  cases: 10
  wins: 8
  win_rate: 0.8
  avg_pnl: 45.0
  last_active: '2026-01-10'
source_cycles:
  - cycle_abc
  - cycle_def
created: '2026-01-01T00:00:00+00:00'
---
## Rule

当价格突破关键阻力位并伴随成交量放大时，趋势延续概率高。

## Conditions
- 价格收盘突破前高
- 成交量 > 20MA 的 150%
- OBV 持续上升

## Invalidation
- 价格回落至阻力位下方
- 成交量萎缩
