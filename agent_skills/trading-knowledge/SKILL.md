---
name: trading-knowledge
description: Shared trading knowledge including market microstructure definitions, regime taxonomy, funding rate semantics, and cross-agent pattern attribution rules.
scope: shared
version: "1.0"
manually_edited: false
---

# Shared Trading Knowledge Skill

## Purpose

This skill provides shared foundational knowledge injected into ALL trading agents. It defines common terminology, regime labels, market microstructure concepts, and attribution rules that ensure consistent interpretation across the four specialized agents.

## Funding Rate Semantics

- **Positive funding rate** (> 0): Long positions pay short positions. High positive funding (> 0.03%) indicates crowded longs — squeeze risk.
- **Negative funding rate** (< 0): Short positions pay long positions. Highly negative (< -0.01%) indicates crowded shorts — short squeeze risk.
- **Funding rate neutral** (near 0): Balanced positioning, less directional signal from positioning alone.

## Regime Taxonomy

The system tags each trading cycle with one or more regime labels:

| Tag | Condition |
|-----|-----------|
| `high_funding` | funding_rate > 0.03% |
| `negative_funding` | funding_rate < -0.01% |
| `high_vol` | volatility > threshold (configured) |
| `low_vol` | volatility < threshold |
| `trending_up` | 7d price change > +5% |
| `trending_down` | 7d price change < -5% |
| `extreme_fear` | Fear & Greed Index ≤ 25 |
| `extreme_greed` | Fear & Greed Index ≥ 75 |

## Spot vs Perpetual Contract Semantics

- **Spot markets**: Actual asset delivery; no funding; basis = 0
- **Perpetual futures (perps)**: No expiry; funding mechanism anchors price to spot; high OI indicates leveraged exposure
- **Basis**: Perp price - Spot price. Positive basis + high funding = leveraged longs dominant

## Pattern Attribution Rules

When referencing patterns in your reasoning:

- **Self-agent context**: Use bare name: `applied: pattern_name`
- **Cross-agent / verdict context**: Use prefix: `applied: agent_id::pattern_name` (e.g., `applied: tech::rsi_oversold_bounce`)
- Patterns are defined in each agent's SKILL.md under "Active Patterns Summary"

## Confidence Calibration Reference

| Range | Meaning |
|-------|---------|
| 0.9-1.0 | Multiple strong converging signals, no contradictions |
| 0.7-0.8 | Clear directional signal, minor contradictions |
| 0.5-0.6 | Mixed signals, slight lean |
| 0.3-0.4 | Weak/conflicting signals |
| 0.1-0.2 | Insufficient data or strong contradictions |

## Data Sufficiency Rules

- **high**: Core data sources present and complete — directional call warranted
- **medium**: Some data missing but key signals present — moderate confidence at best
- **low**: Most core data missing — set confidence ≤ 0.3, direction = neutral, state "insufficient data"
