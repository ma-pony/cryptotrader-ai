---
name: trading-knowledge
description: Shared trading knowledge including market microstructure definitions,
  regime taxonomy, funding rate semantics, and cross-agent pattern attribution rules.
scope: shared
version: '1.0'
manually_edited: false
access_count: 1028
last_accessed_at: '2026-05-12T14:40:27.475965+00:00'
---
# Shared Trading Knowledge Skill

## Purpose

This skill provides shared foundational knowledge injected into ALL trading agents. It defines common terminology, regime labels, market microstructure concepts, and attribution rules that ensure consistent interpretation across the four specialized agents.

## Funding Rate Semantics

- **Positive funding rate** (> 0): longs pay shorts. The snapshot annotates
  `ELEVATED — crowded long` based on a configured threshold; in absence of
  that tag, treat as normal directional bias, not a crowd signal.
- **Negative funding rate** (< 0): shorts pay longs. Snapshot annotates
  `NEGATIVE — crowded short` only when clearly past the negative threshold.
- **Funding rate neutral** (small magnitude either side): balanced
  positioning; do not derive directional signal from funding alone.
- Anchor on the **snapshot's annotation**, not on a memorized % threshold.
  Pair-specific baselines vary materially across BTC/ETH/SOL/DOGE/etc.

## Regime Taxonomy

The system tags each trading cycle with one or more regime labels. Tag
thresholds are coarse defaults — pair-specific normalization happens
downstream:

| Tag | Condition (default) |
|-----|---------------------|
| `high_funding` | funding_rate > configured high threshold (annotated) |
| `negative_funding` | funding_rate < configured low threshold (annotated) |
| `high_vol` | volatility > configured threshold |
| `low_vol` | volatility < configured threshold |
| `trending_up` | 7d price change clearly positive (~+5% baseline; crypto pairs vary) |
| `trending_down` | 7d price change clearly negative (~-5% baseline; crypto pairs vary) |
| `extreme_fear` | Fear & Greed Index ≤ 25 |
| `extreme_greed` | Fear & Greed Index ≥ 75 |

The 5%/7d trend threshold is loose for crypto — DOGE / SOL routinely move
10-20% per week even in chop. Use this only as a coarse regime label, not
as a directional trigger.

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

## Anti-Anchor Rules (apply to ALL agents)

These rules exist because absolute-level anchoring has historically caused
permanent directional biases (PROD-2026-05-07 / 2026-05-12 DXY incident
being the canonical example).

1. **Read direction of change, not absolute level.** A printed value that
   looks "high" may be the normal range for its scale; the snapshot's own
   annotations (`ELEVATED`, `crowded long`, etc.) are the calibrated signal.
2. **Trust the snapshot's annotation over a remembered threshold.** Renderer
   thresholds are coarse defaults that may not match the LLM's prior
   knowledge of similar metrics.
3. **Pair-specific baselines vary.** Funding 0.02% may be elevated on BTC
   but normal on DOGE. Long/short ratio 1.8 may be crowded on ETH but
   chronically biased on smaller alts.
4. **One indicator above its threshold is not a thesis.** Require 2+
   independent indicators in the same direction before claiming a regime
   (crowded long, distribution, breakdown, etc.).
5. **Do NOT chain other agents' framings.** If macro_agent flags USD-rising
   risk-off, news_agent should not parrot the same conclusion as if it were
   independent evidence — that creates false consensus.
6. **State your scale.** When citing a number ("RSI 38", "VIX 23.75"),
   include the band/interpretation ("RSI 38 = lower-neutral", "VIX 23.75 =
   normal range") so verdict layer does not mis-weight it.

## Symmetric Coverage Rules (apply to ALL agents)

Past audits found agents reflexively framing bearish theses while ignoring
parallel bullish setups. To rebalance:

1. **Both directions must be examined per cycle.** Briefly state what would
   make the OPPOSITE direction valid; only then commit to your direction.
2. **Use symmetric pattern language.** If you have a "crowded long → short"
   trigger, you must equally honour "crowded short → long" when applicable.
3. **Default neutral, not default short.** When evidence is mixed or
   amplitude small, `direction = neutral` is the correct answer — not "lean
   bearish because nothing's pushing up".
4. **Counter-evidence weighting.** Acknowledge contradictory signals
   explicitly BEFORE overriding them. Never bury contradictions inside a
   directional thesis.

## Position-State Separation (apply to ALL agents)

Agents see market state and (optionally) an `experience` text from prior
cycles. Agents do NOT see current portfolio positions. Therefore:

1. Do NOT write thesis sentences like "we are already short" — that is the
   verdict layer's job. Agents only opine on market direction.
2. If the `experience` text mentions prior positions or outcomes, use it
   for pattern learning only — do not anchor your current call on whether
   the system would or would not need to act on it.
3. Output `direction`, `confidence`, `sufficiency`, `reasoning` — never a
   `scale` or `action` (those are verdict-layer fields).

## Data Provenance Rules (apply to ALL agents)

1. Only cite fields actually present in the snapshot you were given.
   Never invent values for fields that are missing.
2. If a field is `0` or absent, treat as **missing**, not as a neutral
   reading (see Data Sufficiency Rules above).
3. Note staleness when the snapshot timestamp lags real-time by hours
   in a fast-moving market — downgrade conviction accordingly.
