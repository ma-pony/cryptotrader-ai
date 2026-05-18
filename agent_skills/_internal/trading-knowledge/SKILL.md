---
name: trading-knowledge
description: Shared trading knowledge including market microstructure definitions,
  regime taxonomy, funding rate semantics, and cross-agent pattern attribution rules.
scope: shared
version: '1.0'
manually_edited: false
access_count: 1544
last_accessed_at: '2026-05-18T14:55:08.193671+00:00'
---
# Shared Trading Knowledge Skill

## Purpose

Shared framework injected into all four trading agents. Defines common
output contract, reasoning hygiene, and meta-rules that keep agents from
talking past each other or absorbing the same bias. The skill is a
framework, not a rulebook of thresholds — agents are expected to reason
from the data they actually receive.

## Output Contract (apply to ALL agents)

Each agent returns:

- `direction`: bullish / bearish / neutral
- `confidence`: 0–1, calibrated subjective probability your direction is
  correct over the next cycle (or the appropriate horizon for your
  agent type)
- `sufficiency`: high / medium / low — about the data quality, not the
  conviction of your call
- `reasoning`: concise text citing only what the snapshot actually shows

Agents do **not** output `scale`, `action`, stop levels, or position
sizing — those are the verdict / risk / execution layer's job.

## Reasoning Hygiene (apply to ALL agents)

1. **Read the snapshot, not your priors.** The same number means different
   things across pairs, regimes, and timeframes. Anchor on direction of
   change and on the snapshot's own annotations rather than on numeric
   thresholds remembered from training.
2. **Acknowledge contradictions before overriding them.** Never bury
   counter-evidence inside a one-directional thesis.
3. **Default to `neutral` when evidence is mixed or amplitude small.**
   Forcing a lean without conviction makes the verdict layer noisier.
4. **Require multi-signal corroboration for regime claims.** Single
   indicators above their threshold are observations, not theses.
5. **State an invalidation condition.** For any directional call, name
   what observable event would falsify it — this lets the verdict layer
   size around risk distance and gives the reflection cycle ground truth.

## Cross-Agent Discipline (apply to ALL agents)

1. **Do not chain other agents' framings.** If one agent flags a macro
   read, the others should not parrot it as if it were independent
   evidence — that creates false consensus.
2. **Stay in your lane.** Tech reads price action and indicators; chain
   reads positioning and flow; news reads catalysts; macro reads backdrop.
   Borrowing each other's framings dilutes signal independence.
3. **Position state is verdict-layer's job.** Agents do not see current
   portfolio. Never write thesis sentences like "we are already short"
   — that is leakage from a layer above.

## Data Provenance (apply to ALL agents)

1. Only cite fields actually present in the snapshot. Never invent
   values for fields that are missing.
2. Treat `0` or absent as **missing**, not as a neutral reading.
3. Note staleness when the snapshot timestamp lags real-time in a
   fast-moving market — downgrade conviction accordingly.

## Time Horizon Awareness

The system runs hourly cycles. Different signal classes have very
different information cadences: price-action data refreshes intra-cycle,
positioning and flow refresh per-trade, news refreshes per-event, and
macro / sentiment indices typically refresh once per day or slower. Slow-
cadence signals carry backdrop bias, not fresh evidence every cycle.

## Pair-Level Independence

The system trades multiple pairs in parallel each cycle. When several
pairs share an underlying driver (a macro shift, a sector rotation), the
verdict layer can mistake that for independent confirmations. Flag
shared-driver observations as shared, not multiplicative.

## Attribution

When you cite a pattern in `applied:`:

- Self-agent context uses the bare name: `applied: pattern_name`
- Cross-agent / verdict context uses the prefixed name:
  `applied: agent_id::pattern_name`

Pattern names are short, descriptive labels of what the data showed.
The catalog evolves over time as the system distills patterns from
outcomes; agents do not need to memorize a fixed taxonomy.

## Confidence Calibration

Confidence is a probability, not a vibe. A 0.7 should mean you would
take this trade 7 times out of 10. A 0.5 means you would be indifferent.
Above 0.8 should be rare and reserved for cases with strong, mutually
reinforcing, unambiguous evidence.
