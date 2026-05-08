# Brainstorm Session: Spec 016 — Skill / Memory Evolution Prior-Art Research

**Date**: 2026-05-08
**Session ID**: ca0a20d8-015a-4078-a62d-cfd749ffca21
**Outcome**: Spec 016 created, reviewed (✅ SOUND), committed; Phase 1 research starting immediately

## Context

User raised broader refactor request: extract agent ROLE prompts from .py files into config, compose final prompt from config + memory.md, expand skill knowledge base, evolve skill+memory logic referencing 8 GitHub projects:

- SkillClaw (AMAP-ML)
- MetaClaw (aiming-lab)
- OpenClaw-RL (Gen-Verse)
- Hermes Agent Self-Evolution (NousResearch)
- EvoSkill (sentient-agi)
- EvoSkills (EvoScientist)
- skill-evolution (hao-cyber)
- autoresearch (uditgoenka)

User explicitly suggested splitting research into a separate task. Project context: spec 014 has just landed two-layer agent_skills/ + agent_memory/ architecture (1-2 days in production); spec 015 (just merged) added runtime safety guardrails after trader-grade audit found multiple issues.

## Scope Decomposition

Identified 3 independent subsystems with cleaner separate-spec boundaries:

| Spec | Subsystem | Risk |
|---|---|---|
| 016 | Prior-art research (this spec) | Low — research only, no code changes |
| 017 | Agent prompt externalization | Low — mechanical refactor |
| 018 | Skill evolution v2 | High — may replace spec 014 components |

User chose **option (c) trilogy** — brainstorm all 3 in series, each generating its own spec.

## Decisions Made (Spec 016 Only)

1. **Research time unconstrained** (per user) — all 8 angles covered for all 8 projects (no compromise on depth)
2. **Deliverable structure**: per-project docs + comparison matrix + synthesis + decisions log (4 doc types)
3. **Methodology**: WebFetch + git clone + WebSearch; **no demo/test runs**
4. **Tier prioritization**: SkillClaw / Hermes / OpenClaw-RL deep, MetaClaw / EvoSkill / EvoSkills / skill-evolution细 read, autoresearch scan
5. **Phase split**:
   - **Phase 1** = prompt-assembly + memory mechanisms only (unblocks spec 017)
   - **Phase 2** = all 8 angles fully (unblocks spec 018)
6. **Open mind on current architecture**: spec 014's 5-layer anti-overfitting / PnL maturity FSM / single-writer reflection / SKILL.md protocol are NOT presupposed correct
7. **License recording mandatory**: every project's frontmatter has `license:` field; fork-recommendations must verify compat

## Spec Review Outcome

`spex:review-spec` returned ✅ SOUND — Completeness 4.5/5, Clarity 4/5, Implementability 5/5, Testability 5/5. No P0/P1 issues. 3 Optional improvement suggestions:
- SC-R4: quantify "严重不匹配" with calculable ratio
- Add separate Non-Functional Requirements section
- Background: anchor "skill 命中率偏低" to concrete data point

User chose option (d): commit current spec, defer optional improvements, immediately launch Phase 1 research.

## Open Threads / Next Up

- **Spec 017 brainstorm** — blocked on Phase 1 research completion (SC-R6 satisfied)
- **Spec 018 brainstorm** — blocked on Phase 2 research completion (SC-R7 satisfied)
- **N1 long-signal asymmetry / N6 drawdown_pct discrepancy / N8 regime detection / N10 funding-carry / N11 skill maturity / N12 cross-cycle learning** — spec 015 OOS, may surface again during 018 brainstorm if research findings touch them

## Lessons / Patterns (for future brainstorms)

1. When user request spans multiple subsystems, **decompose first, brainstorm each separately**. Single mega-spec for 3-system request would have been unwieldy.
2. **Phase split with explicit gates** (Phase 1 unlocks 017, Phase 2 unlocks 018) lets downstream work start early without waiting for full research.
3. **Open mind on existing architecture** must be stated in Background; otherwise reviewers tend to defend current design even when better options exist.
4. **License compliance** is much cheaper to record during research than to retrofit later.
