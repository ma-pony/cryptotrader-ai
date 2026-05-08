# Synthesis: Cross-Project Patterns

**Status**: Phase 1 partial — only "Prompt Assembly" + "Memory ↔ Skill" chapters complete.
**Phase 2 deferred**: Evolution Algorithm / Skill Data Structure / Retrieval / Evaluation / Agent↔Skill Boundary / Engineering Implementation Details.

---

## Phase 1 — Chapter 1: Prompt Assembly Patterns

This chapter consolidates how all 8 researched projects build their final LLM input. Source citations point to `projects/<name>.md`.

### Pattern P1: Sectioned Named-Segment Composition

**Adopted by**: Hermes, MetaClaw, EvoSkills, SkillClaw (partial), microsoft/autogen (industrial-strength implementation)

The system prompt is constructed from named, semi-independent sections rather than a single monolithic string. Each section has a clear purpose (memory guidance, skill list, retrieval rules, output format).

| Project | Implementation |
|---|---|
| **Hermes** | `prompt_builder.py` builds segments like `MEMORY_GUIDANCE`, `SESSION_SEARCH_GUIDANCE`, `SKILLS_GUIDANCE`. Some are DSPy-parameterized (evolvable), others fixed. |
| **MetaClaw** | `api_server.py` runs three composition modes (Synergy / Memory-only / Skills-only), each with explicit section assembly + 20k token budget greedy truncation. |
| **EvoSkills** | 13 `SKILL.md` files each describe one section ("research workflow", "data analysis"); declarative `should-trigger` metadata routes which section gets injected. |
| **SkillClaw** | Two-stage: stage 1 injects YAML name+description catalog only; stage 2 injects full SKILL.md body when matched. |
| **microsoft/autogen** | Industrial-strength **four-slot separation**: `_system_messages` (static role) + `_tools` (schema) + `Memory.update_context()` (injection hook) + `ChatCompletionContext` (history variants — buffered / head-tail / token-trimmed). System prompt itself stays out of `ChatCompletionContext` so prompt cache stays warm. |

**Implication for cryptotrader-ai**: spec 017's prompt externalization should use **named sections** (e.g., `agent_role`, `available_skills`, `recent_memory`, `output_schema`), not concat one big ROLE string. Each section can then be edited / evolved / token-budgeted independently. autogen's four-slot pattern is the most directly portable reference implementation — see D-PA-01.

### Pattern P2: Progressive Disclosure / Lazy-Load Skills

**Adopted by**: SkillClaw, skill-evolution

Skills are **not all loaded upfront**. The prompt initially carries only metadata (name + 1-line description), and full skill body is appended ONLY when the LLM signals a hit.

| Project | Mechanism |
|---|---|
| **SkillClaw** | Client Proxy starts with all-skills YAML catalog (~50 bytes/skill); on LLM `<load_skill>` signal, appends full SKILL.md (~1-15KB) for that turn |
| **skill-evolution** | 3-level hierarchy: L1 metadata always in context, L2 SKILL.md on trigger, L3 `references/` files on demand |

**Implication for cryptotrader-ai**: current spec 014 architecture loads all skills as `description` inline. With ~10 skills × 1KB this is fine, but at 100+ skills we'd hit budget. spec 017 should design skill schema to allow lazy-load even if Phase 1 doesn't implement it.

### Pattern P3: Tokenizer-Native Template Rendering

**Adopted by**: OpenClaw-RL (uniquely strong)

Rather than maintain a custom prompt template, delegate to the model tokenizer's chat template (e.g., HuggingFace `tokenizer.apply_chat_template()` with Jinja2). Project code does only role normalization + content shaping; the rendering is done by the tokenizer per model.

**Implication for cryptotrader-ai**: we currently build prompts as Python f-strings. If we adopt tokenizer-native templates, prompt format becomes **model-portable** automatically — switching from glm-5 to gpt-5 to claude-opus-4-7 wouldn't require prompt re-engineering. Worth investigating in spec 017.

### Pattern P4: Last-Message vs System-Prompt Injection Point

**Adopted by**: OpenClaw-RL (last-message); MetaClaw (system); Hermes (system fixed-section); SkillClaw (system catalog + dynamic body)

Two distinct philosophies on where to inject experience/memory:

- **System-prompt injection** (Hermes, MetaClaw, SkillClaw): memory and skills sit in the system prompt, persistent across the conversation
- **Last-message injection** (OpenClaw-RL): only the final user message is augmented with experience; system prompt stays untouched

Each has distinct implications for token caching (Anthropic prompt cache works best when system prompt is stable across turns) and behavioral steering (system prompts are typically more strongly internalized by the model).

**Implication for cryptotrader-ai**: spec 017 should pick one consciously. Given our 4-agent + verdict architecture and use of Anthropic prompt cache (spec 004 LLM resilience), **system-prompt injection with a stable structure is likely the better choice** — but evidence from OpenClaw-RL suggests we may want to put per-cycle experience in the user message tail to avoid cache invalidation.

### Pattern P5: No Mid-Session Hot-Swap

**Adopted by**: Hermes (explicit), SkillClaw (only between sessions via dashboard sync), skill-evolution (uses git as the swap-point)

Once a session starts with a fixed skill set, those skills don't change until the session ends. New skills are added in the **next** session, not mid-conversation. This makes prompt cache more effective and avoids confusing the model with shifting context.

**Implication for cryptotrader-ai**: our current spec 014 reflection job runs **between** trading cycles, which already follows this pattern. spec 017 should make this explicit — agent prompts compose at cycle start, never mutate within a cycle.

### Pattern P6: Token-Budget Greedy Truncation

**Adopted by**: MetaClaw (uniquely concrete with 20k cap)

Compose the full ideal prompt; if it exceeds budget, drop sections in priority order (lowest importance first) until it fits.

**Implication for cryptotrader-ai**: we currently have no explicit token budget enforcement at the agent prompt level (only LLM-level retry on overflow). spec 017 should consider an explicit budget enforcer — important if skill library grows.

### Pattern P7: Declarative Trigger Metadata

**Adopted by**: EvoSkills, SkillClaw (YAML), skill-evolution (frontmatter + descriptions)

Skill files declare WHEN they should be considered, not just WHAT they do. EvoSkills uses `allowed_tools:` and `should-trigger:` fields in frontmatter; SkillClaw's catalog has 1-line descriptions used by the LLM as routing.

**Implication for cryptotrader-ai**: spec 014's SKILL.md frontmatter has `name` and `description`. We should extend it (or define the shape in spec 017) with declarative trigger metadata — e.g., `regime_tags: [trending, choppy]` so a regime-detection layer can pre-filter without LLM round-trip.

### Pattern P8: Dual-Mode Retrieval (Embedding + Keyword)

**Adopted by**: MetaClaw

When deciding which skill / memory to inject, use **two retrieval paths**:
- **Template / keyword match** for zero-latency, high-precision routing
- **Embedding similarity** for semantic / fuzzy matches

**Implication for cryptotrader-ai**: this is more relevant to spec 018 (skill retrieval algorithm) than 017, but the schema implications are real — skill metadata needs both human-readable trigger keywords AND can support embedding indexing.

---

## Phase 1 — Chapter 2: Memory ↔ Skill Wiring (lite)

(This chapter intentionally covers only the prompt-injection wiring of memory, NOT memory evolution algorithms — the latter is Phase 2.)

### Pattern M1: Memory as a Section in the Prompt

All 8 projects (where memory exists) inject memory as one of the sectioned segments described in P1. None inject memory as a tool call result.

### Pattern M2: Importance + Access-Count for Retrieval

**Adopted by**: MetaClaw uniquely.

Memory units carry `importance: float` + `access_count: int` fields. Retrieval ranks by importance, decays by access count to prevent over-citation. **Implication**: spec 017's memory.md format should include similar metadata fields, deferring the exact ranking algorithm to spec 018.

### Pattern M3: Causal-Chain Session Logs

**Adopted by**: SkillClaw uniquely strong.

Sessions are recorded as a structured chain: `user_prompt → tool_calls → intermediate_feedback → final_answer`. This becomes the substrate for both human review AND skill evolution input.

**Implication for cryptotrader-ai**: our `agent_memory/cases/<commit_hash>.md` per-cycle files already capture verdict + risk gate + execution. Extending to a more structured causal chain (data fetch → agent analyses → debate → verdict → execution → realized PnL) is a natural Phase 2 design.

### Pattern M4: Reflection-Write-Back

**Adopted by**: skill-evolution (7-step process).

After failures, the agent itself follows a fixed protocol to write learnings back into the skill file. **No separate "memory database" — skill files ARE the memory.**

**Implication for cryptotrader-ai**: this is the closest pattern to spec 014's reflection job + agent_memory/<agent>/patterns/. Confirms our architecture is in the mainstream.

### Pattern M5: History-Clear-On-Improve

**Adopted by**: EvoSkill uniquely.

After successful improvement, the failure history that drove the improvement is cleared. Otherwise the next iteration's LLM keeps re-rationalizing already-fixed problems.

**Implication for cryptotrader-ai**: when reflection produces a new pattern, the cases that informed it should be marked "consumed" rather than feeding into future reflection. spec 014 doesn't currently do this — worth flagging for spec 018.

---

## Phase 1 — Recommendations Summary (≥ 10 actionable)

These are the candidate inputs for spec 017 (prompt externalization) and a few that already point at spec 018 design choices.

### R1: Use named-section schema for agent prompts (spec 017)
- **Source**: Hermes (`agent/prompt_builder.py`), MetaClaw (`api_server.py`), EvoSkills (13 SKILL.md sections)
- **Affects**: cryptotrader-ai `src/cryptotrader/agents/{tech,chain,news,macro}.py` ROLE strings
- **Action**: Replace single ROLE string with TOML/YAML schema declaring `agent_role`, `available_skills`, `memory_guidance`, `output_schema` as separate keys. Compose at runtime by string concat with section headers.

### R2: Implement progressive skill disclosure schema (spec 017 schema, spec 018 implementation)
- **Source**: SkillClaw (`api_proxy/skill_router.py`), skill-evolution (L1/L2/L3 hierarchy)
- **Affects**: spec 014 `agent_skills/<agent>/SKILL.md` format
- **Action**: Add `level: metadata|body|reference` field. Phase 1 prompt injects only `metadata`; future phase implements body lazy-load.

### R3: Adopt tokenizer-native templates for portability (spec 017 — investigate)
- **Source**: OpenClaw-RL (`/openclaw-rl/utils/template.py`)
- **Affects**: `src/cryptotrader/agents/base.py` prompt construction
- **Action**: Investigate using LangChain's chat-template adapters (already supports HuggingFace tokenizers). Decision: defer — value depends on whether we plan to swap LLM providers.

### R4: Standardize injection point on system prompt (spec 017 decision)
- **Source**: Hermes, MetaClaw, SkillClaw (system); OpenClaw-RL (last-message)
- **Affects**: spec 017's overall prompt strategy
- **Action**: Adopt system-prompt-injection strategy + put per-cycle volatile experience in user message tail to avoid cache invalidation. Document explicitly so cache hit rate stays high.

### R5: Forbid mid-cycle prompt mutation (spec 017 invariant)
- **Source**: Hermes (explicit principle), SkillClaw (sync between sessions only)
- **Affects**: agent_skills loader behavior
- **Action**: Document as a hard invariant. cryptotrader-ai's reflection job already runs between cycles, but the rule should be explicit so future maintainers don't accidentally introduce hot-reload.

### R6: Add explicit token budget enforcer (spec 017)
- **Source**: MetaClaw (`build_prompt` greedy truncation, 20k cap)
- **Affects**: agent prompt assembly
- **Action**: After section composition, count tokens (`tiktoken` or model-specific); if > budget, drop sections in priority order. Configure budget per-model (claude-opus-4-7 has 200k context, glm-5 has 32k).

### R7: Extend SKILL.md frontmatter with declarative triggers (spec 017 schema, spec 018 router)
- **Source**: EvoSkills (`should-trigger` + `allowed_tools`)
- **Affects**: spec 014 `agent_skills/<agent>/<pattern>/SKILL.md` format
- **Action**: Add fields: `regime_tags: [trending, choppy, sideways]`, `triggers_keywords: [...]`. Phase 1 = schema only; Phase 2 (spec 018) implements pre-filter in retrieval.

### R8: Memory units carry importance + access_count (spec 017 schema; spec 018 ranking)
- **Source**: MetaClaw (`MemoryUnit` dataclass)
- **Affects**: `agent_memory/<agent>/patterns/<name>.md` format
- **Action**: Add frontmatter fields `importance: 0-1` (set by reflection) and `access_count: int` (incremented on retrieval).

### R9: Causal-chain session log (spec 018, but schema-decision in 017)
- **Source**: SkillClaw (session DB schema)
- **Affects**: `agent_memory/cases/<commit_hash>.md` content
- **Action**: Extend per-cycle case file from current "verdict + risk_gate" to full causal chain (data → agents → debate → verdict → execute → realized_pnl). Helps Phase 2 reflection.

### R10: History-clear-on-improve (spec 018)
- **Source**: EvoSkill (`feedback_history.md` reset on success)
- **Affects**: spec 014 reflection job's input selection
- **Action**: When reflection produces a new active pattern, mark contributing cases as "consumed" so they're filtered out of future reflection windows. Prevents re-rationalization loops.

### R11: Skill maturity gating signals (spec 018)
- **Source**: skill-evolution (5 maturity signals)
- **Affects**: spec 014 PnL maturity FSM (observed → probationary → active → deprecated)
- **Action**: Compare skill-evolution's 5 signals (sample size, win rate, time, etc.) with our current PnL-only FSM. spec 018 brainstorm should evaluate.

### R12: Two-source skill retrieval (Embedding + keyword) (spec 018)
- **Source**: MetaClaw (Template + Embedding dual mode)
- **Affects**: spec 018's skill retrieval algorithm
- **Action**: Defer to spec 018 brainstorm. Current spec 014 has neither (skills always loaded).

---

## Open Questions for Phase 2

1. **Evolution algorithms** — every project handles this differently (RL in OpenClaw-RL, pairwise eval in EvoSkill, generation counter in MetaClaw, dataset-builder in Hermes). Need full Phase 2 to compare.
2. ~~**autoresearch fit**~~ — ✅ Resolved 2026-05-08: replaced with `microsoft/autogen` per D-PROC-01.
3. **EvoSkill (singular) vs EvoSkills (plural)** — confirmed entirely different projects. Should Phase 2 keep both, or drop one for budget?
4. **autogen evolution gap** — autogen contributes nothing to Phase 2's evolution-algorithm research (pure orchestration framework). The remaining 7 projects must carry that load.
