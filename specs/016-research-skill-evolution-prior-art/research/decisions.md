# Architecture Decision Records

**Status**: Phase 1 partial — only ADRs related to Prompt Assembly + Memory wiring are recorded here. Phase 2 will append more.

ADRs follow a lightweight format: Status / Context / Decision / Consequences. Each is keyed `D-PA-NN` (Phase 1 = prompt assembly), `D-MW-NN` (Phase 1 = memory wiring), or `D-PROC-NN` (process / scope decisions).

---

## D-PROC-01: Replace autoresearch with microsoft/autogen

**Status**: ✅ Accepted (2026-05-08)

**Context**: The `autoresearch` project (uditgoenka, MIT) was researched at Tier 3 scan depth. The Phase 1 finding was that this project's "skill" + "memory" concepts do not map cleanly to LLM agent skill evolution — it's a git-history-based research-iteration framework with static Markdown procedure files, no dynamic skill loading, no memory layer beyond git log. comparison-matrix.md showed 4 of 8 columns marked N/A for autoresearch.

**Decision**: Replace autoresearch with **microsoft/autogen** at Tier 2 close-read depth.

**Rationale for autogen over OpenHands/SWE-agent**:
- Multi-agent coordination + per-agent state management directly mirror cryptotrader-ai's 4-agent + debate + verdict architecture
- `Memory.update_context()` injection hook + `ChatCompletionContext` history variants are exactly the abstractions spec 017 needs to design
- OpenHands is more SWE-focused; the indirection cost would be higher

**Consequences**:
- ✅ autogen.md created at Tier 2 (179 lines, MIT license)
- ✅ Original `autoresearch.md` renamed to `_deferred-autoresearch.md` (preserved for reference, not in active study set)
- ✅ comparison-matrix.md and synthesis.md updated to use autogen as the 8th project
- Phase 2 evolution-algorithm research unchanged (autogen has no skill evolution; that part still relies on SkillClaw / EvoSkills / OpenClaw-RL / EvoSkill / skill-evolution)
- Phase 1 scope: 8 projects fully covered, no deferred-N/A rows

---

## D-PA-01: Use named-section schema for agent prompts (spec 017)

**Status**: Accepted (Phase 1)

**Context**: 4 of 8 researched projects (Hermes, MetaClaw, EvoSkills, SkillClaw partial) all use named-section composition rather than monolithic strings. cryptotrader-ai's current `agents/{tech,chain,news,macro}.py` ROLE strings are 30-60 line monolithic constants — hard to evolve, hard to budget, hard to reuse.

**Decision**: Adopt named-section schema. Each agent's prompt will be specified by a configuration file (TOML/YAML/MD frontmatter — exact format is spec 017's call) that declares sections like:
- `agent_role` — agent identity/role description (longest-lived, most cacheable)
- `available_skills` — current skill catalog snippet (per-cycle)
- `memory_guidance` — how to use memory (relatively fixed)
- `output_schema` — required JSON shape (very stable)
- `domain_checklist` — agent-specific verification rules

At runtime the agent prompt = ordered concat of section bodies with explicit headers.

**Consequences**:
- spec 017 must define section schema + composition order
- Each section becomes individually editable / token-budgeted / cacheable
- Loose coupling allows future skill-aware sections without rewriting whole ROLE
- Slight overhead (config file parsing) on agent init — acceptable

---

## D-PA-02: Inject persistent context to system prompt; per-cycle volatile to user message tail

**Status**: Accepted (Phase 1)

**Context**: Pattern P4 in synthesis.md showed two camps: system-prompt injection (Hermes / MetaClaw / SkillClaw) vs last-message injection (OpenClaw-RL). cryptotrader-ai uses Anthropic prompt cache (spec 004); cache hit requires stable system prompt prefix.

**Decision**: Adopt **hybrid**:
- Stable items (agent_role, available_skills, memory_guidance, output_schema) → system prompt
- Per-cycle volatile (latest market data, recent verdict commit hashes, current portfolio state) → tail of user message

This keeps prompt cache hit rate high while still updating context per cycle.

**Consequences**:
- spec 017 must define section assignment to each prompt slot (system vs user tail)
- Cache hit rate measurable (telemetry already exists post-spec 015 fixes)
- If a section is misassigned (e.g. "available_skills" goes in user tail), cache misses 100% — discoverable via metrics

---

## D-PA-03: Forbid mid-cycle prompt mutation

**Status**: Accepted (Phase 1)

**Context**: Hermes explicitly forbids hot-swap of skills mid-conversation; SkillClaw only refreshes via `dashboard sync` between sessions; skill-evolution uses git as the swap point. Our spec 014 reflection job already runs between trading cycles. Without an explicit invariant, future code might accidentally introduce hot-reload that breaks prompt cache + confuses agent.

**Decision**: Codify as a hard invariant: **agent prompts compose at cycle start, never mutate within a cycle**. spec 017 will document this explicitly. Implementations MUST assert this (e.g., the prompt builder is called once per cycle and the result is treated as immutable).

**Consequences**:
- Reflection job stays scheduled-between-cycles (already the case)
- Agent_skills file changes from external editor don't affect in-flight cycle
- Predictable prompt cache behavior

---

## D-PA-04: Defer tokenizer-native templates (R3) to spec 017 brainstorm investigation

**Status**: Deferred

**Context**: OpenClaw-RL's tokenizer-native rendering (HF Jinja2 chat templates) makes prompt format model-portable for free. cryptotrader-ai uses LangChain ChatOpenAI primarily (model selection via config); LangChain has its own chat-template abstraction.

**Decision**: spec 017 brainstorm will evaluate whether to use LangChain's tokenizer-aware adapters or stay with f-string composition. Decision drivers: (a) how often we plan to swap LLM providers, (b) how much friction LangChain template adapter adds.

**Consequences**:
- No commitment now — investigation in spec 017 brainstorm
- If adopted, prompt rendering becomes more portable but introduces a layer of indirection
- If deferred, prompt format stays per-provider; switching providers requires re-engineering

---

## D-PA-05: Add token budget enforcer to prompt builder

**Status**: Accepted (Phase 1)

**Context**: MetaClaw's 20k token greedy truncation is the only project with explicit prompt-level budget enforcement. cryptotrader-ai currently has no agent-level enforcer (only LLM-call-level retry on overflow, which is wasteful).

**Decision**: spec 017 will define a token budget enforcer in the prompt builder. Configurable per-model (e.g., glm-5 32k → 24k budget, claude-opus-4-7 200k → 64k budget — leaves room for response). When exceeded, drop sections in priority order: `output_schema` (required, never drops) > `agent_role` (required) > `domain_checklist` > `available_skills` (degrade by lazy-load) > `memory_guidance` (degrade by ranking) > recent_cases (drop oldest first).

**Consequences**:
- Predictable behavior at scale (100+ skills)
- Avoids LLM-side context-overflow errors
- Telemetry should expose "prompt_size / budget" per cycle

---

## D-PA-06: Skill metadata schema extension (declarative triggers)

**Status**: Accepted (Phase 1, schema only; logic in spec 018)

**Context**: EvoSkills uses `allowed_tools:` and `should-trigger:` frontmatter for declarative routing. SkillClaw's catalog is YAML-only at startup. Both decouple "WHEN to apply this skill" from "WHAT the skill says" — useful for pre-filter retrieval (avoid LLM round-trip just to check which skill to load).

**Decision**: spec 017 will extend `agent_skills/<agent>/<pattern>/SKILL.md` frontmatter (already exists from spec 014) with:
- `regime_tags: [trending, choppy, sideways, breakout, reversal]` (multi-select)
- `triggers_keywords: [breakout_short, sma_breakdown, …]` (free-form)
- `level: metadata|body|reference` (for future progressive disclosure)

Phase 1 = schema only. spec 018 implements regime-tag pre-filter in retrieval.

**Consequences**:
- Existing 27 SKILL.md files need migration (add new frontmatter fields, default to `regime_tags: []`)
- Migration is one-shot script; spec 017 plan can include it
- spec 018 retrieval gains a fast-path

---

## D-MW-01: Memory unit metadata = importance + access_count

**Status**: Accepted (Phase 1, schema only)

**Context**: MetaClaw uniquely tracks `importance` + `access_count` per memory unit. Without these, retrieval ranking is ad-hoc. cryptotrader-ai's current `agent_memory/<agent>/patterns/<name>.md` only has PnL-track + maturity stage.

**Decision**: spec 017 will extend memory.md frontmatter with:
- `importance: 0.0–1.0` (set by reflection — high if pattern produced consistent PnL)
- `access_count: int` (incremented every time the pattern is injected into a prompt)
- `last_accessed_at: ISO8601` (for time-decay ranking in spec 018)

**Consequences**:
- Existing patterns need migration (default importance = 0.5, access_count = 0)
- spec 018 ranking algorithm gains material to work with
- `importance` field becomes a proxy for "human override" — if user edits importance, reflection should respect it

---

## D-MW-02: Causal-chain session log (extend spec 014 case files)

**Status**: Deferred (decided in spec 018 brainstorm; implementation in spec 018)

**Context**: SkillClaw uniquely captures sessions as causal chains (user_prompt → tool_calls → intermediate_feedback → final_answer). Our `agent_memory/cases/<commit_hash>.md` currently captures verdict + risk gate + execution but not full causal chain. Extending it would help Phase 2 reflection but bloats per-cycle storage.

**Decision**: Defer concrete schema to spec 018 brainstorm — but flag as "high-value lever". Phase 2 must compare SkillClaw's session DB structure to our case file structure.

**Consequences**:
- No Phase 1 action
- spec 018 brainstorm has a clear research target

---

## D-MW-03: History-clear-on-improve

**Status**: Deferred (decided in spec 018)

**Context**: EvoSkill clears `feedback_history.md` after a successful improvement, preventing re-rationalization loops. cryptotrader-ai's reflection job currently keeps all historical cases in scope, including those that already informed previous patterns.

**Decision**: Defer to spec 018 brainstorm — implementation requires consensus on what "consumed" means in our PnL-driven model.

**Consequences**:
- Phase 1: no action
- spec 018: must address (otherwise reflection loops on already-fixed patterns)

---

## Phase 2 — TBD ADRs

The following pattern-categories will produce ADRs during Phase 2 research:

- Evolution algorithms (RL vs pairwise vs dataset-builder vs maturity FSM)
- Skill data structures (Markdown vs JSON vs DSL)
- Retrieval algorithms (keyword + embedding vs graph traversal vs LLM-routed)
- Evaluation methods (verifier reward vs PnL vs human-in-loop)
- Agent ↔ Skill boundary (per-agent vs cross-agent)
- Engineering (file sync vs DB vs in-memory)

These are placeholders only — actual decisions wait for Phase 2 deep reads.
