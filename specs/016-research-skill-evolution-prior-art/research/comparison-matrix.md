# Comparison Matrix

**Status**: Phase 1 partial — only "Prompt Assembly" + "Memory ↔ Skill" columns filled. Other 6 columns remain `_Phase 2_` placeholders.

Cell format: `<one-sentence finding>` + `path/to/file.ext` (link to source). N/A cells use "N/A — <reason>".

| Project | Evolution Algo | Skill Data Struct | Retrieval | **Memory ↔ Skill** ✓ | **Prompt Assembly** ✓ | Evaluation | Agent ↔ Skill Boundary | Engineering |
|---|---|---|---|---|---|---|---|---|
| **SkillClaw** | _Phase 2_ | _Phase 2_ | _Phase 2_ | Causal-chain session DB feeds skill evolution; skills stored as Markdown files locally + S3, Task Ledger persists cross-session goals · `api_proxy/skill_router.py` | Two-stage: catalog YAML in system prompt at start → full SKILL.md appended on hit; `push_min_injections=5` quality gate · `api_proxy/proxy.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **MetaClaw** | _Phase 2_ | _Phase 2_ | _Phase 2_ | 6 categorized MemoryUnit types with importance + access_count; rendered grouped into single system message via `render_for_prompt()` · `memory/units.py` | 3 modes (Synergy / Memory-only / Skills-only); 20k token greedy truncation; sectioned system message · `api_server.py:build_prompt()` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **OpenClaw-RL** | _Phase 2_ | _Phase 2_ | _Phase 2_ | In-memory dicts (`_session_conversations`, `_session_experience`); no persistence; OEL extracts experience async via `loop.create_task()` · `openclaw_rl/oel.py` | Tokenizer-native via `tokenizer.apply_chat_template()` (Jinja2); experience injected ONLY into last user message via `_inject_experience_to_messages()` · `openclaw_rl/utils/template.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **Hermes** | _Phase 2_ | _Phase 2_ | _Phase 2_ | `SessionDB` in `hermes_state.py` dual-purpose: feeds evolution dataset + injects memory blocks into fixed system-prompt section · `agent/hermes_state.py` | Named sections (`MEMORY_GUIDANCE`, `SKILLS_GUIDANCE`, `SESSION_SEARCH_GUIDANCE`); some DSPy-evolvable, some fixed; SKILL.md as user msg at session init, never hot-swapped, ≤15KB · `agent/prompt_builder.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **EvoSkill** | _Phase 2_ | _Phase 2_ | _Phase 2_ | `feedback_history.md` is the memory; cleared after successful improvement; `related_iterations` field traces back · `evoskill/proposer.py` | Proposer LLM gets failure samples + history + skills as ordered prompt; output is `prompt\|skill` binary route then Generator writes to `.claude/skills/` · `evoskill/proposer.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **EvoSkills** | _Phase 2_ | _Phase 2_ | _Phase 2_ | Dual storage: M_I (direction-level) + M_E (strategy-level); top-k cosine similarity into current prompt · `memory/store.py` | 13 SKILL.md files with `allowed_tools` whitelist + `should-trigger` declarative routing; SKILL.md IS the prompt body · `skills/*.md` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **skill-evolution** | _Phase 2_ | _Phase 2_ | _Phase 2_ | No separate memory module; reflection writes back into SKILL.md or scripts via 7-step process; "deterministic ladder" delineates LLM vs scripts · `docs/protocol.md` | 3-level hierarchy: L1 metadata always in context, L2 SKILL.md on trigger, L3 `references/` files on demand · `loader/progressive.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |
| **microsoft/autogen** | N/A — pure orchestration framework, no skill evolution concept | N/A — tools are statically registered, no skill abstraction | _Phase 2_ | `Memory.update_context()` is the official injection hook; chat history flows through `ChatCompletionContext` (variants for buffered / head-tail / token-trimmed) · `python/packages/autogen-core/src/autogen_core/memory/`, `python/packages/autogen-core/src/autogen_core/model_context/` | **Four-slot separation**: `_system_messages` (static role) + `_tools` (schema) + `Memory.update_context()` injection + `ChatCompletionContext` history; assembly order = SystemMessage → memory-injected content → history + new turn; system prompt stays out of context (cache-friendly) · `python/packages/autogen-agentchat/src/autogen_agentchat/agents/_assistant_agent.py` | _Phase 2_ | _Phase 2_ | _Phase 2_ |

---

## Status legend

- ✓ — Phase 1 complete for this column (Prompt Assembly + Memory ↔ Skill)
- _Phase 2_ — Deferred to Phase 2 research
- N/A — <reason> — Project does not exhibit this property; reason explains

## Notes

- **autoresearch swap**: autoresearch was deferred from active study set per D-PROC-01 (only 2 of 8 columns produced signal); replaced by **microsoft/autogen** at Tier 2. Original `autoresearch.md` retained as `_deferred-autoresearch.md` for reference.
- **autogen contribution boundary**: autogen has zero skill-evolution concept (pure orchestration framework). Its value is concentrated in Prompt Assembly + Memory wiring (relevant for spec 017). Phase 2 evolution-algorithm research will continue to rely on SkillClaw / EvoSkills / OpenClaw-RL / EvoSkill / skill-evolution.
- **License coverage**: All 8 active projects use permissive licenses (MIT or Apache-2.0). No GPL contamination risk for spec 018 fork-borrow recommendations.
- **Phase 1 scope confirmed**: 2 columns × 8 projects = 16 cells filled. Remaining 48 cells (6 columns × 8 projects) await Phase 2.
