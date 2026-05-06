# `load_skill` Tool Contract

## Purpose

Allow agent to fetch the **full markdown body** of a Skill at runtime, after seeing only the description in the static prompt injection. Implements the "on-demand body retrieval" half of the Anthropic Skills protocol pattern (paired with description-only static injection in `SkillsInjectionMiddleware.wrap_model_call`).

## Identity

- **Tool name**: `load_skill`
- **Source**: `src/cryptotrader/agents/skills/tool.py`
- **Registered via**: `SkillsInjectionMiddleware.tools = [load_skill_tool]`
- **Type**: LangChain `BaseTool` (Pydantic args schema)
- **Also exposed as**: a plain Python function `load_skill(name: str, requesting_agent: str | None = None) -> dict` for callers that don't go through `create_agent` (e.g., `verdict` node, `learning/skills.py`)

## Input

```python
class LoadSkillInput(BaseModel):
    name: str = Field(
        description=(
            "Skill identifier. Either '<name>' (resolved within calling agent's own dir) "
            "or '<agent>::<name>' (cross-agent explicit). Names are snake_case."
        )
    )
```

## Output (success)

```json
{
  "name": "funding_squeeze_long",
  "agent": "tech",
  "kind": "pattern",
  "body": "# Pattern: Funding Squeeze Long\n\n## When to apply\n..."
}
```

## Output (errors)

| Error | Response | Cause |
|---|---|---|
| Skill not found | `{"error": "skill_not_found", "name": "..."}` | No file matches `name` in resolved agent dir; or `agent::name` resolves to non-existent path |
| Ambiguous short name | `{"error": "ambiguous_name", "candidates": ["tech::x", "chain::x"]}` | Short `name` exists in ≥ 2 agent dirs |
| Corrupt frontmatter | `{"error": "corrupt_file", "path": "...", "details": "<yaml parse error>"}` | YAML parse failed; body cannot be returned reliably |
| Per-cycle rate limit | `{"error": "rate_limit_per_cycle", "limit": 10}` | Same trace_id has called load_skill ≥ 10 times this cycle (hallucination loop guard) |

## Resolution Algorithm

```
def load_skill(name, requesting_agent=None):
    if "::" in name:
        agent, skill_name = name.split("::", 1)
        path = agent_skills/{agent}/**/{skill_name}.md
        if not exists: return skill_not_found
        return read(path)
    else:
        # Short form
        if requesting_agent:
            local = agent_skills/{requesting_agent}/**/{name}.md
            if exists: return read(local)
        # Fall through: scan all agents
        matches = [agent for agent in [tech, chain, news, macro, shared]
                   if agent_skills/{agent}/**/{name}.md exists]
        if len(matches) == 0: return skill_not_found
        if len(matches) > 1: return ambiguous_name
        return read(matches[0])
```

## Performance

- p95 ≤ 50 ms (single file read + YAML parse + body extract)
- File reads use `fcntl.flock` shared lock (LOCK_SH) to coordinate with reflection writer

## Side Effects

- Increments per-cycle counter (keyed by trace_id) used for rate-limiting
- Read-only: never writes to `agent_skills/`
- Logs INFO on each call: `load_skill called: name=<n> agent=<a> result=<ok|err>`

## Concurrency

- Multiple analysis agents may call `load_skill` concurrently (parallel via `asyncio.gather`)
- Reflection writer holds exclusive lock; `load_skill` will briefly block until writer releases (typical < 50 ms)

## Test Scenarios (covered by `tests/test_load_skill_tool.py`)

| Scenario | Input | Expected |
|---|---|---|
| Local skill exists | `load_skill("funding_squeeze_long")` from tech | `{name, agent: "tech", body, ...}` |
| Cross-agent explicit | `load_skill("tech::funding_squeeze_long")` from verdict | `{name, agent: "tech", body, ...}` |
| Skill missing | `load_skill("nonexistent")` | `{"error": "skill_not_found", ...}` |
| Ambiguous short | `load_skill("common_name")` (exists in 2 dirs) | `{"error": "ambiguous_name", ...}` |
| Corrupt yaml | File with bad frontmatter | `{"error": "corrupt_file", ...}` |
| Rate limit | 11th call same cycle | `{"error": "rate_limit_per_cycle", ...}` |
