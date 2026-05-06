# `SkillsInjectionMiddleware` Contract

## Purpose

Hook into LangChain `create_agent` to **statically inject** an agent's available skill descriptions into `request.system_message.content_blocks` before each LLM call. Concurrently registers the `load_skill` tool so agent can fetch full body on demand.

## Class Signature

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable

class SkillsInjectionMiddleware(AgentMiddleware):
    """Per-agent middleware that loads agent_skills/{agent_id}/ + shared/ and
    injects regime-filtered description list into the system prompt.

    Reference impl: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant
    """

    tools = [load_skill_tool]    # registered class-level for create_agent to pick up

    def __init__(self, agent_id: str, skill_dir: Path | None = None):
        self.agent_id = agent_id    # one of: tech, chain, news, macro
        self.skill_dir = skill_dir or Path("agent_skills")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        ...
```

## `wrap_model_call` Behavior

```
def wrap_model_call(request, handler):
    # 1. Extract regime tags from runtime context
    regime_tags = request.runtime_context.get("regime_tags", [])

    # 2. Load agent skill set (regime-filtered for patterns/forbidden, all for instructions/shared)
    skill_set = load_agent_skills(self.agent_id, regime_tags, base_dir=self.skill_dir)

    # 3. Render to markdown text addendum
    addendum = self._render_addendum(skill_set)

    # 4. Append to system_message.content_blocks
    new_content_blocks = list(request.system_message.content_blocks) + [
        {"type": "text", "text": addendum}
    ]
    modified_request = request.model_copy(
        update={"system_message": request.system_message.model_copy(
            update={"content_blocks": new_content_blocks}
        )}
    )

    # 5. Forward
    return handler(modified_request)
```

## Rendered Addendum Structure

```markdown


## Agent Instructions

{instructions.body}

## Available Patterns ({len(patterns)} matched current regime: {regime_tags})

- **tech::funding_squeeze_long**: When funding rate < -0.0001 and price near 30D low, expect bullish reversion within 24h. (12 cases, win_rate 0.67)
- **tech::macd_divergence_short**: ...

## Forbidden Zones ({len(forbidden)} matched)

- **tech::chase_breakout_low_volume**: Don't long breakouts with volume ratio < 0.2. (Loss rate 0.70 over 8 cases)

## Shared Knowledge

- **shared::funding_rate**: funding > 0.0003 = crowded longs; < -0.0001 = crowded shorts; 0 = neutral
- **shared::regime_definitions**: ...

## Loading Rule

Use the `load_skill(name)` tool to retrieve the full body of any item above. Use `<agent>::<name>` form to disambiguate cross-agent references. In your final reasoning, explicitly cite which skills you applied via `applied: <name>` (e.g., `applied: tech::funding_squeeze_long`).
```

**Notes**:
- Names are always emitted with `agent::` prefix in the addendum to enforce unambiguous reference (R3 decision)
- Description includes parenthetical PnL hint (`12 cases, win_rate 0.67`) to help agent prioritize active patterns over probationary
- `## Agent Instructions` only appears if `instructions.md` exists for this agent

## Regime Filter Rule

For `patterns` and `forbidden` lists:
- skill is included if `set(skill.regime_tags) & set(current_regime_tags)` is non-empty
- skill is included if `skill.regime_tags == []` (wildcard)
- skill is excluded otherwise

For `instructions` and `shared/`: always included (no filtering).

## Performance

- Cold load: p95 ≤ 100 ms (per agent, ~50 files)
- Cached: p95 ≤ 5 ms (skill set cached per (agent_id, regime_hash) within one cycle)
- Cache invalidation: any reflection write triggers global cache flush

## Concurrency

- 4 agents may instantiate 4 separate `SkillsInjectionMiddleware` instances; each holds its own cache
- All instances coordinate with reflection writer via `fcntl.flock` shared lock when reading

## Failure Modes

| Failure | Behavior |
|---|---|
| `agent_skills/{agent_id}/` missing entirely | Return empty AgentSkillSet; addendum has no patterns/forbidden sections; logger.warning |
| 单个 frontmatter 损坏 | Skip that file, logger.warning, continue with rest |
| `instructions.md` 缺失 | Omit `## Agent Instructions` section; agent runs with only `system_prompt=` content + skills addendum |
| `regime_tags` 为空（snapshot 缺失） | Treat as wildcard match — include all patterns/forbidden |

## Registration Example

```python
from langchain.agents import create_agent
from cryptotrader.agents.skills.middleware import SkillsInjectionMiddleware

# In agents/base.py:ToolAgent.create_agent
agent = create_agent(
    model=llm,
    system_prompt=base_system,    # Existing role description
    tools=[*self.tools, *SkillsInjectionMiddleware.tools],
    middleware=[SkillsInjectionMiddleware(agent_id=self.agent_id)],
)
```

## Test Scenarios (covered by `tests/test_skills_middleware.py`)

| Scenario | Setup | Expected |
|---|---|---|
| Regime filter applies to patterns | 10 patterns mixed regimes; current regime `[range_bound]` | Only range_bound + wildcard injected |
| Shared/ included for all agents | shared/funding_rate.md exists | All 4 agent prompts include it |
| Missing instructions.md | tech/instructions.md deleted | tech prompt omits `## Agent Instructions`, no crash |
| Corrupt frontmatter | Invalid YAML in 1 file | Skipped + warning; other files OK |
| `agent::` prefix in render | Description rendered | Always `tech::name` format, never bare `name` |
