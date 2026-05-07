# `SkillsInjectionMiddleware` Contract

## Purpose

把 5 个 SKILL.md 中**对应该 agent 的 1 个**和 **trading-knowledge 共享 SKILL.md** 注入到 `request.system_message.content_blocks`，每次 LLM 调用前自动执行。同时注册 `load_skill` tool，agent 可重新拉取任一 skill body。

双层架构下，middleware 不再加载 patterns/cases —— 那些在 memory 层（agent 看不到）。SKILL.md 已经是 curation 整理后的能力包，直接注入即可。

## Class Signature

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

class SkillsInjectionMiddleware(AgentMiddleware):
    """Per-agent middleware that discovers all SKILL.md under agent_skills/,
    filters by frontmatter `scope`, and injects matched skills into the
    system message before LLM call.

    No hardcoded skill→agent mapping — discovery is runtime, based on
    each SKILL.md's `scope` frontmatter field. New skills added by user
    or `arena skills propose-new` are picked up automatically next cycle.

    Reference: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant
    """

    tools = [load_skill_tool]    # registered for create_agent to pick up

    VALID_AGENT_IDS = {"tech", "chain", "news", "macro"}

    def __init__(self, agent_id: str, skill_dir: Path | None = None):
        assert agent_id in self.VALID_AGENT_IDS
        self.agent_id = agent_id
        self.skill_dir = skill_dir or Path("agent_skills")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        ...
```

## Behavior

```python
def wrap_model_call(self, request, handler):
    # 1. Discover all SKILL.md, filter by scope
    matched_skills = discover_skills_for_agent(
        agent_id=self.agent_id,
        skill_dir=self.skill_dir,
    )    # returns list[Skill] where scope == "shared" OR scope == f"agent:{self.agent_id}"

    # 2. Compose addendum (skip individual failures, log warning)
    parts = []
    for skill in matched_skills:
        try:
            content = skill.body
            parts.append(f"<!-- skill: {skill.name} ({skill.scope}) -->\n\n{content}")
        except Exception as e:
            logger.warning("Failed to render %s SKILL.md: %s", skill.name, e)

    if not parts:
        return handler(request)    # No skills loaded → original prompt

    addendum = "\n\n".join(parts)

    # 3. Append to system_message.content_blocks
    new_blocks = list(request.system_message.content_blocks) + [
        {"type": "text", "text": addendum}
    ]
    modified = request.model_copy(
        update={"system_message": request.system_message.model_copy(
            update={"content_blocks": new_blocks}
        )}
    )

    # 4. Forward
    return handler(modified)


def discover_skills_for_agent(agent_id: str, skill_dir: Path) -> list[Skill]:
    """Scan agent_skills/*/SKILL.md, parse frontmatter, filter by scope.

    Cached by mtime of skill_dir + each SKILL.md. Cache invalidated on any
    file change (e.g., after curation or propose-new manual save).
    """
    skills = []
    for skill_md in skill_dir.glob("*/SKILL.md"):
        try:
            skill = parse_skill_md(skill_md)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", skill_md, e)
            continue
        if skill.scope == "shared" or skill.scope == f"agent:{agent_id}":
            skills.append(skill)
    return skills
```

## SKILL.md Body Layout（curation 整理后产物示例）

```markdown
# Tech Analysis

## Role
You are the technical analysis agent for crypto perpetual futures...

## Active Patterns

These patterns have been distilled from the last 60 days of trading and validated against PnL feedback.

- **funding_squeeze_long**: When funding rate < -0.0001 and price near 30D low, expect bullish reversion within 24h. (12 cases, win_rate 0.67, regime: low_funding + range_bound)
- **macd_divergence_short**: ... (8 cases, win_rate 0.62, regime: trending_down)

## Forbidden Zones

- **chase_breakout_low_volume**: Don't long breakouts with volume ratio < 0.2. (8 cases, loss rate 0.70)

## Usage

In your analysis, explicitly cite which pattern(s) you're applying via `applied: <pattern_name>` (e.g., `applied: funding_squeeze_long`). This enables precise PnL attribution for reflection.

If you need more detail on any pattern, use the `load_skill("tech-analysis")` tool to re-read the full SKILL.md.
```

## Failure Modes

| Failure | Behavior |
|---|---|
| Own SKILL.md 缺失 | logger.warning；只注入 shared；agent 仍可工作（基于 system_prompt 原内容）|
| Shared SKILL.md 缺失 | logger.warning；只注入 own |
| 两者都缺失 | logger.warning；不修改 request，原样调 handler |
| Frontmatter 损坏 | logger.warning；视作加载失败 |
| `agent_skills/` 顶级目录缺失 | 第一次启动场景；middleware 调用一次 `ensure_skill_dirs()` 自动创建 |

## Performance

- 冷加载（首次调用）p95 ≤ 50 ms（2 个 SKILL.md 读 + parse）
- 缓存命中 p95 ≤ 5 ms（进程内 LRU）
- 缓存失效条件：curation 写文件后通过 `mtime` 检测自动失效

## Concurrency

- 4 个 agent 节点 4 个 middleware 实例，共享 `load_skill` 实现的进程内缓存
- `threading.Lock` 保护缓存
- curation 写入文件用临时文件 + atomic rename；middleware 读不会拿到部分写入

## Registration Example

```python
# In agents/base.py:ToolAgent.create_agent
from cryptotrader.agents.skills.middleware import SkillsInjectionMiddleware

agent = create_agent(
    model=llm,
    system_prompt=base_role,    # 已有简短 role
    tools=[*self.tools, *SkillsInjectionMiddleware.tools],
    middleware=[SkillsInjectionMiddleware(agent_id=self.agent_id)],
)
```

## Test Scenarios（covered by `tests/test_skills_middleware.py`）

| Scenario | Setup | Expected |
|---|---|---|
| Own + shared 都成功加载 | 5 SKILL.md 都正常 | system_message 含两段 body |
| Own 缺失 | tech-analysis/SKILL.md 删除 | warning + 只注入 shared，cycle 不崩 |
| Shared 缺失 | trading-knowledge/SKILL.md 删除 | warning + 只注入 own |
| Frontmatter 损坏 | YAML 错误 | warning + 跳过该文件 |
| `agent_skills/` 顶级目录缺失 | 首次启动 | 自动创建目录骨架 |
