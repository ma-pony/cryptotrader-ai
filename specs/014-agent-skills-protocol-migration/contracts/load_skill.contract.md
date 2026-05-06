# `load_skill` Tool Contract

## Purpose

允许 agent 在 reasoning 中重新加载已注入的 SKILL.md（例如 multi-turn 对话中需要再次查阅）。也作为普通 Python 函数提供给不走 `create_agent` 的代码（verdict 节点、curation 流程）。

双层架构下，`load_skill` 仅返回**整理后的 SKILL.md body**，不返回内部 patterns / cases 的内容（后者在 memory 层，agent 不直接访问）。

## Identity

- **Tool name**: `load_skill`
- **Source**: `src/cryptotrader/agents/skills/tool.py`
- **Registered via**: `SkillsInjectionMiddleware.tools = [load_skill_tool]`
- **Type**: LangChain `BaseTool`（Pydantic args schema）
- **Also exposed as**: `load_skill(name: str) -> dict` 普通 Python 函数

## Input

```python
class LoadSkillInput(BaseModel):
    name: str = Field(
        description=(
            "Skill name. One of: tech-analysis, chain-analysis, "
            "news-analysis, macro-analysis, trading-knowledge."
        )
    )
```

## Output (success)

```json
{
  "name": "tech-analysis",
  "body": "# Tech Analysis\n\n## Role\n...\n\n## Active Patterns\n..."
}
```

## Output (errors)

| Error | Response | Cause |
|---|---|---|
| Skill not found | `{"error": "skill_not_found", "name": "..."}` | name 不在 5 个允许值中，或对应 SKILL.md 文件缺失 |
| Corrupt frontmatter | `{"error": "corrupt_file", "path": "...", "details": "..."}` | YAML 解析失败 |
| Per-cycle rate limit | `{"error": "rate_limit_per_cycle", "limit": 10}` | 同 trace_id 一个 cycle 内 ≥ 10 次调用 |
| Skill dir missing | `{"error": "skill_dir_missing"}` | `agent_skills/` 顶级目录不存在 |

## Resolution Algorithm

```python
def load_skill(name: str) -> dict:
    base = Path("agent_skills")
    if not base.exists():
        return {"error": "skill_dir_missing"}
    if name not in {"tech-analysis", "chain-analysis", "news-analysis",
                    "macro-analysis", "trading-knowledge"}:
        return {"error": "skill_not_found", "name": name}
    file_path = base / name / "SKILL.md"
    if not file_path.exists():
        return {"error": "skill_not_found", "name": name}
    try:
        content = file_path.read_text()
        frontmatter, body = parse_frontmatter(content)
        return {"name": name, "body": body}
    except yaml.YAMLError as e:
        return {"error": "corrupt_file", "path": str(file_path), "details": str(e)}
```

## Performance

- p95 ≤ 30 ms（单个 SKILL.md 文件读 + YAML parse；文件 ≤ 50KB）
- 进程内 LRU 缓存（max 10 条）

## Side Effects

- 增加 per-cycle 调用计数（rate-limit 用）
- Read-only：不写 `agent_skills/`
- INFO 日志每次调用：`load_skill called: name=<n> result=<ok|err>`

## Concurrency

- 多 agent 节点并发调 `load_skill` 安全（read-only + 进程内锁保护缓存）
- curation 写 SKILL.md 时使用临时文件 + `os.rename` 原子写；`load_skill` 读不会拿到部分写入

## Test Scenarios（covered by `tests/test_load_skill_tool.py`）

| Scenario | Input | Expected |
|---|---|---|
| Skill exists | `load_skill("tech-analysis")` | `{name, body, ...}` |
| Skill missing | `load_skill("nonexistent")` | `{"error": "skill_not_found", ...}` |
| Skill 文件 frontmatter 损坏 | YAML 错误 | `{"error": "corrupt_file", ...}` |
| Rate limit | 第 11 次调用同 cycle | `{"error": "rate_limit_per_cycle", ...}` |
