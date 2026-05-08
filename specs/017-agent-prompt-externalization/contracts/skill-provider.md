# Contract：SkillProvider Protocol

**模块路径**：`src/cryptotrader/agents/prompt_builder.py`（与 PromptBuilder 同模块）

## Protocol 定义

```python
from typing import Protocol


class SkillProvider(Protocol):
    def get_available_skills(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> list[Skill]:
        """返回 ranked skill 列表。

        Args:
            agent_id: 当前 agent 标识（用于过滤 SKILL.md tags）
            snapshot: 当前市场快照（实现方可用其做相关性匹配）
            k: top-k 截断

        Returns:
            list[Skill]，长度 ≤ k；空返回 []。
            实现方负责内部 ranking。
        """
        ...
```

## Skill 数据结构

```python
@dataclass
class Skill:
    skill_id: str
    description: str
    tags: list[str]
    steps: list[str]
    body: str  # 完整 SKILL.md body；本 spec 不直接使用
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `skill_id` | str | 唯一标识（如 `funding-rate-extreme-fade`），通常等于目录名 |
| `description` | str | 一句话描述 |
| `tags` | list[str] | 关联 agent / 主题（如 `["macro", "funding-rate"]`） |
| `steps` | list[str] | 关键步骤摘要（用于 markdown bullet 渲染） |
| `body` | str | 完整 SKILL.md body（spec 018 用于细粒度匹配） |

## 默认实现：DefaultSkillProvider

```python
class DefaultSkillProvider:
    def __init__(self, skills_root: Path = Path("agent_skills")) -> None:
        self._root = skills_root
        self._cache: list[Skill] | None = None

    def get_available_skills(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> list[Skill]:
        skills = self._load_all()
        # 简单 keyword match：agent_id 出现在 tags 中即入选
        relevant = [s for s in skills if agent_id in s.tags]
        # 本 spec 不做进一步排名（spec 018 引入 IDF / Hermes match-score）
        return relevant[:k]

    def _load_all(self) -> list[Skill]:
        if self._cache is not None:
            return self._cache
        if not self._root.exists():
            self._cache = []
            return []
        skills: list[Skill] = []
        for skill_dir in self._root.iterdir():
            if not skill_dir.is_dir():
                continue
            md_path = skill_dir / "SKILL.md"
            if not md_path.exists():
                continue
            try:
                skill = self._parse_skill_md(md_path)
                skills.append(skill)
            except Exception as e:
                logger.warning("解析 %s 失败: %s", md_path, e)
        self._cache = skills
        return skills
```

### 输入

| 参数 | 类型 | 说明 |
|---|---|---|
| `agent_id` | str | 用于匹配 SKILL.md frontmatter `tags` |
| `snapshot` | dict | 本默认实现忽略；spec 018 进化版会用其做相关性匹配 |
| `k` | int | top-k 截断 |

### 输出（示例）

```python
[
    Skill(
        skill_id="funding-rate-extreme-fade",
        description="资金费率极端时反向交易",
        tags=["macro", "funding-rate"],
        steps=[
            "检测 funding > +0.05 或 < -0.05",
            "确认 OI 同步异常（>30% 单日变动）",
            "在 1-3 个 cycle 内反向开仓",
        ],
        body="<完整 SKILL.md body>",
    ),
    ...
]
```

### 在 PromptBuilder 中的渲染

PromptBuilder 把 `list[Skill]` 渲染为 markdown bullet list：

```markdown
- **funding-rate-extreme-fade**: 资金费率极端时反向交易
  - 检测 funding > +0.05 或 < -0.05
  - 确认 OI 同步异常（>30% 单日变动）
  - 在 1-3 个 cycle 内反向开仓
- **macro-fed-meeting-volatility**: 美联储议息会议前后波动率扩大
  - ...
```

空列表时显示占位 "暂无可用技能"。

### 失败模式

| 触发 | 行为 |
|---|---|
| `agent_skills/` 目录不存在 | 返回 `[]` |
| 单个 SKILL.md 解析失败 | 跳过 + warning log，不抛异常 |
| frontmatter 缺少必填字段 | 跳过该 skill + warning log |

### SKILL.md 文件协议（沿用 spec 014）

```markdown
---
skill_id: funding-rate-extreme-fade
description: 资金费率极端时反向交易
tags:
  - macro
  - funding-rate
steps:
  - 检测 funding > +0.05 或 < -0.05
  - 确认 OI 同步异常（>30% 单日变动）
  - 在 1-3 个 cycle 内反向开仓
---

## 完整描述

<本段为 body，spec 018 用于细粒度匹配>
```

## 与 spec 018 的协议契约

spec 018 将提供进化版 `EvolvingSkillProvider`，引入 IDF / Hermes match-score 等 ranking 算法：

```python
# spec 018 示意（不在本 spec 实现范围）
class EvolvingSkillProvider:
    def get_available_skills(self, agent_id, snapshot, k=5) -> list[Skill]:
        # 1. agent_id tag match（基础过滤）
        # 2. snapshot 字段 → IDF 关键词匹配（提升 specificity）
        # 3. Hermes ReflectiveMutation 已学到的 skill→context match-score
        # 4. PnL maturity FSM 过滤未成熟 skill
        # 5. 返回 top-k
        ...
```

本 spec 落地的 Protocol 不约束 spec 018 的 ranking 算法；只约束输入 / 输出类型。
