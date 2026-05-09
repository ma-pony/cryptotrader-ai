# Contract：EvolvingSkillProvider API

**模块路径**：`src/cryptotrader/learning/evolution/skill_provider.py`（NEW）

## 实现 Protocol

`EvolvingSkillProvider` 实现 spec 017a 的 `SkillProvider` Protocol：

```python
class EvolvingSkillProvider:
    def get_available_skills(
        self,
        agent_id: str,
        snapshot: dict,
        k: int = 5,
    ) -> list[Skill]: ...
```

## 公共方法

### `__init__`

```python
def __init__(
    self,
    skill_root: Path = Path("agent_skills"),
    top_k: int = 5,
) -> None: ...
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `skill_root` | `Path("agent_skills")` | spec 014 既有路径 |
| `top_k` | 5 | get_available_skills 默认 top-k |

### `get_available_skills` （实现 SkillProvider Protocol）

```python
def get_available_skills(
    self,
    agent_id: str,
    snapshot: dict,
    k: int = 5,
) -> list[Skill]: ...
```

**行为约定**（FR-W8）：

1. **第一层 — scope + regime_tags 预过滤**：
   - 调 `discover_skills_for_agent(agent_id, skill_dir=self.skill_root)`（spec 014 既有）拿 scope-match skill 候选集
   - 对每个候选 skill：若 `skill.regime_tags == []` 则纳入；否则 `current_regime in skill.regime_tags` 才纳入
   - `current_regime` 从 snapshot 提取（如 snapshot.market.funding_rate > 0.0003 推得 "high_funding"，等等）

2. **第二层 — IDF + 元数据加权**：
   - 计算 IDF 表：`idf_table = compute_idf([s.triggers_keywords for s in candidates])`
   - 对每个 candidate 算 score：
     ```python
     query_keywords = extract_query_keywords(snapshot)  # 从 snapshot 字段名 + 关键值
     idf_score = score_skill(skill.triggers_keywords, query_keywords, idf_table)
     recency_bonus = math.exp(-(now - skill.last_accessed_at).total_seconds() / (7 * 86400))
     score = (idf_score + skill.importance + recency_bonus) × skill.confidence
     ```
   - 排序倒序，取 top-k

3. **回写 access_count / last_accessed_at**：
   - 对返回的 top-k skill 写文件 frontmatter（`access_count += 1` + `last_accessed_at = now()`）

4. **写 telemetry**（FR-W28）：
   - `skill.retrieval.candidates_after_regime_filter` (list[str])
   - `skill.retrieval.top_k_with_scores` (list[dict])
   - `skill.retrieval.filtered_out` (list[dict])
   - `skill.retrieval.duration_ms` (float)

**容错**（FR-W9）：内部任一步骤异常 → catch + log warning + return `[]`。**不抛异常**。

### `get_skill_by_name`

```python
def get_skill_by_name(self, name: str) -> Skill | None: ...
```

**行为约定**（FR-W10）：

1. 扫 `<skill_root>/*/SKILL.md` 找 frontmatter `name == 给定 name` 的 skill
2. 找到 → access_count += 1 + last_accessed_at = now() 写回文件 → 返回 Skill
3. 找不到 → 返回 None
4. IO 异常 → catch + log warning + 返回 None

**用途**：load_skill_tool 改造后调此方法（FR-W13）。

---

## 集成点

### 调用方 1：PromptBuilder（spec 017b/18 既有路径）

```python
# nodes/agents.py:_get_or_build_pb
from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

if _skill_provider is None:
    _skill_provider = EvolvingSkillProvider(skill_root=Path("agent_skills"))
```

替换 spec 017a/b 的 `DefaultSkillProvider`；spec 017a class 整体删除（FR-W11）。

### 调用方 2：load_skill_tool factory（本 spec 改造）

```python
# nodes/agents.py 在 init 时
import cryptotrader.agents.skills.tool as _t
_t.load_skill_tool = _t._make_load_skill_tool(provider=_skill_provider)
```

`_make_load_skill_tool(provider)` 内部 tool 调 `provider.get_skill_by_name(name)`。

---

## Schema 升级路径

spec 020 若需替换为 `OptimizedSkillProvider`（含 cache + daemon）：直接替换 `_skill_provider` 实例即可。本 spec 接口稳定。

## 单测要求

参考 spec.md SC-W4：`tests/test_evolving_skill_provider.py` ≥ 12 用例 PASS：
- (a) 加载 skill 走 D-RT-01 两层算法
- (b) regime_tags=[] 视为 match all
- (c) regime_tags 不匹配的 skill 被过滤
- (d) IDF 计算正确（fixture 5 skill + snapshot 关键词）
- (e) recency_bonus 计算正确
- (f) score = (idf + importance + recency) × confidence
- (g) top-k 默认 5
- (h) access_count 在 retrieval 后回写文件
- (i) IDF 异常时返回空 list + warning log
- (j) IO 异常时返回空 list + warning log
- (k) 空目录返回空 list
- (l) Provider 实现 spec 017a `SkillProvider` Protocol（鸭子类型）
