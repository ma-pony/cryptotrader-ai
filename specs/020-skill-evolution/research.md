# Phase 0：研究与决策

**关联 spec**：[spec.md](spec.md)
**关联前置研究**：[spec 016](../016-research-skill-evolution-prior-art/) / [spec 017a](../017-agent-prompt-externalization/) / [spec 017b](../018-agent-prompt-builder-integration/) / [spec 018](../019-memory-evolution/)
**关联 brainstorm**：[brainstorm/05-spec-019-skill-evolution.md](../../brainstorm/05-spec-019-skill-evolution.md)
**Date**: 2026-05-09

## 概述

本 spec 的 7 项关键设计决策已在 brainstorm 阶段（2026-05-09）完成。本文档不重复决策推导，仅记录最终决定 + 4 项 spot-check 修订 + 实施细节研究。

## Technical Context 中无 NEEDS CLARIFICATION 项

Brainstorm 7 项决策 + 4 项 spot-check 已消除全部 ambiguity。

## 7 项关键决策（来自 brainstorm）

| # | 决策 | 来源 |
|---|---|---|
| Q1 schema 字段 | B 子集采纳（6 字段）— 不加 maturity FSM / pnl_track 等 pattern-only | spec 016 D-DS-01 重审 |
| Q2 检索算法 | B D-RT-01 子集（无 embedding，含 IDF） | "不引入新依赖"约束 |
| Q3 load_skill_tool | B 改造（走 Provider） | D-MW-01 元数据一致性 |
| Q4 触发器 | C 仅 retrieval（高级进化推迟 spec 020 daemon） | trilogy 边界 |
| Q5 数据迁移 | B + Claude 直接做 LLM 工作（5 skill 硬编码 mapping） | 用户偏好"完整"路径 |
| Q6 skill_proposal 改造 | A 完整改造（写 .draft 时跑 LLM） | schema 一致性 |
| Q7 前端 | A 扩展现有 /memory 加 SkillsGrid | spec 018 一致性 |

## 4 项 spot-check 结果（2026-05-09）

| # | 检查项 | 结果与修订 |
|---|---|---|
| 1 | discover_skills_for_agent 签名 | ✓ `(agent_id, skill_dir=None) -> list[Skill]` 含 scope filter，可 reuse |
| 2 | load_skill_tool 实际是 factory 模式 | ❌ `_make_load_skill_tool(skill_dir=None)` factory；spec 修订 FR-W13 改造方式 |
| 3 | skill_proposal 实际函数名 | ❌ `propose_new_skill`（不是 `propose_skill`）；输出到 `.draft` 不直接创 SKILL.md；spec 修订 FR-W16 写 .draft |
| 4 | MemoryPage layout | ❌ 实际 1+2col+1 = 3 sections（含 grid layout）；spec 修订 FR-W23 在末尾加 SkillsGrid 单行 section（变 4 sections） |

## 实施细节决策

### Decision 1：5 skill 硬编码 mapping（已固化）

**Decision**：FR-W3 完整列出基于 brainstorm Q5 阶段 LLM 分析得出的 mapping。包含：
- chain-analysis / macro-analysis / news-analysis / tech-analysis 4 个 agent-specific skill
- trading-knowledge 1 个 shared skill

每个 skill 含：regime_tags（统一为 []，视为 match all）+ 11-14 个 triggers_keywords（基于 SKILL.md description + body 内容）+ importance + confidence。

**Rationale**：
- 5 skill 是手写知识库，全部为 agent role 通用知识，regime_tags=[] 合理
- triggers_keywords 完整覆盖 SKILL.md 内容关键术语
- importance/confidence 为 0.6-0.8（手编辑高质量内容）

### Decision 2：Maturity 不加到 Skill

**Decision**：spec 014 既有 `Maturity = Literal["observed", "probationary", "active", "deprecated", "archived"]`（spec 018 加 archived）只用于 PatternRecord，**不**复用到 Skill。

**Rationale**：
- 5 skill 是手写知识，无 PnL 进化路径
- maturity FSM 信号（pnl_track / fundamental_failure_streak）对 Skill 不适用
- 保持 Skill / Pattern 架构分离

### Decision 3：IDF 算法实现

**Decision**：FR-W8 第二层算法使用 pure Python IDF：

```python
def compute_idf(corpus_keywords: list[list[str]]) -> dict[str, float]:
    """corpus_keywords: 每个 skill 的 triggers_keywords list."""
    n_docs = len(corpus_keywords)
    if n_docs == 0:
        return {}
    df = defaultdict(int)
    for skill_kw in corpus_keywords:
        unique = set(kw.lower() for kw in skill_kw)
        for kw in unique:
            df[kw] += 1
    return {kw: math.log(n_docs / count) for kw, count in df.items()}
```

调用时：
- `query_keywords` = snapshot 字段名 + dict value 转字符串后小写化的关键词集合
- `score_keywords(skill, query, idf_table)` = `sum(idf_table.get(kw, 0) for kw in skill.triggers_keywords if kw.lower() in query_keywords)`

**Rationale**：
- 5 skill 集 IDF 数学：`math.log(5/1) = 1.61`，`math.log(5/5) = 0`
- 在小语料集上区分度低但不为零；为 spec 020 daemon 触发 skill 集增长（≥20）后预留算法基础
- pure Python 无 sklearn / nltk 依赖

### Decision 4：skill_metadata_inference LLM prompt 模板

**Decision**：FR-W17 LLM prompt 结构：

```
SYSTEM: 你是 crypto trading skill metadata 推断专家。基于以下新 skill 内容，输出 JSON 元数据。

USER:
[New skill name: <name>]
[Description: <description>]
[Body summary: <body 前 500 字符 + 最后 200 字符>]

[Spec 014 既有 regime taxonomy:]
- high_funding (funding_rate > 0.03%)
- negative_funding (funding_rate < -0.01%)
- high_vol / low_vol
- trending_up / trending_down
- extreme_fear (Fear & Greed ≤ 25)
- extreme_greed (Fear & Greed ≥ 75)

[现有 5 skill 的 mapping 示例 — 含 regime_tags / triggers_keywords / importance / confidence]

输出 JSON：
{
  "regime_tags": [...],     // 子集 of regime taxonomy；通用 skill 用 []
  "triggers_keywords": [...], // 5-15 个关键词
  "importance": 0.0-1.0,
  "confidence": 0.0-1.0
}
```

**Rationale**：
- 单次 LLM 调用 ~500 token；frequency 1/day → 月成本 ~$0.005
- 现有 5 skill 作为 examples 提升 LLM 推断质量
- JSON 输出结构与 5 skill mapping 一致

### Decision 5：load_skill_tool factory 注入 Provider

**Decision**：FR-W13 改造方式：

```python
# src/cryptotrader/agents/skills/tool.py
def _make_load_skill_tool(provider=None, skill_dir=None):
    @tool
    def load_skill_tool(name: str) -> str:
        if provider is not None:
            skill = provider.get_skill_by_name(name)
            if skill is None:
                return f"Error: skill not found: {name}"
            return skill.body
        # 兜底：spec 014 直接读文件路径（仅当 provider 未注入时）
        result = load_skill(name, skill_dir=skill_dir)
        if "error" in result:
            return f"Error: {result['error']} (skill: {name})"
        return result.get("body", "")
    return load_skill_tool

# Module-level instance — spec 014 既有，本 spec 在 nodes/agents.py wire 时重新创建
load_skill_tool = _make_load_skill_tool()  # provider=None 兜底
```

`nodes/agents.py` wire：

```python
# 替换 module-level load_skill_tool 实例
import cryptotrader.agents.skills.tool as _skill_tool_mod
_skill_tool_mod.load_skill_tool = _skill_tool_mod._make_load_skill_tool(provider=_skill_provider)
```

**Rationale**：
- factory 接受 provider 参数，向后兼容（provider=None 时走 spec 014 路径作 fallback）
- nodes/agents.py 在 init 时替换 module-level instance；ToolAgent 实例化时拿到的是注入版本
- 不需要改 ToolAgent.tools 注入路径

### Decision 6：propose_new_skill 写 .draft 含 LLM metadata

**Decision**：FR-W16 改造方式：

```python
# src/cryptotrader/learning/skill_proposal.py
def propose_new_skill(...):
    # spec 014 既有逻辑：分析 active patterns → 输出 SKILL.md.draft
    draft_content = _build_draft_content(...)

    # 本 spec 新增：调 LLM 推断 metadata
    from cryptotrader.learning.evolution.skill_metadata_inference import infer_skill_metadata
    metadata = infer_skill_metadata(name=proposed_name, description=...desc, body=draft_content)

    # 把 metadata 写入 frontmatter
    draft_with_metadata = _add_metadata_to_frontmatter(draft_content, metadata)

    # 写 .draft 文件（spec 014 既有路径）
    draft_path.write_text(draft_with_metadata)
    return draft_path
```

**Rationale**：
- LLM 推断在 draft 阶段（不是 save 时）
- 用户 manual save 后 .draft 变 SKILL.md，metadata 已就位
- 失败兜底：metadata = 默认值（regime_tags=[] / triggers_keywords=[] / importance=0.5 / confidence=0.5）

## Phase 0 检查项

- [x] 所有 NEEDS CLARIFICATION 已解决
- [x] 所有 dependency 已识别 best practice
- [x] 所有 integration 已找到 pattern

Phase 0 输出完成。
