---
name: EvoSkill
url: https://github.com/sentient-agi/EvoSkill
license: Apache-2.0
tier: 2
last_accessed: 2026-05-07
phase_1_complete: true
phase_2_complete: false
---

# EvoSkill — sentient-agi

## Architecture Overview

EvoSkill 是一个自动化框架，用于从编码代理的失败尝试中发现并创建可复用的 agent 技能（skills），以提升跨平台编码代理的性能。支持的 agent 包括 Claude Code、OpenCode、OpenHands、Goose 和 Codex CLI。

核心五阶段迭代循环：

```
Base Agent → Proposer → Generator → Evaluator → Frontier Update
     ↑                                                   |
     └───────────────────────────────────────────────────┘
```

1. **Base Agent**：当前最优程序尝试基准测试问题集
2. **Proposer**：分析失败样本，提出技能或提示词的修改方案
3. **Generator**：根据提案写入新技能文件或重写系统提示词
4. **Evaluator**：在验证集上对新变体打分（pairwise 比较）
5. **Frontier**：维护 Top-N 最优程序版本，跨迭代保留

主要语言：Python 73.8%，Jupyter Notebook 25.9%。版本 v1.1.0（2026-05 发布），697 stars。

---

## Prompt Assembly（Phase 1）

### 提案提示词构建（`build_proposer_query()`）

`src/loop/runner.py` 中的 `build_proposer_query()` 将以下上下文注入 Proposer 的 LLM 调用：

| 注入内容 | 来源 |
|---|---|
| 失败样本（agent 输出 + 真实答案） | 训练集运行结果，score < 0.8 的样本 |
| 反馈历史（历次提案及结果） | `.evoskill/feedback_history.md` |
| 当前活跃技能列表 | `.claude/skills/*/SKILL.md` 文件名枚举 |
| 任务约束 | 初始化时传入的领域限制 |
| 演化模式路由 | `mode: "skill_only"` / `"prompt_only"` |
| 截断级别（0/1/2） | 若 Proposer 失败则逐级压缩上下文 |

**关键设计**：失败上下文 + 历史反馈 + 现有技能三者共同构成提案上下文，Proposer 做出 `prompt vs skill` 的二元路由决策。

### 当前最优程序（"program"）的表示

存储于 `.evoskill/config.toml` 中的 `program.yaml`，包含：
- 系统提示词（system prompt）
- 工具规范（tool specifications）

程序版本通过 git 分支管理：`program/iter-skill-N`，Top-N 变体打 `frontier/*` 标签。

### 提案输出 Schema

```python
# src/schemas/proposer.py — ProposerResponse
optimize_prompt_or_skill: Literal["prompt", "skill"]
proposed_skill_or_prompt: str   # 实际内容（技能描述或提示词文本）
justification: str               # 修改理由

# src/schemas/skill_proposer.py — SkillProposerResponse
action: Literal["create", "edit"]  # 默认 "create"
target_skill: str | None           # edit 时必填：目标技能名
proposed_skill: str                # 技能的高层描述
justification: str                 # 为何此技能能填补能力缺口
related_iterations: list[str]      # 引用的历史迭代 ID
```

### Feedback Descent（`src/feedback_descent.py`）

核心优化抽象，解耦 Proposer 与 Evaluator：

```python
# Proposer Protocol
generate_initial(problem) -> artifact
propose(current_best, feedback_history) -> candidate

# Evaluator Protocol
evaluate(current_best, candidate) -> EvaluationResult(preferred: bool, rationale: str)
```

终止条件：连续 3 次未改进，或达到最大迭代数（默认 10）。每次迭代无论是否改进都记录反馈；改进成功时清空反馈历史（避免历史污染新起点）。

---

## Memory ↔ Skill（Phase 1 lite）

### 持久化记忆层

| 文件 | 内容 | 作用 |
|---|---|---|
| `.evoskill/feedback_history.md` | 每次提案 + 理由 + 结果 + 得分 + 当时活跃技能 | 跨迭代告知 Proposer 历史决策轨迹 |
| `.evoskill/loop_checkpoint.json` | 采样状态（各类别轮询偏移） | 支持 `--continue` 精确断点续跑 |
| `program/iter-skill-N` git 分支 | 完整程序快照（提示词 + 技能） | 可回滚、可比较任意历史版本 |
| `frontier/*` git 标签 | Top-N 最优程序 | 父程序选择策略的候选池 |

### 技能文件结构

技能存储于 `.claude/skills/<skill-name>/SKILL.md`，为可复用的指令包，包含 frontmatter 元数据。写入后通过 `skill_frontmatter_normalization` 适配各 agent 平台（OpenCode / Goose / Codex CLI 各有差异）。

### 关键记忆设计决策

- **改进后清空反馈历史**：防止旧失败模式污染新起点的提案上下文
- **active skills 注入提案**：Proposer 可感知当前已有技能，避免重复创建
- **`related_iterations` 字段**：显式关联历史迭代，支持跨轮归因

---

## Phase 2 Placeholders

（留空，待 Phase 2 深度阅读）

---

## Borrow Recommendations（Phase 1 only）

以下模式值得在 cryptotrader-ai spec 016（技能演化）中借鉴：

1. **`ProposerResponse` 二元路由**：`prompt vs skill` 的显式分类让下游生成器路由清晰，可类比为 cryptotrader-ai 中 `strategy_rule vs agent_prompt` 的路由。

2. **`feedback_history.md` 叙事式记忆**：以自然语言 Markdown 记录历次提案及结果，而非结构化 DB——对 LLM 友好，Proposer 可直接读取作为 few-shot 上下文。对应 cryptotrader-ai 的 `learning/reflect.py` 经验记忆写入机制有参考价值。

3. **改进后清空反馈历史**：避免跨阶段污染。当前 `ExperienceMemory` 合并策略可考虑类似的"阶段隔断"设计。

4. **`related_iterations` 显式回溯**：技能提案显式引用触发灵感的历史迭代 ID，比隐式 embedding 检索更可解释，适合需要审计轨迹的交易场景。

5. **截断层级（0/1/2）**：当上下文超长时逐级压缩（全量 → 中等 → 激进截断），可直接用于 cryptotrader-ai 的 GSSC 管道中的 token 预算控制。

6. **git 分支作为程序版本控制**：每次迭代快照存为独立 branch，frontier 打 tag——这是一种"无额外存储开销"的版本化方案，比自定义 DB 更易于调试和回滚。

---

## Notes / Open Questions

- **技能粒度**：EvoSkill 的技能是 agent 级别的操作指令（如"如何调用某工具"），而 cryptotrader-ai 的 spec 016 技能可能更偏向交易策略规则——两者的粒度差异需在 spec 设计阶段明确。
- **评估器设计**：EvoSkill 用 pairwise LLM 比较作为评估器，cryptotrader-ai 有真实回测收益可用——后者的评估信号更强，但延迟也更高，需要权衡。
- **跨 agent 迁移**：EvoSkill 强调技能跨 agent 可迁移性，cryptotrader-ai 是单体系统，此特性不直接适用，但其 frontmatter 标准化思路可用于多模型适配。
- **Phase 2 待深读**：`src/loop/helpers.py`（采样逻辑）、`src/evaluation/`（评分器实现细节）、`examples/officeqa/`（端到端示例）。
