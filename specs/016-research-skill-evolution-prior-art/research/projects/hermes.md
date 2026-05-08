---
name: Hermes Agent Self-Evolution
url: https://github.com/NousResearch/hermes-agent-self-evolution
license: MIT
tier: 1
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: false
---

# Hermes Agent Self-Evolution — NousResearch

## 架构概览

NousResearch 的 Hermes Agent Self-Evolution 是一个独立优化框架，通过 DSPy + GEPA（遗传-帕累托提示进化）
对 Hermes Agent 的技能文件、工具描述、系统提示和代码进行自动演化。无需 GPU 训练，全程通过 API 调用完成，
每次优化运行成本约 $2–10。

**仓库统计**：2.9k stars，314 forks，MIT 许可，Python 100%。

**目录结构（关键部分）**：

```
evolution/
  core/          # config.py, constraints.py, dataset_builder.py, fitness.py, external_importers.py
  prompts/       # 提示相关组件
  skills/        # 技能定义
  tools/         # 工具集成
  code/          # 代码级演化（Phase 4）
  monitor/       # 监控
generate_report.py
PLAN.md
```

智能体本体在外部 Hermes Agent 安装中（通过环境变量指向），本仓库为独立优化流水线，
通过 `batch_runner.py` + `trajectory.py` 与智能体交互。

---

## 提示组装（Phase 1 重点）

### 系统提示构造（`agent/prompt_builder.py`）

系统提示由两类组件拼接而成：

**可演化组件**（DSPy 参数化，供 GEPA 优化）：

| 组件名 | 用途 |
|---|---|
| `DEFAULT_AGENT_IDENTITY` | 人格与行为特征描述 |
| `MEMORY_GUIDANCE` | 持久记忆使用模式说明 |
| `SESSION_SEARCH_GUIDANCE` | 历史上下文检索触发条件 |
| `SKILLS_GUIDANCE` | 技能加载与缓存逻辑 |
| `PLATFORM_HINTS` | 平台特定格式规则 |

**固定组件**（不参与演化）：

- 自动生成的技能索引（skill index）
- 来自记忆存储的用户记忆块（user memory blocks）
- 项目上下文文件（project context files）

优化器将可演化部分包装为 DSPy Signature 参数，在保持 schema 结构不变的前提下精炼指令文本。

### 技能注入机制

技能以 `SKILL.md` 文件形式存储在 `skills/` 目录。智能体初始化时的注入流程：

1. 技能文件在初始化时作为用户消息加载进上下文窗口
2. GEPA 将技能文本视为可变异字符串参数进行演化
3. 演化后的变体部署为新版本——**从不在进行中的会话内热替换**
4. 所有更改在下一次新鲜会话（fresh session）时生效

**尺寸约束**：技能文件默认 ≤15KB；提示段落 ≤当前长度的 120%。

**缓存合规性**：Schema 结构（参数名称、类型）保持冻结，只有描述文本参与演化，
以保证对话缓存（conversation caching）兼容性。

---

## 记忆 ↔ 技能连接（Phase 1 精简版）

### SessionDB 双角色（`hermes_state.py`）

`hermes_state.py` 中的 `SessionDB` 对象同时服务于优化流水线和技能上下文：

**用于优化时**：
- 从真实使用模式中挖掘数据，构建评估数据集
- 用户纠错（corrections）标记为"误选信号"（misselection signals）
- 失败会话分析作为失败案例输入演化

**用于技能上下文时**：
- 持久记忆通过 `SESSION_SEARCH_GUIDANCE` 中定义的触发条件被检索
- 记忆质量直接影响技能有效性
- 演化同时改进记忆检索逻辑和技能文本本身

### 记忆–技能耦合路径

```
用户会话
  → trajectory.py（捕获执行轨迹）
  → SessionDB（存储记忆 + 失败案例）
      ├── [优化路径] dataset_builder.py → GEPA 演化技能/提示
      └── [运行路径] SESSION_SEARCH_GUIDANCE 触发 → 检索相关记忆注入上下文
```

记忆检索逻辑本身是一个可演化的提示段（`SESSION_SEARCH_GUIDANCE`），
这意味着系统可以演化"何时检索记忆"的判断逻辑，而不仅仅是"检索什么"。

### 工具描述演化中的记忆影响

工具 Schema 存于 `tools/registry.py`，描述字段随每次 API 调用发送。
优化器：
- 将工具描述作为 DSPy Signature 字段处理
- 评估"工具选择准确性"（tool selection accuracy）
- 使用跨工具适应度函数，防止某一描述"抢占"其他工具的选择份额
- 强制 ≤500 字符/工具描述（"每个额外字符都会在整个对话中乘积累加"）

---

## Phase 2 占位符

以下内容留待 Phase 2 研究（演化算法深挖）时填充：

- 演化算法（GEPA 详细机制 / DSPy 集成方式）
- 技能数据结构（`SKILL.md` 完整 schema）
- 检索机制（相似度计算 / 向量化方式）
- 评估流水线（fitness 函数详细实现）
- 智能体 ↔ 技能边界（接口合约）
- 工程细节（批量运行、并行评估、Git 谱系追踪）

---

## 借鉴建议（仅 Phase 1 范围）

以下建议基于当前研究范围，针对 cryptotrader-ai spec 016：

### 高优先级借鉴

1. **分段系统提示（Sectioned System Prompt）**
   将系统提示拆分为具名段落（`MEMORY_GUIDANCE`、`SKILLS_GUIDANCE` 等），
   每段独立可测试、可演化。cryptotrader-ai 当前的系统提示可参照此模式重构，
   使各段具备独立的字符预算和语义约束。

2. **技能–会话隔离原则**
   "不在进行中的会话内热替换技能"的原则适用于 cryptotrader-ai：
   演化后的经验规则/技能提示应在下一个交易周期生效，
   不应在当前决策链路中途注入。

3. **记忆检索触发逻辑可演化化**
   `SESSION_SEARCH_GUIDANCE` 的设计思路极具参考价值：
   把"何时检索历史经验"本身作为一个可优化的提示参数，
   而非硬编码的规则。对应 cryptotrader-ai 的 `verbal.py` / `context.py`。

4. **误选信号作为评估数据**
   用户/回测纠错（actual outcome vs. predicted）标记为误选信号，
   自动构建演化数据集的思路，与 cryptotrader-ai 的经验记忆（`ExperienceRule`）高度兼容。

### 需要适配的差异

- Hermes 以 `SKILL.md` 文本文件存储技能，cryptotrader-ai 使用结构化 `ExperienceRule` 对象；
  需要设计从结构化数据到提示文本的序列化层。
- 工具描述演化（Phase 2 of Hermes）在 cryptotrader-ai 中对应 Agent 工具调用描述的优化，
  可作为后续扩展方向。
- Hermes 的 15KB 技能文件上限对应 cryptotrader-ai 的 token 预算约束（已有 `_estimate_tokens()`）。

---

## 说明 / 开放问题

1. **`evolution/core/fitness.py` 未能获取源码**：适应度函数的具体实现（是否使用 LLM-as-judge、
   如何量化"更好"）未知，留待 Phase 2 深挖。

2. **`evolution/prompts/` 目录内容未知**：该目录仅发现 `__init__.py`，
   可能包含 GEPA 的变异/交叉提示模板，Phase 2 应重点检查。

3. **Hermes Agent 本体代码未检查**：`agent/prompt_builder.py` 和 `hermes_state.py`
   位于外部 Hermes Agent 仓库（非本仓库），本次研究未覆盖。

4. **Phase 4（代码演化）使用 Darwinian Evolver（AGPL v3）**：
   如需引入代码演化能力，需注意许可证兼容性问题。

5. **`trajectory.py` 轨迹格式未知**：执行轨迹的具体 schema 影响
   cryptotrader-ai 是否能复用类似机制，需 Phase 2 确认。
