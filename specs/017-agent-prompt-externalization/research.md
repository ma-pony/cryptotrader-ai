# Phase 0：研究与决策

**关联 spec**：[spec.md](spec.md)
**关联前置研究**：[specs/016-research-skill-evolution-prior-art/research/decisions.md](../016-research-skill-evolution-prior-art/research/decisions.md)
**Date**: 2026-05-08

## 概述

本 plan 涉及的关键技术决策已经在 spec 016 完成系统性研究（8 项目 × 8 角度）。本文档不重复研究内容，仅引用已落地的决策、补充本 spec 落地特有的实现细节决策（YAML 解析、错误处理、Token 估算精度验证、telemetry 集成路径）。

## Technical Context 中无 NEEDS CLARIFICATION 项

Technical Context 已无未决项：
- Language / Dependencies / Storage / Testing / Platform / Performance / Constraints / Scale 全部由项目现状或 spec 决定
- 无新依赖引入需求

## 决策记录

### Decision 1：Markdown frontmatter 作为 agent 配置载体

**Decision**：`config/agents/<agent_id>.md` = YAML frontmatter（必填字段）+ Markdown body（含 `## section_name` 段落）。

**Rationale**：
- spec 016 D-PA-01 已采纳此方案（Hermes / autoresearch / Claude Code skills 多个项目验证）
- 单文件 = config + prompt，避免双文件失同步
- YAML frontmatter 可直接 `yaml.safe_load()`，body 用 markdown header 切段
- 编辑友好：直接 vim / IDE 改 prompt，无需 JSON 转义

**Alternatives considered**：
- 纯 YAML（拒绝：长 prompt 多行字符串可读性差，易出缩进 bug）
- 纯 JSON（拒绝：长字符串需 escape `\n`，不可维护）
- TOML + 单独 prompt 文件（拒绝：双文件易失同步）

### Decision 2：YAML 解析使用 PyYAML 已有依赖

**Decision**：用项目已有的 `PyYAML`（通过 `yaml.safe_load`）解析 frontmatter。

**Rationale**：
- 已是项目依赖（`pyproject.toml` 已含）
- `safe_load` 防 YAML 注入
- frontmatter 切分用正则 `^---\n(.*?)\n---\n(.*)`（DOTALL）

**Alternatives considered**：
- `python-frontmatter` 第三方库（拒绝：增加新依赖，FR-X 明确不引入）
- 手写 YAML 子集解析器（拒绝：YAGNI 复杂化）

### Decision 3：Token 估算继续使用 spec 014 `_estimate_tokens()`

**Decision**：复用 spec 014 已存在的 `_estimate_tokens()`（CJK÷1.5 + ASCII÷4）。

**Rationale**：
- spec 014 已落地、生产验证误差 < 10% vs tiktoken（中英混合 prompt）
- 不引入 tiktoken（避免下载模型权重 / 启动期网络 fetch）
- FR-X13 明确要求复用

**Alternatives considered**：
- tiktoken（拒绝：新依赖，启动期需要 BPE 文件，cold-start 慢）
- HuggingFace tokenizer（拒绝：更重，且只对特定模型准）

**Implementation 备忘**：
- spec 014 函数定义于 `src/cryptotrader/learning/context.py:_estimate_tokens()`（按既有路径引用）
- 若该函数后续移动，PromptBuilder 需同步更新 import；可在 `prompt_builder.py` 顶层 `from cryptotrader.learning.context import _estimate_tokens` 集中处理

### Decision 4：Slot 分配策略

**Decision**：默认 slot 分配 = `system_prompt` / `available_skills` / `output_schema` 入 SystemMessage；`recent_memory` / snapshot / portfolio / agent_analyses / `user_tail` 入 UserMessage。`slot_overrides` 可在 frontmatter 覆盖。

**Rationale**：
- spec 016 D-PA-02 + D-PA-05 决策：长稳定 prefix 入 system（利于 Anthropic prompt cache 命中），动态短内容入 user-tail
- system 含静态约束（角色 + skill 列表 + JSON schema），user 含每 cycle 变化的 snapshot/memory
- D-MW-02：recent_memory 默认入 user_tail（每 cycle 不同），但可 override 入 system 测试 prompt cache 影响

**Alternatives considered**：
- 全部入 system（拒绝：每 cycle snapshot 变会破坏 cache）
- 全部入 user（拒绝：浪费 system slot，cache 命中率为 0）
- 三段式 system/user/assistant（拒绝：assistant 段易混淆 LLM）

### Decision 5：Token Budget Enforcer 优先级机制

**Decision**：用户在 frontmatter `priority: dict[str, int]` 显式声明每个 section 的优先级（数字越小越保留）。`system_prompt` 与 `output_schema` 强制保留（不可丢、不可降）。

**Rationale**：
- 显式 > 约定：避免运行时猜测重要性
- system_prompt = 角色定义不可丢；output_schema = JSON contract 不可丢（丢了 LLM 输出无法 parse）
- 可丢顺序：snapshot 字段 > recent_memory > available_skills（业务直觉：最新数据 > 历史记忆 > 技能提示）
- 截断方式：recent_memory / available_skills 各保留前 N 条（按 Provider 内部排名）

**Alternatives considered**：
- 全自动按 token 占比丢（拒绝：可能丢掉关键 system_prompt）
- 仅截断不丢段（拒绝：长 prompt 截断后语义破碎，LLM 易困惑）

### Decision 6：Provider 协议用 typing.Protocol

**Decision**：`MemoryProvider` / `SkillProvider` 用 `typing.Protocol` 定义结构子类型，不强制继承基类。

**Rationale**：
- spec 018 可在不修改 spec 017 代码的情况下注入新 Provider 实现
- Protocol 在 mypy 检查支持鸭子类型，比 ABC 灵活
- DefaultMemoryProvider / DefaultSkillProvider 不强制继承，仅满足 Protocol 即可

**Alternatives considered**：
- abc.ABC + abstractmethod（拒绝：spec 018 必须继承基类，耦合更紧）
- 无协议直接 callable（拒绝：失去类型检查）

### Decision 7：ConfigValidationError fail-fast 时机

**Decision**：`ConfigLoader` 在 PromptBuilder 实例化时（构造函数内）一次性校验全部 4 个 config 文件；任一失败抛 `ConfigValidationError`，进程通过 `nodes/agents.py` 启动期实例化时崩。

**Rationale**：
- fail-fast 优于运行时 cycle 中崩（cycle 中崩可能丢部分 trade decision）
- 启动期 1 次 IO 解析全部 config，性能影响可忽略
- `ConfigValidationError` 含 file_path + 失败原因，便于运维定位

**Alternatives considered**：
- 懒加载（拒绝：cycle 中第一次调用才 fail，不可接受）
- 每 cycle 重新加载（拒绝：浪费 IO，且不支持热更新需求超出 OOS）

### Decision 8：Telemetry 注入路径

**Decision**：`PromptBuilder.build()` 在 OpenTelemetry 当前 active span 上 `set_attribute()` 写入 8 个字段；如果当前无 active span（非 LangGraph 上下文调用），降级写入 structured log。

**Rationale**：
- spec 010 已建立 tracing 基础设施（每个 cycle 有 root span）
- 子节点调用 `tracer.get_current_span()` 即获得 cycle span，挂 attribute 即可
- 无新 collector / exporter 配置需求
- structured log fallback 保证测试 / 单元调用也有可观测信号

**Alternatives considered**：
- 起新 span（拒绝：每 cycle 多 4 个 span 增加 trace 噪声）
- 只写 log 不挂 trace（拒绝：trace 上下文丢失，难关联 cycle）

### Decision 9：DefaultMemoryProvider 的 patterns + cases 拼接格式

**Decision**：DefaultMemoryProvider 输出 markdown 文本，结构 = 顶部 `### Patterns`（精炼规则 N 条）+ `### Cases`（历史案例 N 条），patterns + cases 各自独立 top-k（k 由 Provider 内部按 spec 014 已有逻辑选）。

**Rationale**：
- spec 016 D-MW-01：混合 patterns + cases 优于纯 cases（Hermes / SkillClaw / EvoSkills 共识）
- markdown 子标题让 LLM 易扫描
- 不在本 spec 引入新进化逻辑（spec 018 才动 Provider 内部排名）

**Alternatives considered**：
- 仅 patterns（拒绝：丢失具体 case 上下文，LLM 难类比）
- 仅 cases（拒绝：spec 014 已经是这样，spec 016 研究指出该方案次优）

### Decision 10：4 个 agent 文件的初始 prompt 内容来源

**Decision**：T2-T5 每个 agent 的 `config/agents/<name>.md` 中 `system_prompt` 段的初始内容**完全照搬**当前 `src/cryptotrader/agents/<name>.py` 中的 `ROLE` 字符串；`output_schema` 段照搬该 agent 当前 expect 的 JSON schema 描述；`available_skills` / `recent_memory` 段为占位（运行时由 Provider 注入）。

**Rationale**：
- spec OOS 明确："仅迁移搬运，不重写 prompt 内容"
- 保持语义等价，让 SC-X7（token 差异 < 15%）可量化
- 后续 prompt 优化是独立工作，不在本 spec 范围

**Alternatives considered**：
- 顺便重写 prompt（拒绝：违反 OOS，扩大变更面）
- 留空让用户填（拒绝：违反 SC-X1 "frontmatter 合法 + body 至少 5 个 section"）

## Phase 0 检查项

- [x] 所有 NEEDS CLARIFICATION 已解决（Technical Context 中无未决项）
- [x] 所有 dependency 已识别 best practice（PyYAML 解析、_estimate_tokens 复用、OpenTelemetry 集成）
- [x] 所有 integration 已找到 pattern（Provider Protocol、Slot 分配、Telemetry 路径）

Phase 0 输出完成，进入 Phase 1。
