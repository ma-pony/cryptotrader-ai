# Brainstorm: Spec 017b — Agent Prompt Builder Integration

**Date:** 2026-05-08
**Status:** active
**Spec:** （待 `/speckit-specify` 创建）

## Problem Framing

承接 [spec 017a](../specs/017-agent-prompt-externalization/) 的 PromptBuilder 基建（已合并 main，commit `cfd3acc` + `f1e37a9`）。017a 仅交付了基础模块 + 单测，4 个 analysis agent（tech / chain / news / macro）仍跑硬编码 ROLE 路径，没有真正接到 PromptBuilder。

本 spec 完成"集成切换"工作，包含 3 块整合：

1. **base.py** 的 `BaseAgent` / `ToolAgent` 重构 — 删 `role_description` 字段、删 `ANALYSIS_FRAMEWORK` 常量、`analyze()` 改用 `prompt_builder.build()`
2. **config.py** 的 `AgentsConfig.build()` 重构 — 删 `_resolve_role` / `_resolve_skills` / `prompt_template` 字段；接受 `prompt_builder` 注入
3. **agents/skills/middleware.py** 删除 — 由 `DefaultSkillProvider` + `PromptBuilder._render_skills()` 替代；`load_skill_tool` 保留为独立 LangChain tool

落地后 `grep -rn "^ROLE\s*=" src/cryptotrader/agents/` 返回空；4 agent 文件每个 < 150 行；`agents/skills/middleware.py` 不存在。

## 6 项关键设计决策

### Q1 — 整合力度（meta-决策）

**选项**：A 保守（薄包装）/ B 中度（替换 message 拼接层）/ C 激进（删完所有旧路径）

**决策**：**C 激进**

**理由**：
- 用户在 017a 已确认"直接删旧不留 fallback"偏好
- B 仍保留 SkillsInjectionMiddleware，spec 018 还要再清一遍 skill 注入双路径，浪费
- C 一次性删除：SkillsInjectionMiddleware / AgentRegistry.prompt_template / ANALYSIS_FRAMEWORK 常量 / role_description 字段

### Skill-Q — 删除 SkillsInjectionMiddleware 后 skill 如何加载

**选项**：A verbatim body（PromptBuilder 装载完整 body）/ B summary only / C 双段

**决策**：**A verbatim body**

**理由**：
- 行为等价于现状是最稳妥的迁移路径，spec 017b 不该顺便改"LLM 看到多少 skill 内容"语义
- spec 018 才是"按 PnL ranking + top-k 截断"的合适落地点
- 017a TokenBudgetEnforcer 的 `degraded_sections` 兜底已够用

**实施细节**：
- `DefaultSkillProvider.get_available_skills()` 改用 `discover_skills_for_agent()`（spec 014 既有），按 `scope == "shared"` 或 `scope == f"agent:{agent_id}"` 过滤（**修正 017a 的 `agent_id in tags` bug**）
- `PromptBuilder._render_skills()` 输出完整 SKILL.md body，格式 `\n\n---\n## Skill: {name}\n\n{body}` 与 SkillsInjectionMiddleware 一致
- `load_skill_tool` 保留为独立 LangChain tool，由 `nodes/agents.py` 在实例化 ToolAgent 时通过 `tools=[..., load_skill_tool]` 注入

### Q2 — ANALYSIS_FRAMEWORK 内容去哪

**选项**：A 完整复制到 4 个 config / B 拆为 discipline+JSON schema 两段 / C `_shared/` 共享 include

**决策**：**B 拆两段**

**理由**：
- output_schema 段在 017a FR-X11 强制保留（不可丢/降），把 JSON schema 放那里有架构层 protection
- C 引入新 include 机制，违反 017a 稳定契约
- ANALYSIS_FRAMEWORK 进化频率低，不值得加魔法

**实施**：
- discipline 部分（Rules / Pre-signal checklist / Confidence calibration / Data sufficiency）→ 4 个 config 的 `system_prompt` 段末尾 verbatim 复制
- JSON schema 部分（"CRITICAL: Output ONLY..." + 字段定义）→ 4 个 config 的 `output_schema` 段 verbatim 复制

### Q3 — 4 agent 现有 `_build_prompt(snapshot)` 怎么处理

**选项**：A PromptBuilder 全吸收 / B 双层（agent helper + pre_rendered_snapshot 参数）/ D 独立 renderer 模块

**决策**：**D 独立 renderer**

**理由**：
- 隔离 crypto 领域逻辑，PromptBuilder 保持 generic
- 4 agent 文件可达成 < 150 行
- spec 018 可在 PromptBuilder 加 `snapshot_renderer` 构造参数注入新 renderer，不破坏本 spec

**实施**：
- 新建 `src/cryptotrader/agents/snapshot_renderer.py`，含 `render_crypto_snapshot(snapshot: dict) -> str`
- 搬运 `BaseAgent._build_prompt()` 全部逻辑（funding annotation / news headlines / data warnings / sanitize_input / experience cap / TechAgent indicators）
- `PromptBuilder._render_snapshot()` 默认调 `render_crypto_snapshot()`
- TechAgent.compute_indicators 计算逻辑保留在 tech.py；agent.analyze() 在调 build() 前 merge indicators 进 snapshot dict

### Q4 — ToolAgent backtest_mode 怎么处理

**选项**：A 保留二分支 / B 删除 / C 上移到工厂层

**决策**：**A 保留**

**理由**：backtest_mode 语义（防 forward-looking bias）保留；改动最小；超出 spec 范围的不动

**实施**：
- `ToolAgent.__init__(*, agent_id, prompt_builder, tools, model="", backtest_mode=False)` 签名改
- backtest_mode=True → super().analyze()（走 BaseAgent / PromptBuilder 路径）
- backtest_mode=False → `prompt_builder.build()` → `create_agent(llm, tools=self.tools, system_prompt=sys_msg.content)` → `agent.ainvoke({"messages":[{"role":"user","content":usr_msg.content}]})`

### Q5 — `AgentsConfig.build()` 的 PromptBuilder 注入方式

**选项**：A 中央化（AgentsConfig 内部构造）/ B nodes/agents.py module-level singleton / C PromptBuilderFactory 新模块

**决策**：**B nodes/agents.py module-level singleton**

**理由**：
- config.py 是基础配置模块，反向依赖 agents/prompt_builder 会让模块依赖图出现潜在环
- nodes/agents.py 本来就是 runtime wiring 层
- 测试隔离性更好

**实施**：
- `nodes/agents.py` 顶层加 `_memory_provider` / `_skill_provider` / `_prompt_builders: dict[str, PromptBuilder]` singleton
- `_get_or_build_pb(agent_id, model)` helper lazy-init Provider + 缓存 PromptBuilder per agent_id
- `cfg.agents.build(...)` 调用处传 `prompt_builder=_get_or_build_pb(agent_id, model_override)`
- `AgentsConfig.build(agent_id, *, prompt_builder, backtest_mode, model_override)` 签名加必填参数；删 `regime_tags` 参数（_resolve_skills 已删）

### Q6 — 迁移顺序与回滚颗粒度

**选项**：A 单 commit 大爆炸 / B 三 commit 单 PR / C 渐进 with compat shim（违反 Q1 C）

**决策**：**B 三 commit 单 PR**

**理由**：
- C1 / C3 review 简单，C2 atomic 但语义聚焦
- 与 Q1 C 一致（无 fallback）
- revert 时优先 revert C2（最大且最关键）

**Commit 序列**：
- **C1（纯新增）**：4 个 `config/agents/<name>.md` + `snapshot_renderer.py` + `test_snapshot_renderer.py`，无 behavior 变化
- **C2（atomic 切换）**：base.py + config.py + middleware 删 + 4 agent + nodes/agents.py wiring + 4 agent test 更新
- **C3（E2E + gate）**：`test_e2e_prompt_externalization.py` + ruff per-file-ignores + grep gate

## 6 段 Spec Outline

### Section 1 — Purpose

完成 4 agent 真正切换：base.py / config.py / middleware 三处重构 + 4 agent 类重构 + 4 个 config 文件创建 + snapshot_renderer 模块。落地后 ROLE 常量退役 + agent 文件 < 150 行 + middleware 文件不存在。

### Section 2 — User Stories

- **US-Y1（Architect）— 4 agent 配置驱动 prompt（P1）**：修改 `config/agents/tech.md` 重启即生效
- **US-Y2（Architect）— Skill 加载语义零回归（P1）**：删 middleware 后 LLM 看到的 skill body 与现状等价
- **US-Y3（Maintainer）— Snapshot 渲染领域逻辑零回归（P1）**：funding annotation / news 防注入 / data warnings 全保留，物理隔离在 snapshot_renderer.py
- **US-Y4（Operator）— Backtest 路径零回归（P2）**：ToolAgent.backtest_mode 行为不变
- **US-Y5（Reviewer）— Telemetry 与 E2E 验证（P2）**：1 cycle 后 4 agent telemetry 含 8 字段

### Section 3 — Functional Requirements

39 条 FR-Y1 至 FR-Y39，分组：
- Configuration（FR-Y1~Y3）：4 个 config 文件 + ANALYSIS_FRAMEWORK 拆段
- base.py 重构（FR-Y4~Y10）：BaseAgent / ToolAgent 签名 + analyze() + 删 ANALYSIS_FRAMEWORK
- snapshot_renderer.py（FR-Y11~Y14）：新模块 + render_crypto_snapshot + PromptBuilder 接入
- 4 agent 类重构（FR-Y15~Y18）：删 ROLE / 删 _build_prompt / 构造器加 prompt_builder / 行数 gate
- config.py 重构（FR-Y19~Y24）：AgentsConfig 签名 + _resolve_* / prompt_template 删除
- SkillsInjectionMiddleware 删除（FR-Y25~Y27）
- DefaultSkillProvider 修正（FR-Y28~Y30）：scope filter + 完整 body 渲染
- nodes/agents.py wiring（FR-Y31~Y34）：module singleton + load_skill_tool 注入
- graph.py / 其他（FR-Y35~Y36）
- Telemetry / Migration（FR-Y37~Y39）

### Section 4 — Success Criteria

17 条 SC-Y1 至 SC-Y17：
- 配置 / 文件存在性 gate（SC-Y1~Y6）
- grep / 残留代码检查（SC-Y7）
- 单元测试 PASS（SC-Y8~Y9，含 snapshot_renderer + 4 agent test）
- E2E + 回归（SC-Y10~Y12）
- 行为零回归（SC-Y13~Y14，scope filter + skill body）
- Telemetry 验证（SC-Y15）
- 评审 gate（SC-Y16~Y17）

### Section 5 — Dependencies & Out of Scope

**Upstream**：spec 017a / 014 / 010 / 015
**Downstream**：spec 018（Provider 替换接入点）
**Out of Scope**：skill / memory 进化算法 / SKILL.md schema 升级 / verdict-debate-risk prompt 外置 / regime_tags 重设计 / 配置热重载 / load_skill_tool 删除 / Anthropic prompt cache 配置

### Section 6 — Implementation Outline

3 commit 单 PR，详见 Q6 决策。预估 ~35 task，2.5 天。

## Approaches Considered（核心 6 决策综合）

每个 Q 都列了 2-3 个 alternative，详见上面 6 项决策段落。整体架构哲学：
- "explicit > magic"：不引入 include 机制 / 不引入 DI 容器 / 不引入 fallback flag
- "isolation > coupling"：snapshot_renderer 独立模块 / nodes/agents.py 持有 wiring
- "no fallback"：直接删旧路径，靠 git revert 回滚
- "scope-driven evolution"：本 spec 不动 skill / memory 算法（留 spec 018），不动行数限制 / cache 策略（留 spec 018）

## Decision

按 6 项决策落地：C / Skill-A / B / D / A / B / B。整合范围：~16 文件，~1500 行 diff。

## Open Threads（已 spot-check 解决，2026-05-08）

### ✅ DefaultMemoryProvider 路径错误（已发现，决策 Option-2）

**发现**：017a `DefaultMemoryProvider` 设计与 spec 014 实际目录结构**不匹配**：

| 字段 | 017a 期望 | spec 014 实际 |
|---|---|---|
| Patterns | `agent_memory/<agent_id>/patterns.md`（单文件） | `agent_memory/<agent_id>/patterns/*.md`（子目录，目前空） |
| Cases | `agent_memory/<agent_id>/cases.jsonl`（每 agent 一个 jsonl） | `agent_memory/cases/<cycle_id>.md`（**全局** per-cycle markdown，YAML frontmatter + body 含 4 个 agent analyses） |

**实际影响**：017a 落地后 `DefaultMemoryProvider` 是僵尸代码，永远返回"暂无历史记忆"。但**没人发现** —— 因为 017a 只交付基建未集成，没人真的调用过。

**决策**：**Option-2 — DefaultMemoryProvider fix 推迟到 spec 018**
- 理由：spec 017b 已经够大（~16 文件 / 1500 行 diff），不在此扩；spec 018 EvolvingMemoryProvider 反正要重写 Provider，一起修整自然
- 实施方式：
  - **修订 FR-Y6**：BaseAgent.analyze() 的 `experience: str` 参数保留并直接传给 PromptBuilder.build()，PromptBuilder 把 experience 作为 `recent_memory` section 的覆盖输入（如 experience 非空，跳过 MemoryProvider 调用）
  - **新增 FR-Y6b**：`PromptBuilder.build(snapshot, portfolio, agent_analyses=None, experience: str = "")` 签名加 `experience` 参数；当 `experience` 非空时，作为 `recent_memory` section 内容；当 `experience` 为空时，调用 `memory_provider.get_recent_memory(...)` fallback（占位）
  - **本 spec 不修 DefaultMemoryProvider 内部实现**（仍按 017a path 设计返回"暂无历史记忆"）；spec 018 重写
- 风险：零回归 — experience 参数路径与 spec 014 verbal reinforcement 流转一致；现有 4 agent 仍走 experience 注入

### ✅ regime_tags 流转

`nodes/agents.py:56` 当前传 `regime_tags=regime_tags` 给 `cfg.agents.build()`。Q5 决策删除该参数。**已隐含在 FR-Y19 / FR-Y33**，C2 实施时一并清理 `nodes/agents.py:42` 的 `regime_tags = state["data"].get("regime_tags", [])` 与 `:56` 的传参。其他用处（`tracing.py` / `journal/search.py`）保留不动，是不同模块。

### ✅ prompt_template 字段在 config/

**spot-check 结果**：`grep -rn "prompt_template" config/` **无匹配**。现有 TOML 配置不含 prompt_template，C2 删除 dataclass 字段安全。

### ✅ graph.py agent 实例化

**spot-check 结果**：`grep -n "TechAgent\|ChainAgent\|NewsAgent\|MacroAgent" src/cryptotrader/graph.py` **无匹配**。graph.py 不直接实例化 4 agent，FR-Y35 视作 NOOP。

## Decision Updates（基于 spot-check 结果，2026-05-08）

| 字段 | 旧定义 | 新定义 |
|---|---|---|
| FR-Y6 | experience 参数被 PromptBuilder 视作 recent_memory section 的覆盖输入 | experience: str 参数保留，PromptBuilder.build() 加 `experience: str = ""` 参数；非空时作为 recent_memory section；空时 fallback 调 MemoryProvider |
| FR-Y6b（新增） | — | PromptBuilder.build() 签名扩展：`build(snapshot, portfolio, agent_analyses=None, experience: str = "")`；experience 非空 → 跳过 memory_provider.get_recent_memory；experience 空 → 走原 017a 路径（DefaultMemoryProvider 占位） |
| OOS（新增条目） | — | DefaultMemoryProvider 的 patterns / cases 实际路径修复（与 spec 014 目录对齐）→ spec 018 |
