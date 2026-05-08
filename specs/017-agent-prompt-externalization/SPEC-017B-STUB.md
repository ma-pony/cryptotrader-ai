# Spec 017b 立项 Stub（待决定 spec 编号）

**状态**：未立项 — 仅记录 spec 017a 拆分时迁出的内容，方便后续 `/spex:brainstorm` / `/speckit-specify` 启动

**关联**：
- 上游：spec 017a（agent-prompt-externalization 本目录）— 已交付 PromptBuilder 基建
- 下游：与 spec 018（skill-evolution-v2）相对独立，可同时进行

## 范围

完成 spec 017a 拆出去的所有"集成"工作：

1. **重构 BaseAgent / ToolAgent**（`src/cryptotrader/agents/base.py`）
   - 构造器加必填 `prompt_builder: PromptBuilder` 参数
   - `analyze()` 改为调 `prompt_builder.build()` 而非 `self.role_description + ANALYSIS_FRAMEWORK`
   - 删除或弃用 `role_description` 字段（评估对 `security.py` 注释 / `extract_content` 等的影响）
   - ToolAgent 的 `SkillsInjectionMiddleware.build_system_addendum()` 调用：要么由 SkillProvider 替代，要么保留并记录技术债

2. **重构 AgentRegistry**（`src/cryptotrader/config.py:_resolve_role` / `_resolve_skills`）
   - 让 `AgentRegistry.build()` 注入 PromptBuilder 而非 `role_description` 字符串
   - 评估 `prompt_template` 字段：是否还需要（PromptBuilder 已读 `config/agents/*.md`），还是把 `prompt_template` 重定向到 PromptBuilder 入口
   - 删除 `_resolve_skills` 拼接到 `agent.role_description` 的旧路径

3. **创建 4 个 agent 配置文件**
   - `config/agents/tech.md` / `chain.md` / `news.md` / `macro.md`
   - system_prompt 直接搬运当前 `src/cryptotrader/agents/<name>.py:ROLE` 字符串
   - output_schema 段照搬 `ANALYSIS_FRAMEWORK` 中的 JSON schema 部分

4. **重构 4 个 agent 类**
   - `src/cryptotrader/agents/{tech,chain,news,macro}.py`
   - 删除 ROLE 常量
   - 构造器改必填 `prompt_builder`
   - `_build_prompt()`（snapshot 渲染）保留还是迁入 PromptBuilder 待评估

5. **更新调用方**
   - `src/cryptotrader/nodes/agents.py`：启动期实例化 `DefaultMemoryProvider` / `DefaultSkillProvider` / 4 PromptBuilders
   - `src/cryptotrader/graph.py`：检查并同步
   - 4 个 agent 实例化处注入 prompt_builder

6. **退役 SkillsInjectionMiddleware**
   - `src/cryptotrader/agents/skills/middleware.py:SkillsInjectionMiddleware` 是否完全删除还是保留作为 LangChain `create_agent` 的旧路径兼容
   - 评估对 `_resolve_skills` 等其他调用方影响

7. **E2E 测试**
   - `tests/test_e2e_prompt_externalization.py`：单次 mocked cycle 跑完 4 agent → debate → verdict → risk
   - Telemetry 8 字段断言
   - SC-X7 token 节省 < 15% 差异观测

8. **CI Gate**
   - `grep -rn "^ROLE\s*=" src/cryptotrader/agents/` 必须返回空
   - `wc -l src/cryptotrader/agents/{tech,chain,news,macro}.py` 各 < 150

## 已迁出的 FR / SC（来自 spec 017）

- FR-X3（4 agent config 文件）
- FR-X16 / FR-X17（ROLE 退役 + 构造器签名变更）
- SC-X1（4 个 config 文件存在）
- SC-X4（agent 迁移完成判定）
- SC-X5（E2E mocked cycle）
- SC-X6（cycle 中 telemetry 字段）
- SC-X7（token 差异 < 15%）
- SC-X8（grep ROLE 返回空）
- SC-X9（agent 文件 < 150 行）

## 已迁出的 Task（来自 spec 017 tasks.md）

- T015-T020（TechAgent migration）
- T021-T034（Chain/News/Macro + nodes/agents.py + graph.py + grep gate + wc -l）
- T035-T037（E2E + telemetry + token diff）
- T038-T043（Polish + final gates）

## 启动建议

立项时按以下顺序：

1. `/spex:brainstorm`（决策："SkillsInjectionMiddleware 是删还是保留？" "AgentRegistry.prompt_template 字段是否退役？" "agent._build_prompt(snapshot) 渲染逻辑迁入 PromptBuilder 还是保留在 agent？"）
2. `/speckit-specify`（基于 brainstorm 结论生成 spec）
3. `/spex:review-spec`（确保覆盖所有迁出 FR / SC）
4. `/speckit-plan` + `/speckit-tasks`
5. `/speckit-implement`（建议 **不** 用 `/spex:ship` 全自动 — 集成颗粒度大，建议分 commit 手动审视）

## 风险

- BaseAgent / ToolAgent 改动会影响：journal 写入、test_*_agent.py 等多处。预计回归测试范围 ≥ 50 个测试文件
- AgentRegistry 是 `arena` CLI / `arena serve` 的核心入口，重构需谨慎
- `SkillsInjectionMiddleware` 退役会让现有 skills 注入路径完全切换到 PromptBuilder，spec 018 的进化版 SkillProvider 也得跟上
- spec 014 既有 cases.jsonl 真实 schema 与 DefaultMemoryProvider `_format_case` 期望字段（case_id / context / outcome / pnl）若不匹配，需要先抽样验证

## 时间估算

- Phase 1（base.py / config.py 重构 + middleware 评估）：1 天
- Phase 2（4 agent + 4 config）：0.5 天
- Phase 3（E2E + 回归）：0.5 天
- 合计：2 天工作量
