# Spec 评审：Agent Prompt Externalization

**关联 spec**：[specs/017-agent-prompt-externalization/spec.md](spec.md)
**评审时间**：2026-05-08
**评审人**：Claude（spex:review-spec）

## 总体评估

**状态**：✅ SOUND

**摘要**：spec 结构完整、需求明确、可测试性强；4 个 user story 覆盖核心动机（配置驱动 / Provider 协议 / Telemetry / 代码精简）；19 条 FR 与 10 条 SC 一一对应；T1-T6 任务图清晰可执行。仅发现 2 处轻微不一致（P3 改进项），无 P0 / P1 issues。

## 完整性：5/5

### 结构

- ✓ 强制 section 全部完成（User Scenarios / Requirements / Success Criteria / Assumptions）
- ✓ 推荐 section 完整（Edge Cases / Dependencies / Out of Scope / Reversibility / Implementation Outline）
- ✓ 无 TBD / 占位符

### 覆盖

- ✓ 19 条 FR-X 覆盖配置层 / 运行时 / Token 预算 / Memory & Skill 拼接 / Migration / Telemetry 6 个子模块
- ✓ 7 条 Edge Cases（含空 memory / 空 skill / snapshot 缺字段 / token 超 budget / config 损坏 / slot_overrides 错误 / section 重复槽位）
- ✓ Success Criteria 10 条全部可度量

## 清晰度：5/5

### 语言质量

- ✓ 全部使用 MUST / MUST NOT，无 should/might/could 等弱化语
- ✓ 无 "fast" / "user-friendly" / "etc." 等含糊词
- ✓ 接口签名（PromptBuilder 构造器、enforce 方法）明确，避免歧义

### 检测到的轻微不一致

1. **SC-X5 引用了 telemetry 字段 `prompt_source = "config"`，但 FR-X18 列出的 8 个 telemetry 字段中没有 `prompt.builder.prompt_source`**
   - 严重程度：P3（轻微，不阻塞实现）
   - 建议：在 FR-X18 字段列表中补一条 `prompt.builder.prompt_source` (str, e.g. "config")，或在 SC-X5 改为引用 `sections_included` 非空作为"prompt 来自 config"的代理证据

2. **SC-X9 "4 个 agent 文件每个 < 150 行" 是 quality 度量，spec template 警示避免实现细节型 SC**
   - 严重程度：P3（合理，本项目是内部库，行数为 maintainability 的合理 proxy）
   - 建议：可保留，但补一句业务理由（"业务逻辑可读性"）

## 可实现性：5/5

### Plan 生成

- ✓ T1-T6 任务图带依赖箭头（T1 → {T2 ∥ T3 ∥ T4 ∥ T5} → T6）
- ✓ 全部文件路径具体（src/cryptotrader/agents/prompt_builder.py、config/agents/<name>.md、tests/test_*.py）
- ✓ 依赖明确（Upstream 016/014/010、Downstream 018、External PyYAML 已有）
- ✓ 范围合理（仅 4 个 analysis agent，不含 verdict/debate/risk）

### 协议契约设计

- ✓ MemoryProvider / SkillProvider 用 `Protocol` 抽象，不绑死实现
- ✓ DefaultMemoryProvider / DefaultSkillProvider 复用 spec 014 现有目录结构（agent_memory/、agent_skills/），无破坏性变更
- ✓ EnforceResult dataclass 字段齐全，便于 telemetry 直接消费

## 可测试性：5/5

### 验收

- ✓ 每条 FR 对应可触发的代码路径
- ✓ SC-X1..X10 含具体阈值（"7 用例 PASS"、"5 用例 PASS"、"差异 < 15%"、"返回空"）
- ✓ User Story Acceptance Scenarios 全部 Given/When/Then 格式
- ✓ E2E 测试文件路径明确（`tests/test_e2e_prompt_externalization.py`）

### 度量手段

- ✓ Token 误差："估算 token vs tiktoken 实测，CJK + ASCII 混合样本 < 10%"——可自动化验证
- ✓ ROLE 退役：`grep -rn "^ROLE\s*=" src/cryptotrader/agents/` 返回空——CI gate
- ✓ Telemetry：8 个 attribute 字段明确，触发 cycle 后查询 trace 即可验证

## Constitution 对齐

无 `.specify/memory/constitution.md` 文件存在；按 CLAUDE.md 规则：
- ✓ Markdown 内容使用简体中文
- ✓ 不修改 CLAUDE.md
- ✓ 仅在 `specs/017-*/` 范围内创建文件
- ✓ 不引入新 runtime 依赖
- ✓ 与 spec 014（agent_skills/agent_memory）目录约定兼容

## 跨 Spec 一致性

- ✓ 与 spec 016 决策对齐：D-PA-01（Markdown frontmatter）、D-PA-02（system/user-tail 槽位）、D-PA-03（config/agents/）、D-MW-01（patterns + cases 混合）、D-MW-02（DefaultMemoryProvider）、D-MW-03（DefaultSkillProvider）
- ✓ 显式声明 spec 018 依赖本 spec 的 Provider 协议接口（trilogy 衔接清晰）
- ✓ Out of Scope 明确划清与 spec 018（skill/memory 进化算法）边界

## 推荐改进

### Critical（必须在实现前修复）

- 无

### Important（应当修复）

- 无

### Optional（建议）

- [ ] FR-X18 telemetry 字段列表补一条 `prompt.builder.prompt_source` 与 SC-X5 对齐（或调整 SC-X5 不再引用该字段）
- [ ] SC-X9 行数限制补一句业务理由（"业务逻辑可读性，避免 prompt 字符串遮蔽业务代码"）
- [ ] 考虑在 Edge Cases 补一条 "config 文件含 BOM / 非 UTF-8 编码" 的处理（fail-fast 即可，但显式列出更稳）

## 结论

spec 017 是 trilogy 中合格的"基建 spec"：范围聚焦（仅 4 个 agent + PromptBuilder + TokenBudget）、接口契约清晰（MemoryProvider/SkillProvider 协议）、迁移策略激进但可控（T1-T6 颗粒度足够小可逐 commit revert）。已具备进入 `/speckit-plan` 的条件。

**是否就绪**：是

**下一步**：
1. （可选）修复 2 条 P3 改进项（FR-X18 / SC-X9）
2. 进入 `/speckit-plan` 生成 plan.md
3. 进入 `/speckit-tasks` 生成 tasks.md
4. 通过 `/spex:review-plan` 后 commit
