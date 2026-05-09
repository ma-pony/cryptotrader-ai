# Spec 评审：Agent Prompt Builder Integration（spec 017b）

**关联 spec**：[specs/018-agent-prompt-builder-integration/spec.md](spec.md)
**评审时间**：2026-05-08
**评审人**：Claude（spex:review-spec，ship pipeline stage 2）

## 总体评估

**状态**：✅ SOUND

**摘要**：spec 由完整 brainstorm 6 决策 + 4 项 open thread spot-check 后生成；39 条 FR + 17 条 SC + 5 user stories 全部具体可测；Out of Scope 显式划分"移至 spec 018"与"本 spec 不动"两类，边界清晰。无 P0 / P1 issues，2 条 P3 可选改进。

## 完整性：5/5

- ✓ 强制 section 全部完成（User Scenarios / Requirements / Success Criteria / Assumptions / Dependencies）
- ✓ 推荐 section 完整（Edge Cases / Out of Scope / Reversibility / Implementation Outline）
- ✓ 39 条 FR-Y 覆盖 9 子模块（Configuration / base.py / snapshot_renderer / 4 agent / config.py / SkillsInjectionMiddleware 删除 / DefaultSkillProvider / nodes/agents.py / graph.py / Telemetry / Migration）
- ✓ 6 条 Edge Cases 显式列出（含 backtest_mode / 同 cycle 多次调用 / experience 注入路径等）

## 清晰度：5/5

- ✓ 无 [NEEDS CLARIFICATION] markers
- ✓ 全部使用 MUST / MUST NOT，无 should/might/could
- ✓ 接口签名（`BaseAgent.__init__` / `PromptBuilder.build` / `AgentsConfig.build`）明确
- ✓ Commit 序列（C1/C2/C3）边界清楚

## 可实现性：5/5

- ✓ 3 commit 单 PR 计划具体（每个 commit 文件范围 + diff 估算）
- ✓ 删除清单具体到文件路径 + 类 / 方法名
- ✓ Upstream / Downstream 依赖识别完整（spec 017a / 014 / 010 / 015）
- ✓ Out of Scope 显式排除"DefaultMemoryProvider 路径修复"等可能扩散的修复 → 锁定 spec 018

## 可测试性：5/5

- ✓ SC-Y1..Y17 含具体阈值（"返回空" / "< 150 行" / "≥ 6 用例 PASS"）
- ✓ User Story Acceptance Scenarios 全部 Given/When/Then 格式
- ✓ FR / SC / User Story 三层一致性高（每条 FR 都有 SC 验收）

## Constitution 对齐

无 `.specify/memory/constitution.md` 项目原则定义。按 CLAUDE.md：
- ✓ Markdown 中文
- ✓ 不修改 CLAUDE.md
- ✓ 仅在 `specs/018-*/` 范围内创建文件
- ✓ 不引入新 runtime 依赖
- ✓ 与 spec 014 / 017a 既有架构契约兼容

## 跨 Spec 一致性

- ✓ 与 spec 017a 决策对齐：保留 PromptBuilder 公开 API（构造签名 / `build()` 返回值结构）；扩展时仅加 `experience: str = ""` 参数（向后兼容）
- ✓ 与 spec 016 决策对齐：D-PA-01（Markdown frontmatter）/ D-MW-01（cases + patterns 混合）/ D-MW-02 / D-MW-03（Default Provider 实现）
- ✓ 显式声明 Out of Scope 中 "DefaultMemoryProvider 路径修复 → spec 018"，避免与 spec 018 进化算法实现并发冲突

## 推荐改进

### Critical（必须修复）

- 无

### Important（应当修复）

- 无

### Optional（建议）

1. **SC-Y15 是手动 smoke test，不在自动 CI**：建议在 spec 落地后增加 prod smoke test checklist（独立 markdown），以便发版前 reviewer 确认 OTel trace 后端可见 8 字段。**严重程度**：P3。本 spec 不阻塞，可在 review-code 阶段补充
2. **FR-Y6b 中 PromptBuilder.build() 加 experience 参数会修改 spec 017a 已固化的 API**：虽然新加可选参数（`experience: str = ""`）是向后兼容的，但建议在落地后更新 spec 017a 的 `contracts/prompt-builder.md` 反映新签名。**严重程度**：P3

## 结论

spec 017b（目录 018）准备就绪，可进入 plan 阶段。

**是否就绪**：是
