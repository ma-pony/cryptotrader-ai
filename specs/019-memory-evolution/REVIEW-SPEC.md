# Spec 评审：Memory Evolution（spec 018）

**关联 spec**：[specs/019-memory-evolution/spec.md](spec.md)
**评审时间**：2026-05-09
**评审人**：Claude（spex:review-spec，ship pipeline stage 2）

## 总体评估

**状态**：✅ SOUND

**摘要**：spec 由完整 brainstorm 6 决策 + 4 项 spot-check 后生成；33 条 FR + 20 条 SC + 5 user stories 全部具体可测；Out of Scope 显式分"移至 spec 019" / "移至 spec 020" / "本 spec 不动" 三类，trilogy 边界清晰。无 P0 / P1 issues，3 条 P3 advisory。

## 完整性：5/5

- ✓ 强制 section 全部完成
- ✓ 推荐 section 完整（Edge Cases / Dependencies / Out of Scope / Reversibility / Implementation Outline）
- ✓ 33 条 FR-Z 覆盖 9 子模块（Schema/Migration / Provider / FSM / Pareto / IVE / nodes / Frontend / Telemetry / Migration Tooling）
- ✓ 8 条 Edge Cases 显式列出（含 IVE LLM timeout / FSM 字段缺失 fallback / 迁移损坏 frontmatter 等）

## 清晰度：5/5

- ✓ 无 [NEEDS CLARIFICATION] markers
- ✓ 全部 MUST / MUST NOT，无含糊词
- ✓ 接口签名（PatternRecord 字段 / Maturity Literal / evaluate_node / FSM 状态转换条件）明确
- ✓ Commit 序列（C1/C2/C3/C4）边界清楚

## 可实现性：5/5

- ✓ 4 commit 单 PR 计划具体（每 commit 文件范围 + diff 估算）
- ✓ 删除清单具体到 class / function 名（DefaultMemoryProvider class / journal 函数 / verbal_reinforcement 接口契约）
- ✓ Upstream / Downstream 依赖完整（spec 017a/b/14/15/10/16）
- ✓ Out of Scope 显式排除"Anthropic prompt cache" / "git lineage" 等可能扩散项 → 锁定 spec 020

## 可测试性：5/5

- ✓ SC-Z1..Z20 含具体阈值（"≥ 8 用例 PASS" / "≥ 10 用例 PASS" / "≥ 12 用例 PASS" / "返回空" / "通过基线 ≥ 2173"）
- ✓ User Story Acceptance Scenarios 全部 Given/When/Then 格式
- ✓ E2E 测试条件具体（mocked cycle 含 fsm_transition + 5 ive_classification + 6 telemetry 字段）

## Constitution 对齐

无 `.specify/memory/constitution.md` 项目原则定义。按 CLAUDE.md：
- ✓ Markdown 中文
- ✓ 不修改 CLAUDE.md
- ✓ 仅在 `specs/019-*/` + 既有 `src/` / `web/` 范围内创建/修改文件
- ✓ 不引入新依赖
- ✓ 与 spec 014 / 17a / 17b 既有架构契约兼容（PatternRecord / CaseRecord / Maturity Literal / Provider Protocol）

## 跨 Spec 一致性

- ✓ 与 spec 017a 决策对齐：保留 PromptBuilder 公开 API（构造签名 / `build()` 返回值）；DefaultMemoryProvider class 删除
- ✓ 与 spec 017b 决策对齐：experience 参数路径仍兼容（spec 014 verbal_reinforcement 流转不变）
- ✓ 与 spec 016 决策对齐：D-MW-01..03 / D-EV-02..04 / D-DS-01 / D-EVAL-01 全部覆盖
- ✓ 与 spec 014 既有架构对齐：沿用 PatternRecord / CaseRecord / PnLTrack / Maturity Literal（不重新定义）；新字段都有 default 兼容旧实例

## 推荐改进

### Critical（必须修复）

- 无

### Important（应当修复）

- 无

### Optional（建议）

1. **SC-Z11 graph 节点位置验证 — 自动化测试覆盖**：spec 强调"evaluate 节点在 risk_gate 之后、journal_trade/rejection 之前"，但仅靠人工 review 易疏漏。建议加 `tests/test_graph_structure.py` 测 LangGraph 编译后的 node 顺序。**严重程度**：P3
2. **FR-Z31 备份提示文案考虑双语**：迁移脚本 print 提示当前是中文；spec 014 落地的 CLI 大多 English。建议同时含中英双语。**严重程度**：P3
3. **SC-Z15 E2E 性能预期未定义**：单 cycle 跑全链路（4 agent + IVE 5 case + evaluate_node）可能耗时较长。建议加 SC：单 cycle 总耗时 ≤ 30s（不含 LLM 实际推理）。**严重程度**：P3

## 结论

spec 018（目录 019）准备就绪，可进入 plan 阶段。

**是否就绪**：是
