# Spec Review: Spec 022 — Agent-Native Skill Protocol Layer

**Spec**: specs/025-agent-native-skill-protocol/spec.md
**Date**: 2026-05-18
**Reviewer**: Claude (spex:review-spec)

## Overall Assessment

**Status**: ✅ SOUND

**Summary**：外部 SKILL.md 协议层 spec 结构完整 — 4 user stories（3 P1 + 1 P2）+ 25 FR + 10 binary-verifiable SC，全部锚定具体既有代码点。9 项关键设计决策已在 brainstorm 阶段 resolved（详 research.md）。同时闭 spec 021 T021 outstanding gap（SC-Q3 直接验证 `/api/memory/patterns total >= 3`）。Ready for plan + implementation。

## Completeness: 5/5

- ✓ 所有 mandatory sections present（User Scenarios / Requirements / Success Criteria / Key Entities / Dependencies / Assumptions / Out of Scope / Reversibility / Implementation Outline）
- ✓ 25 FR 覆盖 4 user story 全部需求（FR-022-1 到 24 + FR-022-X1）
- ✓ Edge cases 8 项
- ✓ Out of Scope 9 项明示边界
- ✓ Dependencies 区分 upstream / downstream / external tooling
- ✓ Reversibility 段落 9 项详细回退步骤

## Clarity: 5/5

- ✓ Requirements 全部 MUST 句式 + 具体文件路径
- ✓ Acceptance Scenarios 全部 Given/When/Then 三段式（4 stories × 3-4 scenarios = 14 scenarios）
- ✓ 0 NEEDS CLARIFICATION 标记（9 Q ambiguity 在 brainstorm 阶段已消除）
- ✓ Key Entities 4 个全含 schema 关键字段 + 持久化方式
- ✓ Edge cases 全部含触发条件 + 期望行为

## Implementability: 5/5

- ✓ FR 全部映射具体文件 + handler：
  - `agent_skills/_internal/` + `_external/`（路径迁移）
  - `src/api/routes/{events,memory,skills,metrics}.py`
  - `src/cryptotrader/{learning/evolving_skill_provider,nodes/journal,observability/heartbeat_metrics,ops/daemon}.py`
  - `scripts/{export_openapi,demo_external_client}.py`
- ✓ Dependencies 显式列出（spec 010 / 015 / 018 / 019 / 020a / 020b / 021）
- ✓ Constraints 现实（无新依赖 + 全 read-only + 复用 spec 020a aggregator 模式 + 单 API_KEY）
- ✓ Scope 可控（单 PR ≤ 4 commit ~1-2 周 / ~10 新文件 / ~150 行新代码）
- ✓ Reversibility 路径清晰（git revert + docker restart 即可）

## Testability: 5/5

- ✓ SC-Q1 ~ SC-Q10 全部含可执行 verification（curl / ls / grep / pytest / python script / 数字阈值）
- ✓ Acceptance Scenarios 明确触发条件 + 期望结果
- ✓ FR-022-14 OTel span attrs 约束便于 e2e 断言
- ✓ SC-Q3 直接闭 spec 021 T021 outstanding gap（`curl /api/memory/patterns total >= 3`）
- ✓ SC-Q7 baseline 测试不回归（2476 pass / 0 fail）

## 与 spec 模板对齐度

⚠️ **Note**：spec.md FR 含具体 src 路径（如 `src/api/routes/events.py`）— 违反 spec-kit 默认 "WHAT not HOW" 原则。**但本项目（spec 020a/b/c/021 全部）一致采用此约定**，spec 与 plan 边界模糊但便于单人项目维护，**不视为 issue**。

## Recommendations

### Critical / Important
无

### Optional (Nice to Have)

- [ ] **FR-022-X1 编号**：作为 `/skill/<name>` endpoint 的独立 FR，可在 plan 阶段重编号为 FR-022-25 与连续序列对齐（当前 X1 后缀因 spec 后期补加）
- [ ] **FR-022-22 lazy update**：spec 仅说 "/metrics 触发前 lazy update from aggregator"，可在 plan 阶段明确 lazy update 触发点（lock 配合 last_update_at 阈值？复用 spec 020a 模式即可）
- [ ] **Edge case "deprecated route"**：当前项目是否存在 deprecated routes 未确认，可在 plan 阶段 grep `@router.<method>.*deprecated=True` 验证；若 0 hits 则该 edge case 仅作为 future-proof 约束
- [ ] **SC-Q4 cursor "valid" 定义**：spec 写 "valid cursor"，可在 plan/contracts 阶段明确 — base64-url decode 后能正确还原 `(timestamp, trace_id)` 双元组（contracts/events-heartbeat.yaml 已含 example，等价于"valid"定义）
- [ ] **scheduler watchdog 衔接**：基于 P0 incident（5/18 18:57-19:57 silent miss 2.5h），可在本 spec 加 SC-Q11 "heartbeat 端点 evolution event 含 scheduler_alive 子类型"（每个 cycle 完成时 push 一条 system_health event）— **但属于范围扩展，建议进 spec 023 而非 spec 022**

### Risks Flagged

- 🟡 **path migration（FR-022-1 / 6）blast radius**：spec 019 ~50 处测试 fixture 引用 `agent_skills/<agent>/` — atomic mv 需 grep + sed 一次性改 + spec 020a/b 的 EvolvingSkillProvider 测试同步过。建议 plan 阶段加 task 显式 grep `agent_skills/` 全量审计
- 🟡 **arena serve 重启风险**：path migration commit ship 时需 restart 才能 reload skill provider。生产中 restart 可能导致 cycle 错过（5/18 17:56 restart 已重现 cycle 偏移 38min）。建议 plan 阶段 task 明确 ship 时 restart timing（cycle 间隔窗口内执行）

## Conclusion

Spec 结构完整、需求明确、可测可实现，且承接 trilogy + spec 020a/b/c + spec 021 关系清晰。9 项关键设计决策在 brainstorm 阶段已 resolved（详 research.md），0 NEEDS CLARIFICATION 残留。同时巧妙闭 spec 021 T021 outstanding gap。可进入 plan + implementation 阶段。

**Ready for implementation**: Yes

**No P0 / P1 issues**：✓
