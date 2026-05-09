# 规格质量检查清单：Agent Prompt Builder Integration（spec 017b）

**目的**：在进入 plan 阶段前，验证 spec 的完整性与质量
**创建时间**：2026-05-08
**关联文件**：[spec.md](../spec.md)

## 内容质量

- [x] 不含实现细节（语言 / 框架 / API）—— 含必要的 trilogy 协议契约接口签名（继承 spec 017a 已稳定的 PromptBuilder API），属于"接口约定"而非"实现细节"
- [x] 聚焦用户价值与业务诉求（5 个 user story 明确价值与优先级）
- [x] 面向非技术干系人可读（中文表述）
- [x] 所有强制 section 完成（User Scenarios / Requirements / Success Criteria / Assumptions）

## 需求完整性

- [x] 不含 [NEEDS CLARIFICATION] 标记
- [x] 需求可测试且无歧义（39 条 FR-Y 全部含具体可验证条件）
- [x] 成功标准可度量（SC-Y1..Y17 均含具体阈值或断言条件）
- [x] 成功标准与技术无关（除 trilogy 协议契约必需的接口签名 + grep / wc -l 等 CI gate）
- [x] 所有验收场景已定义（5 user story 各含 acceptance scenarios）
- [x] 边界 case 已识别（6 条 edge cases）
- [x] 范围明确划界（Out of Scope 显式分"移至 spec 018" / "本 spec 不动"两类）
- [x] 依赖与假设已识别（Upstream / Downstream / 7 条 Assumptions）

## 功能就绪度

- [x] 全部 functional requirement 含明确验收标准
- [x] User scenarios 覆盖主要工作流（配置驱动 / Skill 零回归 / Snapshot 零回归 / Backtest 零回归 / Telemetry 验证 5 条 P1-P2）
- [x] 功能满足 Success Criteria 中的可度量结果
- [x] 实现细节未渗入规格（接口签名为协议契约必需，非实现选择）

## 备注

- 本 spec 是 trilogy（016 / 017a / 017b / 018）中的"集成切换"环节，必须保留 PromptBuilder.build() / Provider Protocol 接口签名以维持 spec 017a 已上线的契约
- 标记不通过项需在进入 `/speckit.clarify` 或 `/speckit.plan` 前修复 —— 本检查清单全部通过
