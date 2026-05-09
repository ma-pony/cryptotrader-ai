# 规格质量检查清单：Memory Evolution（spec 018）

**目的**：在进入 plan 阶段前，验证 spec 的完整性与质量
**创建时间**：2026-05-09
**关联文件**：[spec.md](../spec.md)

## 内容质量

- [x] 不含实现细节（语言 / 框架 / API）—— 含必要的 trilogy 协议契约接口签名（spec 014/17a 既有 dataclass 名称：PatternRecord / CaseRecord / Maturity / PnLTrack；spec 014/17a 既有节点路由：risk_gate / journal_trade / journal_rejection），属于"接口约定"
- [x] 聚焦用户价值与业务诉求（5 user story 明确价值与优先级）
- [x] 面向非技术干系人可读（中文表述）
- [x] 所有强制 section 完成（User Scenarios / Requirements / Success Criteria / Assumptions）

## 需求完整性

- [x] 不含 [NEEDS CLARIFICATION] 标记
- [x] 需求可测试且无歧义（33 条 FR-Z 全部含具体可验证条件）
- [x] 成功标准可度量（SC-Z1..Z20 均含具体阈值或断言）
- [x] 成功标准与技术无关（除 trilogy 协议契约必需的接口名）
- [x] 所有验收场景已定义（5 user story 各含 acceptance scenarios）
- [x] 边界 case 已识别（8 条 edge cases）
- [x] 范围明确划界（Out of Scope 显式划分"移至 spec 019" / "移至 spec 020" / "本 spec 不动" 三类）
- [x] 依赖与假设已识别（Upstream / Downstream / 7 条 Assumptions）

## 功能就绪度

- [x] 全部 functional requirement 含明确验收标准
- [x] User scenarios 覆盖主要工作流（4 agent 收 memory / FSM 转换 / IVE 归档 / 迁移 / 前端可视）
- [x] 功能满足 Success Criteria 中的可度量结果
- [x] 实现细节未渗入规格（接口签名为协议契约必需）

## 备注

- 本 spec 是 trilogy（016 / 017 / 018-020）切分后的 Memory 子域，必须保留 spec 014/17a 既有 dataclass / Maturity Literal / 节点名以兼容现有架构
- 4 项 spot-check 在 brainstorm 阶段已完成，对应修订均已落入 spec.md（FR-Z6 用 PatternRecord / FR-Z11 沿用现有 4 状态 + 加 archived / FR-Z23 用 risk_gate 节点名 / FR-Z27 用 components/layout/sidebar.tsx 路径）
- 标记不通过项需在进入 `/speckit.clarify` 或 `/speckit.plan` 前修复 —— 本检查清单全部通过
