# Spec 评审：Skill / Memory 进化前序研究

**Spec**：[`specs/016-research-skill-evolution-prior-art/spec.md`](spec.md)
**日期**：2026-05-08
**评审者**：Claude（spex:review-spec）

## 总体评价

**状态**：✅ SOUND（无 P0 / P1 问题）

**摘要**：规范完整、可实施、可验证。研究范围（8 项目 × 8 角度）、产物结构（4 类文档）、阶段交付（Phase 1/2 解锁下游 spec）都明确。少量文字含糊点和缺失的 NFR 段是可优化项，但都不阻断实施。建议直接进入 plan 或启动 Phase 1 研究。

---

## 完整性 Completeness：4.5 / 5

### 结构

- ✓ 所有必填段都存在：Purpose（Background）/ User Scenarios / Edge Cases / Functional Requirements / Success Criteria / Assumptions / Dependencies / Out of Scope / Reversibility
- ✓ 推荐段：Key Entities 已包含
- ✗ 缺独立的 **Non-Functional Requirements** 段（FR-R8 / FR-R9 隐含覆盖了"不引入依赖、产物隔离"，但形式上没单独列出）
- ✓ 无 TBD / 占位符

### 覆盖

- ✓ 9 条 FR 完整（每个对应可验证产物）
- ✓ 7 条 SC 完整（每条都对应可校验断言）
- ✓ 6 个 Edge Cases 显式列出
- ✓ 4 个 User Stories 各自有 acceptance scenarios
- ⚠️ 错误处理大部分用 Edge Cases 涵盖，没有单独的 "Error Handling" 章节标题（spec-kit 模板未硬性要求，可接受）

**问题**：

- 1 条建议：补 NFR 段，列明"研究产物可在 < 10 分钟内被 reviewer 全面 scan"等可量化非功能性指标。**严重度：optional**。

---

## 清晰度 Clarity：4 / 5

### 语言质量

- ✓ Tier 1/2/3 阅读深度定义具体（"主代码路径全读" vs "关键源码理解" vs "README + 代表性 source file"）
- ✓ Phase 1/2 边界明确（解锁哪个 spec 用哪个 SC 判定）
- ✓ "8 角度"在 Background + FR-R2 都列出明细
- ⚠️ 少量含糊措辞需收紧

### 发现的含糊点

1. **SC-R4**："若与 SC-R3 严重不匹配（例如建议 10 条但 decisions 只有 1 条）"
   - **问题**："严重不匹配" 是定性描述
   - **建议**：可改为"若 `decisions.md` 中 ADR 数量 < SC-R3 的建议数 × 0.5，视作研究深度不足"。给出可验证比率。

2. **Background 段**："skill 命中率偏低"
   - **问题**："偏低" 无量化基线
   - **建议**：改为"skill 命中率（按 verdict 引用 applied: 算）观察值约为 60%（10 笔决策样本，spec 015 记录）"。锚定到具体数据点便于 future review 比对。

3. **US-2 Independent Test**："架构师能在 4 小时内完成 spec 018 的初步 brainstorm 起草"
   - **问题**："4 小时" 是研究产物质量的间接代理
   - **建议**：拆为更直接的 testable 条件——"每个核心设计决策都能引用至少一个外部项目作为证据"已经在原文里更可验证；可考虑去掉"4 小时"或保留为"目标"而非"判定"。

**严重度**：都是 minor / optional —— 不阻断实施。

---

## 可实施性 Implementability：5 / 5

### 计划生成

- ✓ 可生成 plan：4 类产物文档 × 8 项目，单元清晰可分包
- ✓ Dependencies 列出：git / 网络访问 / 5GB 磁盘
- ✓ Phase 1 / Phase 2 自然形成 plan 的两个 milestone
- ✓ 范围适中：不涉及代码、不引入依赖、纯文档产出

**问题**：无。

---

## 可测试性 Testability：5 / 5

### 验证

- ✓ SC-R1：`ls specs/016-.../research/projects/ | wc -l == 8` 可机器验证
- ✓ SC-R2：8×8 矩阵每格非空可机器扫描
- ✓ SC-R3：计数 ≥ 10 的"建议"块可统计
- ✓ SC-R5：frontmatter 字段可正则匹配
- ✓ SC-R6 / SC-R7：显式判定条件 + reviewer signoff
- ✓ Edge Cases 大部分可程序化校验（"frontmatter `license:` 字段不能空"等）

**问题**：无。

---

## Constitution 对齐

`.specify/memory/constitution.md` 是未填充的模板（占位符 `[PROJECT_NAME]`、`[PRINCIPLE_1_NAME]` 等）。无可比对的实质性原则。**视为通过**（不存在违反对象）。

后续若用户填充 constitution，建议至少加入：
- "spec 不修改运行时代码 = 研究类 spec 必须 OOS 列明"
- "Phase 化交付的 spec 必须显式定义 phase 解锁条件"

---

## 改进建议

### 必须修复（实施前阻断项）

无。

### 应当修复

1. **SC-R4 量化**：把"严重不匹配"改为可计算的比率（建议 1/2 比例）。修改成本极低（一句话），但显著提升 reviewer 信任。

### 可选改进

2. 补独立的 **Non-Functional Requirements** 段，把 FR-R8 / FR-R9 抽出来重新组织（产物隔离 / 不引入依赖 / 50 行代码引用上限）。
3. Background 中"skill 命中率偏低"加锚定数据（10 笔决策、spec 015 观察）。
4. US-2 的"4 小时"判定可以放宽为"目标值"或删除（现有更直接的 acceptance scenario 已足够）。

---

## 结论

**可进入实施**：✅ 是 —— 所有 SC/FR 都满足 spec-kit 的 quality gate；ambiguity 项都是 optional 改进而非阻断点。

**下一步**：

1. 用户可选择直接接受当前版本进入 commit + Phase 1 研究启动
2. 或者先采纳上面的 Optional 改进（修改 SC-R4 / 补 NFR 段 / 锚定数据），再进入下一步
3. brainstorm skill 流程后续：commit spec → 启动 spec 017 的 brainstorm（同样模式）→ 最后 spec 018

无 P0 / P1 issues —— Phase 1 研究可立即启动，不阻断 spec 017 的 brainstorm。
