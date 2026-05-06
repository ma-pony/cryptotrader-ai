# Spec Review Report: 014-agent-skills-protocol-migration

**Reviewer**: spex:review-spec
**Date**: 2026-05-06
**Subject**: [spec.md](./spec.md)
**Verdict**: 🟡 **APPROVE WITH 6 GAPS TO ADDRESS**（4 个建议在 plan 阶段补，2 个建议在 spec 直接补）

---

## Soundness（设计合理性）

### ✅ 通过

- 文件存储 + 单写者模型架构合理，契合 4 agent 同进程、reflection 异步触发的现实
- "5 layer 防过拟合" know-how 显式保留是正确决定（交易场景独特价值）
- Out of Scope 段明确，避免 scope creep
- Acceptance scenarios 都用 Given/When/Then 形式，行为可机器验证

### ⚠️ 1 个潜在缺陷

**REVIEW-1（建议 spec 补）：reflection job 失败模式不完整**
- FR-018 说"reflection 失败不阻塞下一个 cycle"
- 但**未定义"失败"的表现**：进程崩溃？文件部分写入？解析异常？
- Edge case "reflection 在写第 3 个文件时进程被强杀"提到了，但 FR 没强制要求
- **建议补 FR-018a**：reflection job MUST 使用原子写（temp file + rename）确保部分写入不会留下损坏文件

---

## Completeness（缺漏）

### ⚠️ 4 个具体缺漏

**REVIEW-2（建议 plan 阶段补）：reflection job 触发机制未指定**
- FR-014 定义了 reflection 的输入输出，但**未说明谁触发它**
- 当前系统：`nodes/data.py:verbal_reinforcement` 通过 `[experience] every_n_cycles` 配置触发；删了之后呢？
- Assumptions 段提到"reflection 触发频率与当前一致"——但实现位置未定
- **plan.md 必须明确**：reflection 是 cron-style 独立 job？还是依然嵌在 graph 节点？还是 CLI 触发？

**REVIEW-3（建议 spec 直接补）：5 层防过拟合未列举**
- FR-016 说"完整保留 5 层"，user story 3 Independent Test 列了 5 维度
- 但**spec 主体未明确这 5 层是什么**（user story 提了一句，FR 没复现）
- **建议补在 FR-016 下**："具体包括：(1) regime-aware 胜率统计 (2) 最少样本量门槛 (3) 全局 vs 区段统计差距识别 (4) 对手验证 (5) 时间衰减"

**REVIEW-4（建议 spec 直接补）：`manually_edited` 设置方式未定义**
- Edge Case + FR-019 提到该字段保护人手编辑
- 但**没说怎么 set**：用户手编辑后自己加 frontmatter 字段？pre-commit hook 自动加？CLI `arena skills mark-edited`？
- Brainstorm 阶段我提到 "pre-commit hook detect body diff" 但用户并未确认
- **建议补 FR-019a**：用户手工设置（添加 `manually_edited: true` 到 frontmatter）；本 spec 不强制实现自动检测机制

**REVIEW-5（建议 plan 阶段补）：性能基线缺**
- SC-001 要求 token 下降 ≥ 30%，但**全 spec 无 cycle latency 保证**
- 4 agent × 每 cycle × N 个文件 IO + YAML parse = 性能开销未知
- 估算：~50 文件 × 4 agent ≈ 200 文件读 + parse / cycle，磁盘 SSD 应在 50ms 内，但 git lfs / NFS 等场景可能 > 500ms
- **plan.md 应给基线**：`load_agent_skills()` p95 ≤ 100ms（或类似 SC）
- **缓解方案在 plan**：在 LiveExchange 已用的 `_markets_loaded` 模式下，loader 也可缓存到内存，仅在 reflection 写入后失效

---

## Implementability（可实施性）

### ⚠️ 1 个建议补充

**REVIEW-6（建议 plan 阶段澄清）：跨 agent 引用消歧规则**
- FR-005 + FR-020 都允许两种形式：`applied: name` 与 `applied: agent::name`
- 当 `name` 在多个 agent 都存在时，简短形式如何消歧？
- 选项：(a) 报 ambiguous 跳过 (b) 默认匹配当前 cycle 的发起 agent (c) 全部匹配并广播 PnL
- **plan.md 需选定**——本 spec 不强制，但实现必须有确定行为

---

## Ambiguities（含糊语句）

### ⚠️ 2 个发现

**AMBI-1：FR-019 的"短暂时间窗"**
- 原文："reflection job 等待短暂时间窗后写入"
- "短暂"是 1ms？100ms？1s？
- **建议在 plan 阶段定数**（如 reflection 写入 wait 100ms 让 agent 当前读取批结束）

**AMBI-2：SC-008 的"启动时间"**
- "从仓库 clone 到运行起 trading cycle 的时间不变或下降"
- 没给基线值（当前是多久？2 分钟？10 分钟？）
- **plan 阶段建议补**：测一次当前基线作为 anchor

---

## 协议对齐（Anthropic Skills 协议合规性）

✅ FR-004 列的 frontmatter 字段与 Anthropic Skills SKILL.md 协议**一致**（name + description 是核心，其他 metadata 可扩展）
✅ 文件命名 + 目录树结构与协议一致
✅ description 注入到 prompt（FR-010）符合协议"agent 看到 description 决定是否调用"的设计意图
⚠️ **本期采用静态注入而非 tool-calling**——协议本意是 agent 通过 tool 自主 load_skill；spec 在 Out of Scope 里明确推迟到 follow-up，符合阶段性合理

---

## Constitution 对齐

`.specify/memory/constitution.md` 仍是模板未填写状态。本次跳过 constitution 检查（合理）。

---

## 总结：6 个 gap 处理建议

| Gap | 严重度 | 处理时机 | 处理方式 |
|---|---|---|---|
| REVIEW-1 reflection 原子写 | 中 | spec 现在补 | 加 FR-018a |
| REVIEW-2 reflection 触发机制 | 中 | plan 阶段 | plan.md 明确（独立 cron / graph 节点 / CLI） |
| REVIEW-3 5 层防过拟合明示 | 低 | spec 现在补 | FR-016 内联列举 |
| REVIEW-4 manually_edited 设置 | 中 | spec 现在补 | 加 FR-019a |
| REVIEW-5 性能基线 | 中 | plan 阶段 | plan.md 测基线 + 加 SC |
| REVIEW-6 命名消歧规则 | 中 | plan 阶段 | plan.md 选定 |
| AMBI-1 "短暂时间窗" | 低 | plan 阶段 | plan.md 定数 |
| AMBI-2 启动时间基线 | 低 | plan 阶段 | plan.md 测基线 |

## 决策

- **3 项立刻补 spec**（REVIEW-1, REVIEW-3, REVIEW-4）：5 分钟改动
- **5 项推迟到 plan 阶段**（REVIEW-2, REVIEW-5, REVIEW-6, AMBI-1, AMBI-2）

补完 3 项后 spec 即可进入 `/speckit-plan`。
