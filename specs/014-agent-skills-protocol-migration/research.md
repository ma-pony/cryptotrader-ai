# Phase 0 Research — 014 双层架构 v2

解决 spec review 推迟 + plan 阶段引入的决策点。

---

## R1：Reflection 触发 vs Curation 触发（双层架构修订）

**Decision**：两个流程**频率与触发完全不同**，分开实现：

- **Reflection（memory 层）**：在 graph 中作为 `nodes/reflection.py`，由 `nodes/data.py:verbal_reinforcement` 按 `[experience] every_n_cycles` 触发
  - 频率高（每 N cycles）
  - 自动
  - 失败不阻塞 cycle
  - 输出：`agent_memory/<agent>/patterns/*.md` 增删改
- **Curation（skills 层）**：通过 CLI `arena skills curate <name> [--llm]` 手工触发
  - 频率低（每周或按需）
  - 用户控制（手工 + 可选 LLM 辅助）
  - 输出：`agent_skills/<skill>/SKILL.md` 草稿（不直接 overwrite）
  - 本期不上 cron；follow-up 可加 `--all` 批量自动化

**Rationale**：
- Reflection 是数据层的 ETL，必须自动化
- Curation 是知识整理，需要人类 judgment（"也可以自定义"）
- 分开可以独立调整频率、独立失败处理

**Alternatives considered**：
- Curation 自动跑：风险大，LLM 可能错乱整理
- Reflection 也手工触发：违反"每 cycle 反馈数据要尽快进 memory"原则

---

## R2：性能 SC

**Decision**：
- `load_agent_skills(agent)` p95 ≤ 50 ms（仅 1-2 个 SKILL.md 读取）
- cycle 写 case p95 ≤ 100 ms（单文件 markdown 写）
- cycle 总耗时较基线增量 ≤ 5%

**Rationale**：
- 双层架构下 middleware 只读 2 个 SKILL.md（own + trading-knowledge），比之前估算的 50 个 pattern 文件快太多
- 写 case 是 sequential I/O，目标宽松

---

## R3：命名消歧（双层架构下大幅简化）

**Decision**：
- 5 个 skill name 全局唯一（kebab-case）：`tech-analysis` / `chain-analysis` / `news-analysis` / `macro-analysis` / `trading-knowledge`
- `load_skill(name)` 仅接 5 个之一，无歧义
- pattern names 在 SKILL.md body 文本里以 `<name>` 形式（snake_case）；verdict reasoning 引用 `applied: <name>`
- 跨 agent 引用 pattern 时显式 `applied: <agent>::<name>` 形式
- reflection 解析 `applied:` 时短形式按调用 agent 自身查找；歧义时报 warning 跳过

**Rationale**：
- 双层架构下 patterns 不再是 skills，所以 `load_skill` 不需要 `agent::pattern` 嵌套
- pattern 引用用 reasoning 文本约定即可

---

## R4：文件锁（简化）

**Decision**：
- 进程内 `threading.Lock` 即可——本期 reflection / curation / cycle 写 case 都在同进程
- 写文件用 temp + `os.rename` 保证原子写
- 跨进程锁 fcntl.flock 留 follow-up（curation 走独立 cron 时再加）

**Rationale**：当前 scope 单进程，threading 锁开销小；过早跨进程锁是 over-engineering

---

## R5：启动时间基线

**Decision**：实施前测 5 次中位数（clone → docker compose up → arena migrate → arena run BTC/USDT 完成）；实现完成后再测对比。

**Predicted impact**：
- 新增 5 个 SKILL.md ≤ 50KB → git clone 几乎无差
- `agent_memory/` 是 gitignored 空目录 → 无 clone 影响
- 无新依赖 → uv sync 不变
- 综合：< 2s 增量，远小于噪声

---

## R6：load_skill 双接口

**Decision**：
- LangChain `BaseTool` + 普通 Python 函数双导出（同实现）
- 4 个 agent 节点通过 middleware 自动获得 tool 调用能力
- verdict / curation 直接调函数

**Rationale**：双接口共享实现 = 行为一致 + 单点测试

---

## R7：Curation 的 LLM 整理流程（plan 阶段新增）

**Decision**：
- `arena skills curate <name>` 默认手工模式：列出 active patterns 让用户人工编辑 SKILL.md
- `arena skills curate <name> --llm` 调 LLM 整理：
  - 输入：当前 SKILL.md + active patterns + recent cases summary
  - 输出：`agent_skills/<skill>/SKILL.md.draft`
  - **不 overwrite** 原文件——用户 diff + merge
- frontmatter `manually_edited: true` 时跳过整个文件
- 本期实现 CLI 框架 + 简单 prompt；优化迭代留 follow-up

**Rationale**：
- 用户保留控制权（spec User Story 3 强调"也可以自定义"）
- LLM 输出不直接覆盖避免破坏手工编辑
- 简化第一版实现

---

## R8：Cases 文件格式

**Decision**：
- 命名：`agent_memory/<agent>/cases/<YYYY-MM-DD>-cycle-<commit_hash[:8]>.md`
  - 日期前缀方便手工归档（如 `tar czf 2026-Q1.tar.gz agent_memory/tech/cases/2026-{01,02,03}-*.md`）
- frontmatter（YAML）：
  ```yaml
  ---
  cycle_id: <full hash or trace_id>
  timestamp: 2026-05-06T14:29:54Z
  pair: ETH/USDT:USDT
  agent: tech
  verdict_action: short
  applied_patterns: [funding_squeeze_long, macd_divergence_short]
  risk_gate_passed: true
  execution_status: {succeeded: true, stage: filled}
  final_pnl: -7.68    # 平仓后回填；null 未平仓
  ---
  ```
- body（markdown）：
  ```markdown
  ## Snapshot Summary
  ```yaml
  funding_rate: -0.0002
  price: 2364.79
  rsi: 46.5
  ...
  ```

  ## Agent Analysis
  {tech_agent.analysis_text}

  ## Verdict Reasoning
  {verdict.reasoning}
  ```

**Rationale**：
- frontmatter 结构化便于 reflection 程序读取
- body 保留人类可读的原始文本便于复盘
- 平仓后回填 `final_pnl` 是必要的——cycle 结束时 PnL 通常未实现

---

## R9：agent_memory 的 retention 策略（spec 留 follow-up）

**Decision (本期)**：不做自动 retention。所有数据永久保留。

**Future**：可加 `arena memory archive --before YYYY-MM-DD` CLI 命令把老 cases 压缩到 `archive/<year>-<quarter>.tar.gz`。

**Rationale**：
- 用户要"尽可能保存多的数据用于分析"
- 本地 SSD 容量足够（~50 cycles/天 × 365 × 4 agents × 5KB/case ≈ 365MB / 年）
- retention 策略需要 user-specific 业务判断，不应硬编码

---

## R10：trading-knowledge SKILL.md 的内容来源（plan 阶段新增）

**Decision**：本期纯**手工写**——
- 内容包括：funding_rate 含义（替代 base.py:35-37 硬编码）、regime_definitions（与 learning/regime.py 阈值同步）、trading_pair_semantics（spot vs perp 区别等）
- 不从 cases / patterns 蒸馏（这些是领域常识，非可学习经验）
- 4 agent 在 prompt 中都看到 trading-knowledge body

**Rationale**：领域常识是稳定知识，无需自动化整理

---

## 决策对 spec / plan / implementation 影响摘要

| Research | Spec 影响 | Plan 影响 | Implementation 影响 |
|---|---|---|---|
| R1 双流程 | ✅ FR-008/FR-016 | ✅ nodes/reflection.py + arena skills curate CLI | ✅ 两个 entry point |
| R2 性能 SC | 否 | ✅ p95 < 50ms | ✅ 加 microbench |
| R3 命名 | 否（FR-026 已覆盖）| ✅ 双层下简化 | ✅ load_skill 单参数 |
| R4 锁 | 否 | ✅ threading.Lock | ✅ 进程内即可 |
| R5 启动 | 否 | ✅ 测前 5 次 | ✅ 实施前后对比 |
| R6 双接口 | 否 | ✅ 同实现双导出 | ✅ tool.py |
| R7 Curation | 否（FR-014/FR-016/FR-017）| ✅ CLI 设计 | ✅ learning/curation.py |
| R8 Cases 格式 | ✅ FR-006 | ✅ schema | ✅ journal node 写入 |
| R9 Retention | ✅ Out of Scope | 否 | 否 |
| R10 Knowledge | ✅ FR-004（trading-knowledge）| ✅ 初始内容来源 | ✅ 手工写 SKILL.md |

**所有 spec NEEDS CLARIFICATION 已 0**：confirmed by re-grep `specs/014-*/spec.md`，无 `[NEEDS CLARIFICATION]` 标记。
