# Spec 014 Long-Term Success Criteria Verification

部分 Success Criteria 是 14 天后线上观察指标，无法在合并 PR 时验证。本文档记录每条
SC 的观察方法 + 查询命令，便于运维者周期性 spot-check。

## SC-003 — cases 累计不丢数据

**Spec 要求**：完成 ≥ 14 天运行后，`agent_memory/cases/` 累计 cycles 数 ≥ trading
cycle 实际执行次数。

**观察方法**：

```bash
# (a) 实际 cycle 执行次数（来自 journal store）
uv run psql "$DATABASE_URL" -c \
  "SELECT count(*) FROM decision_commits WHERE created_at > NOW() - INTERVAL '14 days';"

# (b) 写入 cases 文件数
ls agent_memory/cases/*.md | wc -l

# 期望：(b) ≥ (a)
```

**告警阈值**：丢失率 > 1% → 检查 `nodes/journal.py:write_case` 异常日志（FR-007 写
失败 logger.warning，不阻塞）。

---

## SC-004 — ≥1 active pattern 自动产生

**Spec 要求**：完成 ≥ 14 天 reflection 运行后，`agent_memory/<agent>/patterns/` 自
动产生 ≥ 1 条 maturity=active 的 pattern。

**观察方法**：

```bash
# 任一 agent 含 active pattern 则通过
for agent in tech chain news macro; do
  count=$(grep -lE "^maturity: active$" agent_memory/$agent/patterns/*.md 2>/dev/null | wc -l)
  echo "$agent: $count active patterns"
done
```

**告警阈值**：14 天后所有 agent 均为 0 → 检查：
1. reflection 节点是否被 graph 触发（搜 `agent_memory/cases/` 是否在增长）
2. PnL 数据是否正常回填（FR-027 `update_pattern_pnl`）
3. 4 层防过拟合是否过严（L2 默认 N=5；L3 默认 min_delta=0.1）

---

## SC-006 — applied: 引用率 ≥ 60%

**Spec 要求**：完成 ≥ 14 天后 verdict reasoning 中 `applied:` 引用率 ≥ 60%（在有
active pattern 时）。

**观察方法**：

```bash
# 提取最近 14 天 verdict reasoning，统计 applied: 出现频率
uv run psql "$DATABASE_URL" -c "
  SELECT
    count(*) FILTER (WHERE verdict_reasoning ~ 'applied:') AS with_applied,
    count(*) AS total,
    round(100.0 * count(*) FILTER (WHERE verdict_reasoning ~ 'applied:') / NULLIF(count(*), 0), 1) AS pct
  FROM decision_commits
  WHERE created_at > NOW() - INTERVAL '14 days';
"
```

**告警阈值**：< 50% → 检查：
1. `nodes/verdict.py` prompt 是否仍含 `applied: <pattern>` 强制要求（FR-026）
2. SKILL.md body 中是否列出 active pattern 名（curation 是否过期）
3. `parse_applied()` 解析逻辑是否丢弃合法引用（看 logger.warning）

---

## SC-010 — reflection 失败不阻塞 cycle

**Spec 要求**：reflection job 失败注入测试中 trading cycle 完成率 100%。

**观察方法**：

### A. 单元测试 fault-injection（在 PR 合并时验证）

`tests/test_reflection_pattern_distill.py` 中含一项测试：

```python
def test_distill_failure_does_not_block_cycle(monkeypatch):
    """FR-012 + SC-010: reflection 内部异常不能阻塞下一个 cycle。"""
    monkeypatch.setattr(
        "cryptotrader.learning.memory.distill_patterns",
        Mock(side_effect=RuntimeError("simulated reflection failure")),
    )
    # 触发 reflection 节点
    state = run_reflection_node(initial_state)
    # 关键：state 仍可继续被下个 cycle 消费
    assert state["next_action"] is not None  # cycle 主流程未中断
```

> 该测试已包含在 spec 014 PR 中，由 Phase 6 review-code 阶段验收。

### B. 线上观察（合并后 14 天）

```bash
# 查 reflection 节点日志（应见 ERROR 但 cycle 继续完成）
grep -E "reflection.*failed|distill_patterns.*Exception" logs/cryptotrader.log | wc -l   # 失败次数
grep -E "trading cycle complete" logs/cryptotrader.log | wc -l                          # 完整 cycle 数

# 期望：failure_count > 0 时，cycle_count 仍按预期递增（每 5min 一个）
```

**告警阈值**：reflection 失败后下一 cycle 缺失 → 检查 `nodes/reflection.py` 是否捕
获了所有异常并返回 unchanged state（FR-012）。

---

## 操作建议

- 14 天后由值班运维者跑一遍上述 4 个 check（约 10 分钟）
- 任一指标 fail → 记录到 `specs/014-agent-skills-protocol-migration/POST-DEPLOY-NOTES.md`
- 连续 2 周 fail → 触发 retro，spec 可能需要 follow-up 改动

---

*Last reviewed: 2026-05-06 — see [spec.md Success Criteria](../specs/014-agent-skills-protocol-migration/spec.md#measurable-outcomes) for definition source of truth.*
