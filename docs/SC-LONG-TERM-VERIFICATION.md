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
1. Evolution daemon 是否每日运行（`pattern_extraction` step；搜 `agent_memory/cases/`
   是否在增长）。in-cycle `nodes/reflection.py` 触发路径已于 2026-05-13 删除——
   蒸馏现在只走 daemon 通道。
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

## SC-010 — distill_patterns 失败不阻塞 daily daemon（2026-05-13 后）

**历史**：原 SC-010 验证 in-cycle `nodes/reflection.py` 失败不阻塞 trading
cycle。该节点 2026-05-13 删除——蒸馏迁出到独立 Evolution Daemon 进程
（`ops/daemon.py`），与 trading-cycle hot path 隔离，不再有"reflection 失败
阻塞 cycle"风险。

**当前等价问题**：`distill_patterns` 在 daemon 内失败是否触发 next-day
fallback / soft-degrade？

**观察方法**：

```bash
# Evolution daemon 日志（spec 020b）
grep -E "distill_patterns.*FAIL|reflect daemon.*Exception|pattern_extraction.*SKIP" \
     /tmp/evolution-daemon.log | tail -20

# Prometheus daemon 指标
curl -s http://localhost:8003/metrics | grep -E \
     "evolution_daemon_(run_count|llm_failure_rate)"
```

**告警阈值**：daemon `pattern_extraction` 连续 2 天 SKIP / FAIL → 检查
`learning/memory.py:distill_patterns` 异常路径和 daemon `soft_degrade` 处理
（spec 020b FR-D10）。

---

## 操作建议

- 14 天后由值班运维者跑一遍上述 4 个 check（约 10 分钟）
- 任一指标 fail → 记录到 `specs/014-agent-skills-protocol-migration/POST-DEPLOY-NOTES.md`
- 连续 2 周 fail → 触发 retro，spec 可能需要 follow-up 改动

---

*Last reviewed: 2026-05-06 — see [spec.md Success Criteria](../specs/014-agent-skills-protocol-migration/spec.md#measurable-outcomes) for definition source of truth.*
