# Phase 0：研究与决策

**关联 spec**：[spec.md](spec.md)
**关联 brainstorm**：[brainstorm/09-spec-021-pattern-cold-start.md](../../brainstorm/09-spec-021-pattern-cold-start.md)
**Date**: 2026-05-11

## 概述

5 项 brainstorm 决策 + 3 项 clarify + 5 项 spot-check 已消除全部 ambiguity。本文档记录实施细节。

## 5 项关键决策（来自 brainstorm）

| # | 决策 | Rationale |
|---|---|---|
| Q1 | distill_patterns 内置 cold-start | 单文件 surgical，命名语义吻合 |
| Q2 | daemon daily + CLI manual | 与 spec 020b 模式一致 + dev/debug 紧急触发 |
| Q3 | min_cases_per_pattern=5 配置化 | 平衡 patterns 集增长 vs 噪声 |
| Q4 | maturity="observed" 初始 | spec 014 FSM 默认 |
| Q5 | A 完整 spec ship | 跨 4 模块需要标准切分 |

## 3 项 clarify 决策

| # | Question | Answer |
|---|---|---|
| C1 | pattern slug 规则 | lowercase + 替换非 alnum 为 `-` + 截断 ≤ 60 字符 + 去除前后 `-`；collision `-N` 后缀 |
| C2 | pnl_track 空值处理 | 跳过 None；全 None 时空 PnLTrack 但仍创建 pattern |
| C3 | regime_tags top 3 投票 | 频次降序前 3，并列字母序兜底 |

## 5 项 spot-check 结果

| # | 检查项 | 结果 |
|---|---|---|
| 1 | `_parse_applied_from_body` + `VALID_AGENT_IDS` 既有 | ✓ memory.py:476 + agents/skills/_constants.py |
| 2 | `PatternRecord` schema 完整 | ✓ name/agent/description/body/regime_tags/pnl_track/maturity/source_cycles 全 |
| 3 | `ExperienceConfig.min_cases_per_pattern` | ⚠ 不存在，FR-P6 新增 |
| 4 | daemon `actions` list-driven | ✓ ops/daemon.py:110 |
| 5 | CLI `arena experience` 子命令 | ⚠ 未实现，FR-P12 新建 |

## 实施细节

### Decision 1：cold-start 路径在 distill_patterns 内的位置

```python
# src/cryptotrader/learning/memory.py:distill_patterns()
def distill_patterns(...):
    cases = _read_cases(...)
    run.cases_processed = len(cases)
    if not cases:
        return run

    # 既有逻辑：统计 applied_pattern 频次 + PnL
    agent_pattern_counts = ...  # (existing)

    # spec 021 NEW: cold-start — 从 frequent applied_patterns 创建新 PatternRecord
    cfg = load_config()
    threshold = cfg.experience.min_cases_per_pattern  # default 5
    for agent, pattern_pnls_map in agent_pattern_counts.items():
        patterns_dir = mem / agent / "patterns"
        patterns_dir.mkdir(parents=True, exist_ok=True)
        for applied_text, case_data_list in pattern_pnls_map.items():
            if len(case_data_list) < threshold:
                continue
            slug = _make_pattern_slug(applied_text, existing_dir=patterns_dir)
            target_path = patterns_dir / f"{slug}.md"
            if target_path.exists():
                continue  # collision-after-slug—patterns 同名已存在，跳过
            pattern = _create_pattern_from_cases(
                slug=slug,
                agent=agent,
                applied_text=applied_text,
                case_data_list=case_data_list,
            )
            try:
                _save_pattern(pattern, target_path)
                run.patterns_created += 1
            except Exception:
                logger.warning("cold-start: failed to save pattern %s", slug, exc_info=True)

    # 既有逻辑：对 patterns_dir 中已有 patterns 跑 maturity FSM 更新
    for agent in VALID_AGENT_IDS:
        ...  # (existing, unchanged)
```

**Rationale**：
- cold-start 路径在 case 解析后 + 既有 maturity 更新前
- 复用 `_parse_applied_from_body` 得到的 `agent_pattern_counts` 数据
- 失败 isolated（单 pattern 失败不影响其他）

### Decision 2：`_make_pattern_slug` helper

```python
def _make_pattern_slug(applied_text: str, existing_dir: Path) -> str:
    """生成 filesystem-safe slug。规则：lowercase + 非 alnum 替换 - + 截断 60 + collision 加 -N。"""
    import re
    base = re.sub(r"[^a-z0-9]+", "-", applied_text.lower()).strip("-")[:60]
    if not base:
        base = "unnamed"
    if not (existing_dir / f"{base}.md").exists():
        return base
    # collision: 加 -2 / -3 / ...
    for n in range(2, 1000):
        candidate = f"{base}-{n}"
        if not (existing_dir / f"{candidate}.md").exists():
            return candidate
    raise ValueError(f"slug collision space exhausted for: {applied_text}")
```

### Decision 3：`_create_pattern_from_cases` helper

```python
def _create_pattern_from_cases(
    slug: str,
    agent: str,
    applied_text: str,
    case_data_list: list[dict],  # [{cycle_id, pnl, regime_tags}]
) -> PatternRecord:
    from collections import Counter
    pnls = [float(c["pnl"]) for c in case_data_list if c.get("pnl") is not None]
    source_cycles = [c["cycle_id"] for c in case_data_list][:5]
    # regime tags 频次 top 3
    tag_counter: Counter[str] = Counter()
    for c in case_data_list:
        for t in (c.get("regime_tags") or []):
            tag_counter[t] += 1
    # 频次降序，并列字母序
    sorted_tags = sorted(tag_counter.items(), key=lambda x: (-x[1], x[0]))
    regime_tags = [t for t, _ in sorted_tags[:3]]
    body = f"# {applied_text}\n\nAuto-distilled from {len(case_data_list)} cases.\n\nSource cycles (first 5): {source_cycles}\n"
    return PatternRecord(
        name=slug,
        agent=agent,
        description=f"Auto-distilled pattern: {applied_text} (from {len(case_data_list)} cases)",
        body=body,
        pnl_track=PnLTrack(pnls=pnls),
        regime_tags=regime_tags,
        maturity="observed",
        source_cycles=source_cycles,
    )
```

**Rationale**：纯函数 + 无 LLM 调用 + 完整字段填充

### Decision 4：Daemon `_action_pattern_extraction` 实现

```python
# src/cryptotrader/ops/daemon.py
async def _action_pattern_extraction(self) -> ActionResult:
    from cryptotrader.learning.memory import distill_patterns
    from cryptotrader.config import load_config
    cfg = load_config()
    run = distill_patterns(cycles_window=cfg.experience.lookback_commits)
    return ActionResult(
        name="pattern_extraction",
        status="PASS",
        duration_ms=...,
        details={
            "new_count": run.patterns_created,
            "updated_count": run.patterns_updated,
            "archived_count": run.patterns_archived,
            "cases_processed": run.cases_processed,
        },
    )
```

Dispatch：
```python
async def _run_action(self, name: str) -> ActionResult:
    try:
        if name == "pareto":     return await self._action_pareto()
        elif name == "regime":   return await self._action_regime()
        elif name == "skill_proposal": return await self._action_skill_proposal()
        elif name == "pattern_extraction": return await self._action_pattern_extraction()  # NEW
    except (OpenAIAPIError, TimeoutError, NetworkError) as e:
        ...  # existing soft degrade
```

### Decision 5：CLI `arena experience distill` 实现

```python
# src/cli/main.py
experience_app = typer.Typer()

@experience_app.command("distill")
def experience_distill(
    memory_dir: str = typer.Option("agent_memory", "--memory-dir"),
    cycles_window: int = typer.Option(None, "--cycles-window"),
) -> None:
    """从 cases 蒸馏 patterns（spec 021 cold-start 入口）。"""
    from pathlib import Path
    from cryptotrader.learning.memory import distill_patterns
    from cryptotrader.config import load_config

    cfg = load_config()
    window = cycles_window if cycles_window else cfg.experience.lookback_commits
    try:
        run = distill_patterns(memory_dir=Path(memory_dir), cycles_window=window)
        typer.echo(f"cases_processed: {run.cases_processed}")
        typer.echo(f"patterns_created: {run.patterns_created}")
        typer.echo(f"patterns_updated: {run.patterns_updated}")
        typer.echo(f"patterns_archived: {run.patterns_archived}")
        if run.error:
            typer.echo(f"error: {run.error}")
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1) from e

app.add_typer(experience_app, name="experience")
```

## Phase 0 检查项

- [x] 所有 NEEDS CLARIFICATION 已解决
- [x] 所有 dependency 已识别 best practice
- [x] 所有 integration 已找到 pattern

Phase 0 输出完成。
