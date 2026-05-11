# Implementation Plan: Spec 021 — Pattern Cold-Start

**Branch**: `024-pattern-cold-start` | **Date**: 2026-05-11 | **Spec**: [spec.md](spec.md)

## Summary

trilogy 进化数据链 cold-start gap 补完：

1. `learning/memory.py:distill_patterns()` 加 "from cases → create new PatternRecord" 路径
2. `config.py:ExperienceConfig` + `default.toml [experience]` 加 `min_cases_per_pattern = 5` 字段
3. `ops/daemon.py` 加第 4 个 action `_action_pattern_extraction()`
4. `default.toml [evolution_daemon].actions` 默认列表加 `"pattern_extraction"`
5. `src/cli/main.py` 加 `arena experience distill` typer command

技术路径：纯算法 + 文件 IO，无 LLM 调用，无新 runtime 依赖。复用 spec 014 既有 `_parse_applied_from_body` / `_save_pattern` / `PatternRecord` schema。单 PR 4 commit。

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: 复用 spec 014 既有（pathlib / dataclasses / typer）；无新 dependency
**Storage**: 文件系统（`agent_memory/<agent>/patterns/<slug>.md`），无 schema 变更
**Testing**: pytest 8.x + pytest-asyncio
**Target Platform**: Linux server (生产) / macOS (dev)
**Project Type**: Backend service（CLI + daemon 加 action）
**Performance Goals**:
- distill_patterns cold-start 200 cases ≤ 2s（纯文本 parse + 字典聚合 + 文件写入）
- daemon `_action_pattern_extraction` ≤ 5s（与 pareto/regime 量级一致）
- CLI command ≤ 5s
**Constraints**:
- 不引入新 runtime 依赖
- 不破坏 spec 014/15/17a/17b/18/19/20a/20b/20c 公开 API
- 单 PR ≤ 4 commit
- 落地 ~3-5 天
**Scale/Scope**:
- 后端：~150 LOC 新增 + ~30 LOC 修改
- CLI：~30 LOC 新增 typer command
- 测试：~200 LOC 单测 + e2e

## Constitution Check

`.specify/memory/constitution.md` 不存在，跳过。CLAUDE.md 对齐：
- ✓ Markdown 简体中文
- ✓ 直接删旧不留 fallback
- ✓ 不破坏 spec 014+ 公开 API
- ✓ 不引入新 runtime 依赖

**Gate**: PASS

## Project Structure

```text
src/cryptotrader/
├── learning/
│   └── memory.py          # MODIFY: distill_patterns 加 cold-start 路径 + _create_pattern_from_cases helper
├── ops/
│   └── daemon.py          # MODIFY: 加 _action_pattern_extraction + dispatch 分支
└── config.py              # MODIFY: ExperienceConfig 加 min_cases_per_pattern

src/cli/
└── main.py                # MODIFY: 加 arena experience distill typer command

config/
└── default.toml           # MODIFY: [experience] min_cases_per_pattern=5 + [evolution_daemon].actions 加 pattern_extraction

tests/
├── test_distill_patterns_cold_start.py    # NEW: 5 单测
├── test_pattern_slug_generation.py        # NEW: slug 规则单测
├── test_daemon_pattern_extraction.py      # NEW: daemon action 单测
├── test_cli_experience_distill.py         # NEW: CLI command 单测
└── test_e2e_pattern_cold_start.py         # NEW: 端到端 fixture cases → patterns → API
```

**Structure Decision**：纯 surgical 改动既有文件 + 5 个新测试文件；无新源码模块（slug helper 内置在 memory.py 私有函数）。

## 实施约束

- C1 commit：distill_patterns cold-start + config（FR-P1~P7，纯新增路径）
- C2 commit：daemon `pattern_extraction` action + CLI command（FR-P8~P13）
- C3 commit：单测 + e2e
- C4 commit：final gate（跑 SC 验证 + ruff + dashboard smoke）

落地后用户验证：
- `uv run arena experience distill --cycles-window 200` 创建首批 patterns
- dashboard `/memory` 页 RulesGrid 显示 patterns 数据
- `arena evolution-daemon --once` 4 actions 全 PASS
