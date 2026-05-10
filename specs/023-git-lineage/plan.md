# Implementation Plan: Spec 020c — Git Lineage

**Branch**: `023-git-lineage` | **Date**: 2026-05-09 | **Spec**: [spec.md](spec.md)

## Summary

trilogy 终段。落地 D-ENG-02 git lineage：

1. `src/cryptotrader/ops/lineage.py`：`GitLineageHook` 类（subprocess git CLI，不引入 gitpython）
2. daemon `run_once()` 末尾调 `commit_changes(actions_summary)`
3. spec 018 fsm.py 4 transitions 调用方 batch 触发 lineage
4. commit author=current git user + trailer `Auto-Generated-By: spec-020c` + 独立 `evolution` branch（orphan 创建）
5. lineage 失败 soft fail（改动保留 + OTel error）
6. 3 P2 advisory 收尾：daemon `_try_acquire_locks` 改 async + `run_forever()` SIGTERM handler + SkillsGrid badge aria-label
7. 2 lineage Prometheus Gauge

技术路径：复用 spec 020a/020b observability + daemon 基建；subprocess git CLI 不引入新依赖。单 PR 4 commit。

## Technical Context

**Language/Version**: Python 3.12+ (后端) / TypeScript 5.9 (前端)
**Primary Dependencies**: subprocess（stdlib）/ signal（stdlib）/ APScheduler / OpenTelemetry SDK / prometheus-client / React 19 / Vitest（全部已存在）
**Storage**: 不涉及 schema 变更；evolution branch git log = audit storage
**Testing**: pytest 8.x + pytest-asyncio + Vitest
**Target Platform**: Linux server / macOS dev
**Project Type**: Backend service + frontend a11y
**Performance Goals**:
- `commit_changes()` ≤ 500ms（subprocess git overhead）
- daemon SIGTERM graceful shutdown ≤ 30s（等当前 run_once 完成）
- a11y 不引入额外 render 成本
**Constraints**:
- 不引入新 runtime 依赖
- 不破坏 spec 014/15/17a/17b/18/19/20a/20b 公开 API
- 单 PR ≤ 4 commit
- 落地 ~3-5 天
**Scale/Scope**:
- 后端：~250 LOC 新增（lineage.py + 2 aggregator）+ ~80 LOC 修改（daemon 3 处 + fsm 调用方）
- 前端：~30 LOC 修改（SkillsGrid 3 处 aria-label）
- 测试：~250 LOC 新增

## Constitution Check

`.specify/memory/constitution.md` 不存在，跳过。CLAUDE.md 对齐 ✓。**Gate**: PASS

## Project Structure

```text
src/cryptotrader/
├── ops/
│   ├── daemon.py                       # MODIFY: run_once 加 lineage / _try_acquire_locks 改 async / run_forever 加 SIGTERM
│   └── lineage.py                      # NEW: GitLineageHook
├── observability/
│   └── daemon_metrics.py               # MODIFY: 加 2 lineage aggregator
└── learning/evolution/
    └── fsm.py                          # MODIFY: transition 调用方 batch lineage hook（仅 transition 收集点；不重写 fsm 算法）

src/api/routes/
└── metrics.py                           # MODIFY: 注册 2 lineage Gauge

web/src/pages/memory/components/
├── SkillsGrid.tsx                      # MODIFY: 3 类 badge 加 aria-label
└── SkillsGrid.test.tsx                 # MODIFY: 加 a11y 断言

tests/
├── test_lineage.py                     # NEW: GitLineageHook unit
├── test_daemon_lineage_integration.py  # NEW: daemon + lineage integration
├── test_daemon_signal_handler.py       # NEW: SIGTERM graceful shutdown
└── test_e2e_git_lineage.py             # NEW: 端到端 evolution branch
```

**Structure Decision**：复用既有 src/cryptotrader/ops/ + observability/ 子包；新增 lineage.py 单文件模块；所有改动 surgical。

## 实施约束

- C1：lineage 模块 + observability aggregator + Prometheus Gauge（纯新增）
- C2：daemon 集成 + fsm 调用方 batch + 3 P2 修复（asyncio.sleep / SIGTERM）
- C3：前端 a11y aria-label
- C4：E2E + final gate

落地后用户验证：
- `arena evolution-daemon --once` → `git log evolution -1` 含 trailer
- `kill -TERM <daemon-pid>` → ≤ 30s graceful exit
- `/memory` 页 SkillsGrid badges 含 aria-label
