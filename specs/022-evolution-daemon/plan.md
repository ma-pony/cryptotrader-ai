# Implementation Plan: Spec 020b — Evolution Daemon

**Branch**: `022-evolution-daemon` | **Date**: 2026-05-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/022-evolution-daemon/spec.md`

## Summary

trilogy 收尾 ops 子域第 2 段。落地 spec 016 D-ENG-01 reflect daemon：

1. `src/cryptotrader/ops/daemon.py`：`EvolutionDaemon` 类（`run_once` + `run_forever`）
2. `arena evolution-daemon` CLI 命令（`--once` dry-run / 默认 forever cron 模式）
3. 3 reflect actions：Pareto rerank（FR-D6）/ Regime filter（FR-D7）/ Skill proposal auto-trigger per-agent（FR-D8）
4. Soft degrade（LLM 失败 → SKIP 当前 step，继续下一个）+ 文件 lock（fcntl.flock 字母顺序）
5. `[evolution_daemon]` TOML config + env override
6. 3 个新 Prometheus Gauge + 复用 spec 020a observability aggregator pattern
7. 独立 docker-compose `evolution-daemon` service

技术路径：复用既有基建（spec 010 OTel / spec 015 metrics / spec 018 pareto+regime / spec 019 propose_new_skill / spec 020a aggregator）；单 PR 4 commit（C1 文档 / C2 算法 / C3 docker+monitoring / C4 E2E）。

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: APScheduler / fcntl / OpenTelemetry SDK / prometheus-client（全部已存在）
**Storage**: 不涉及（无 schema 变更；复用 spec 014/18 既有 cases/patterns 文件路径）
**Testing**: pytest 8.x + pytest-asyncio
**Target Platform**: Linux server（生产 docker-compose）/ macOS（dev `arena --once`）
**Project Type**: Backend service（独立 docker-compose service）
**Performance Goals**:
- daemon `run_once()` ≤ 30s（mocked LLM）/ ≤ 120s（生产含 LLM）
- 3 reflect actions 顺序跑（不并行；无 race condition 风险）
- fcntl.flock 5s timeout（跨进程并发安全）
**Constraints**:
- 不引入新 runtime 依赖
- 不破坏 spec 014/15/17a/17b/18/19/20a 公开 API
- 单 PR ≤ 4 commit
- 落地 ~1 周
**Scale/Scope**:
- 后端：~400 LOC 新增（daemon.py + 3 aggregator + thin wrapper + CLI）
- 配置：~10 LOC（TOML section + EvolutionDaemonConfig dataclass）
- Docker：~25 LOC（docker-compose.yml service）
- 测试：~400 LOC（unit + e2e）

## Constitution Check

`.specify/memory/constitution.md` 不存在，跳过。

与 CLAUDE.md 对齐：
- ✓ Markdown 简体中文
- ✓ 直接删旧不留 fallback
- ✓ 不破坏既有 API
- ✓ 不引入新 runtime 依赖

**Gate**: PASS

## Project Structure

### Documentation (this feature)

```text
specs/022-evolution-daemon/
├── plan.md              # This file
├── spec.md              # 已生成
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── REVIEW-SPEC.md       # 已生成
├── tasks.md             # Phase 2 output（/speckit.tasks 生成）
└── checklists/
    └── requirements.md  # 已生成
```

### Source Code (repository root)

```text
src/cryptotrader/
├── ops/                                # NEW 子包
│   ├── __init__.py
│   └── daemon.py                       # EvolutionDaemon 类
├── observability/
│   └── daemon_metrics.py               # NEW: 3 sliding window aggregator
├── learning/
│   └── memory.py                       # MODIFY: 加 thin public wrapper
└── config.py                           # MODIFY: 加 EvolutionDaemonConfig dataclass

src/api/routes/
└── metrics.py                           # MODIFY: 注册 3 Prometheus Gauge

src/cli/
└── main.py                              # MODIFY: 加 arena evolution-daemon 命令

config/
└── default.toml                         # MODIFY: 加 [evolution_daemon] section

docker-compose.yml                       # MODIFY: 加 evolution-daemon service

tests/
├── test_evolution_daemon.py             # NEW: EvolutionDaemon unit tests
├── test_daemon_metrics.py               # NEW: 3 aggregator
├── test_e2e_evolution_daemon.py         # NEW: mocked cycle E2E
└── test_evolution_daemon_cli.py         # NEW: CLI 入口测试
```

**Structure Decision**：复用既有 src/cryptotrader 包 + spec 020a observability 子包；新增 `ops/` 子包专门收 daemon 代码；docker-compose 既有 7 service 模式扩展加 1 service。

## Complexity Tracking

无 Constitution 违规，无需填写。

## 实施约束 + 边界

- **C1 commit**：基础设施 + config（daemon 骨架 + CLI + TOML），纯新增，无 behavior 变化
- **C2 commit**：3 reflect actions + soft degrade + 文件 lock + 单测
- **C3 commit**：observability + docker-compose service + 单测
- **C4 commit**：E2E 测试 + 最终 gate

落地后用户验证：
- `uv run arena evolution-daemon --once` 单次 dry-run（mocked LLM）成功 exit 0 + 3 actions 全 PASS
- `docker compose up evolution-daemon` 单独启动成功
- `curl /metrics | grep evolution_daemon` 返回 3 Gauge
