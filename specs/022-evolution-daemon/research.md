# Phase 0：研究与决策

**关联 spec**：[spec.md](spec.md)
**关联 brainstorm**：[brainstorm/07-spec-020b-evolution-daemon.md](../../brainstorm/07-spec-020b-evolution-daemon.md)
**Date**: 2026-05-09

## 概述

7 项关键决策已在 brainstorm 阶段完成，3 项 ambiguity 已在 clarify 阶段解决，4 项 spot-check 已修订相关 FR 锚点。本文档记录最终决定 + 实施细节研究。

## Technical Context 中无 NEEDS CLARIFICATION 项

Brainstorm 7 项决策 + 3 项 clarify + 4 项 spot-check 已消除全部 ambiguity。

## 7 项关键决策（来自 brainstorm）

| # | 决策 | 来源 |
|---|---|---|
| Q1 spec 拆分 | B 拆 020b（daemon）+ 020c（lineage + 020a P2） | trilogy 风格 / 风险隔离 |
| Q2 进程模型 | B 独立 docker-compose service | 故障隔离 / 与 spec 014/15 模式一致 |
| Q3 触发频率 | C 每天 cron `0 0 * * *` UTC | 批量算法适合 daily / 与 prompt cache 反向匹配 |
| Q4 reflect actions | C Pareto + Regime + Skill proposal auto-trigger | 不 auto-save 保留 human gate / 不做 cross-validation |
| Q5 failure mode | B Soft degrade + 仅 dashboard 可视 | 与 spec 020a Q5 一致 / 进化连续性 |
| Q6 monitoring | B 3 个核心指标（run_count / llm_failure_rate / draft_count） | 锚定 Q5/Q4 / 不告警 |
| Q7 config | A TOML + env override | 与 spec 014/15/20a 风格一致 |

## 3 项 clarify 决策（来自 spec.md Clarifications 段）

| # | Question | Answer |
|---|---|---|
| C1 | FR-D6 Pareto archive 判定 | 被任一 frontier rule 支配的 rules（非 frontier 成员）转 archived；frontier 成员保留 active |
| C2 | FR-D8 propose_threshold 范围 | per-agent 独立检查（4 agents 各独立），单次 daemon 可触发 0-4 propose_new_skill |
| C3 | FR-D12 lock 获取顺序 | 字母顺序（先 cases 再 patterns）防 deadlock |

## 4 项 spot-check 结果（2026-05-09）

| # | 检查项 | 结果与修订 |
|---|---|---|
| 1 | `pareto.py:rank_rules` 可独立调用 | ✓ src/cryptotrader/learning/evolution/pareto.py:54 |
| 2 | `propose_new_skill` 入口 | ✓ src/cryptotrader/learning/skill_proposal.py:202 |
| 3 | scheduler.py APScheduler 模式 | ✓ AsyncIOScheduler + CronTrigger 现成可借鉴 |
| 4 | docker-compose 多 service 模式 | ✓ 7 services（含 scheduler），加 evolution-daemon 惯例 |
| 5 | CLI 入口路径 | ⚠️ 修订：实际 src/cli/main.py（非 src/cryptotrader/cli/）；FR-D2 锚定该路径 |
| 6 | ops/ 子包 | ⚠️ 修订：不存在，FR-D1 需新建 |
| 7 | regime filter 函数位置 | ⚠️ 修订：在 src/cryptotrader/learning/memory.py:358（非 evolution 子包）；FR-D7 加 thin wrapper |

## 实施细节决策

### Decision 1：EvolutionDaemon 类骨架

**Decision**：FR-D1 改造方式：

```python
# src/cryptotrader/ops/daemon.py
import asyncio
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from opentelemetry import trace as _otel_trace

from cryptotrader.config import EvolutionDaemonConfig

_tracer = _otel_trace.get_tracer(__name__)


@dataclass
class ActionResult:
    name: str
    status: Literal["PASS", "SKIP", "FAIL"]
    duration_ms: int
    details: dict = field(default_factory=dict)


@dataclass
class RunResult:
    actions_run: list[ActionResult]
    total_duration_ms: int
    exit_code: int


class EvolutionDaemon:
    def __init__(self, config: EvolutionDaemonConfig) -> None:
        self.config = config
        self._scheduler: AsyncIOScheduler | None = None

    async def run_once(self) -> RunResult:
        with _tracer.start_as_current_span("evolution.daemon.run") as span:
            with self._acquire_locks():
                results: list[ActionResult] = []
                for action_name in self.config.actions:
                    result = await self._run_action(action_name)
                    results.append(result)
                exit_code = 0  # soft degrade: SKIP 不算 fail
                return RunResult(actions_run=results, total_duration_ms=..., exit_code=exit_code)

    async def run_forever(self) -> None:
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self.run_once,
            CronTrigger.from_crontab(self.config.cron, timezone="UTC"),
        )
        self._scheduler.start()
        # 阻塞直到 SIGTERM/SIGINT
        await asyncio.Event().wait()

    @contextmanager
    def _acquire_locks(self):
        # 字母顺序：cases → patterns
        lock_paths = sorted([
            Path("agent_memory/cases/.lock"),
            Path("agent_memory/patterns/.lock"),
        ])
        fds = []
        try:
            for lp in lock_paths:
                lp.parent.mkdir(parents=True, exist_ok=True)
                fd = lp.open("w")
                # 5s timeout via fcntl
                ...
                fds.append(fd)
            yield
        finally:
            for fd in fds:
                fcntl.flock(fd, fcntl.LOCK_UN)
                fd.close()
```

**Rationale**：
- 复用 spec 014 scheduler.py AsyncIOScheduler 模式
- `_acquire_locks` 用 contextmanager 保证释放
- soft degrade：SKIP 不影响 exit_code；只有 lock timeout 才整次跳过
- OTel span 父子关系：`evolution.daemon.run` → `evolution.daemon.<action>`

### Decision 2：3 reflect actions 实现

**Decision**：FR-D6/D7/D8 实现：

```python
async def _run_action(self, name: str) -> ActionResult:
    with _tracer.start_as_current_span(f"evolution.daemon.{name}") as span:
        try:
            if name == "pareto":
                return await self._action_pareto()
            elif name == "regime":
                return await self._action_regime()
            elif name == "skill_proposal":
                return await self._action_skill_proposal()
        except (OpenAIAPIError, TimeoutError, NetworkError) as e:
            span.record_exception(e)
            span.set_attribute("step.status", "SKIP")
            return ActionResult(name=name, status="SKIP", duration_ms=..., details={"reason": str(e)})

async def _action_pareto(self) -> ActionResult:
    from cryptotrader.learning.evolution.pareto import rank_rules
    from cryptotrader.learning.evolution._io import load_active_rules, save_pattern

    active = load_active_rules()  # 复用 spec 018 既有 IO
    ranked = rank_rules(active)
    frontier = {r.id for r in ranked if r.is_frontier}
    archived_count = 0
    for rule in active:
        if rule.id not in frontier:
            rule.maturity = "archived"
            save_pattern(rule)
            archived_count += 1
    return ActionResult(name="pareto", status="PASS", duration_ms=..., details={"archived_count": archived_count})

async def _action_regime(self) -> ActionResult:
    from cryptotrader.learning.memory import refilter_records_by_regime  # NEW thin wrapper
    changed_count = refilter_records_by_regime()
    return ActionResult(name="regime", status="PASS", duration_ms=..., details={"changed_count": changed_count})

async def _action_skill_proposal(self) -> ActionResult:
    from cryptotrader.learning.skill_proposal import propose_new_skill
    from cryptotrader.learning.evolution._io import load_active_rules_per_agent

    drafts_created = []
    for agent_id in ["tech", "chain", "news", "macro"]:
        rules = load_active_rules_per_agent(agent_id)
        if len(rules) >= self.config.propose_threshold:
            draft_path = await propose_new_skill(scope=f"agent:{agent_id}")
            drafts_created.append(draft_path)
    return ActionResult(name="skill_proposal", status="PASS", duration_ms=..., details={"drafts_created": drafts_created})
```

**Rationale**：
- 每个 action 独立 try/except → SKIP（soft degrade FR-D10）
- pareto archive 判定按 clarify Q1：非 frontier 成员 → archived
- skill_proposal 按 clarify Q2：4 agents 独立循环

### Decision 3：refilter_records_by_regime thin wrapper

**Decision**：FR-D7 改造方式：

```python
# src/cryptotrader/learning/memory.py（既有文件 modify）
def refilter_records_by_regime() -> int:
    """Public wrapper for daemon. Re-tags all cases by current regime; returns count of changed cases."""
    cases = _load_all_cases()
    current_snapshot = _build_current_snapshot()  # 复用 spec 018 既有 helper
    changed = 0
    for case in cases:
        old_tags = case.regime_tags
        new_tags = _filter_records_by_regime([case], current_snapshot)[0].regime_tags
        if old_tags != new_tags:
            case.regime_tags = new_tags
            _save_case(case)
            changed += 1
    return changed
```

**Rationale**：
- 私有 `_filter_records_by_regime` 接口不变（保护 spec 018 测试）
- 加 public wrapper 单一职责：批量 re-tag
- 返回 changed_count 给 ActionResult.details

### Decision 4：3 sliding window aggregator

**Decision**：FR-D13 改造方式（复用 spec 020a CacheMetricsAggregator 模式）：

```python
# src/cryptotrader/observability/daemon_metrics.py
from collections import deque
from threading import Lock
from time import time

class DaemonRunCountAggregator:
    """24h sliding window count of daemon runs."""
    def __init__(self) -> None:
        self._buffer: deque[float] = deque()
        self._lock = Lock()
        self._window = 86400  # 24h

    def record(self) -> None:
        with self._lock:
            now = time()
            self._buffer.append(now)
            self._evict_expired(now)

    def count(self) -> int:
        with self._lock:
            self._evict_expired(time())
            return len(self._buffer)

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._window
        while self._buffer and self._buffer[0] < cutoff:
            self._buffer.popleft()


class DaemonLLMFailureAggregator:
    """24h sliding window LLM failure rate."""
    def __init__(self) -> None:
        self._buffer: deque[tuple[float, bool]] = deque()
        self._lock = Lock()
        self._window = 86400

    def record(self, failed: bool) -> None:
        ...

    def failure_rate(self) -> float:
        with self._lock:
            self._evict_expired(time())
            if not self._buffer:
                return 0.0
            return sum(1 for _, f in self._buffer if f) / len(self._buffer)


class SkillProposalDraftAggregator:
    """7d sliding window total drafts created."""
    def __init__(self) -> None:
        self._buffer: deque[float] = deque()
        self._lock = Lock()
        self._window = 7 * 86400

    def record(self) -> None:
        ...

    def total(self) -> int:
        ...

# Module-level singletons (similar to spec 020a)
RUN_COUNT_AGG = DaemonRunCountAggregator()
LLM_FAILURE_AGG = DaemonLLMFailureAggregator()
SKILL_PROPOSAL_AGG = SkillProposalDraftAggregator()
```

**Rationale**：
- 与 spec 020a `CacheMetricsAggregator` 模式一致（deque + Lock + sliding window）
- 单实例 module-level（daemon 进程内访问）
- 跨 service 进程：daemon 写 + api/metrics 读 共享通过 OTel exporter 而非内存（这里是同进程内 mock 场景）

**注意**：本 spec daemon 是独立 docker-compose service，与 api service 不同进程。aggregator 是 **api service 进程内的状态**，daemon 通过 OTel span 暴露事件，api service 端 `/metrics` 通过 OTel collector 反向回调更新 aggregator？这条路径太复杂。

**简化方案**：daemon 把每次 run 的事件写入 redis（已存在 service）`evolution_daemon:events` list；api `/metrics` endpoint 读 redis list 计算指标。无需跨进程内存共享。

修订 FR-D13：sliding window aggregator 实际通过 redis sorted set 实现（key=`evolution_daemon:events`，score=timestamp，member=event_json）；api 端 `/metrics` lazy 计算时读 redis。

### Decision 5：docker-compose service 配置

**Decision**：FR-D15 改造方式：

```yaml
# docker-compose.yml（既有 modify）
services:
  evolution-daemon:
    build:
      context: .
      dockerfile: Dockerfile
    command: ["arena", "evolution-daemon"]
    environment:
      - EVOLUTION_DAEMON_ENABLED=${EVOLUTION_DAEMON_ENABLED:-true}
      - REDIS_URL=redis://redis:6379/0
      - OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4317}
    volumes:
      - ctdata:/data
      - ./agent_memory:/app/agent_memory
      - ./agent_skills:/app/agent_skills
      - ./config:/app/config:ro
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
```

**Rationale**：
- 复用既有 `ctdata` volume + `redis` / `postgres` service
- 独立资源限制（512M / 0.5 CPU）远低于 trading scheduler（1G / 1 CPU），符合 daily 1 trigger 的轻量级特征
- mount agent_memory + agent_skills 写入路径
- env override `EVOLUTION_DAEMON_ENABLED=false` 直接关闭 entrypoint

### Decision 6：CLI 入口

**Decision**：FR-D2 改造方式：

```python
# src/cli/main.py（既有 modify）
import asyncio
import os
import typer

evolution_app = typer.Typer()

@evolution_app.command("evolution-daemon")
def evolution_daemon(
    once: bool = typer.Option(False, "--once", help="Run once (dry-run) and exit"),
    config_path: str = typer.Option("config/default.toml", "--config"),
) -> None:
    """Run the evolution reflect daemon (Pareto / regime / skill proposal)."""
    if os.getenv("EVOLUTION_DAEMON_ENABLED", "true").lower() != "true":
        typer.echo("[evolution-daemon] disabled by EVOLUTION_DAEMON_ENABLED=false; exiting.")
        raise typer.Exit(0)

    from cryptotrader.config import load_config
    from cryptotrader.ops.daemon import EvolutionDaemon

    cfg = load_config(config_path)
    daemon = EvolutionDaemon(config=cfg.evolution_daemon)

    if once:
        result = asyncio.run(daemon.run_once())
        typer.echo(f"[evolution-daemon] run_once exit_code={result.exit_code}")
        for action in result.actions_run:
            typer.echo(f"  [{action.name}] {action.status} {action.duration_ms}ms")
        raise typer.Exit(result.exit_code)
    else:
        asyncio.run(daemon.run_forever())

app.add_typer(evolution_app, name="")  # 平铺到 arena 顶层
```

**Rationale**：
- typer 与既有 CLI 风格一致
- env check 在最早期，避免不必要的 import
- `--once` exit code 直接来自 `RunResult.exit_code`

## Phase 0 检查项

- [x] 所有 NEEDS CLARIFICATION 已解决
- [x] 所有 dependency 已识别 best practice
- [x] 所有 integration 已找到 pattern

Phase 0 输出完成。
