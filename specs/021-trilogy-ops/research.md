# Phase 0：研究与决策

**关联 spec**：[spec.md](spec.md)
**关联 brainstorm**：[brainstorm/06-spec-020a-trilogy-ops.md](../../brainstorm/06-spec-020a-trilogy-ops.md)
**Date**: 2026-05-09

## 概述

本 spec 5 项关键决策已在 brainstorm 阶段（2026-05-09）完成，3 项 spec ambiguity 已在 clarify 阶段（2026-05-09）解决。本文档记录最终决定 + 4 项 spot-check 结果 + 实施细节研究。

## Technical Context 中无 NEEDS CLARIFICATION 项

Brainstorm 5 项决策 + 4 项 spot-check + 3 项 clarify 已消除全部 ambiguity。

## 5 项关键决策（来自 brainstorm）

| # | 决策 | 来源 |
|---|---|---|
| Q1 spec 拆分 | B 拆 020a + 020b 两段 | trilogy 风格一致 |
| Q2 cache 策略 | B 弱 cache（仅观测） | 5min TTL vs 1h cycle 命中率 ≈ 0 |
| Q3 advisory 收尾 | A 全 3 个打包 | surgical fix 打包不增复杂度 |
| Q4 部署清单 | A 验证脚本 + runbook 都做 | 跨 spec 部署风险压缩 |
| Q5 monitoring | C 仅 2 核心指标，不告警 | alert fatigue 防范 |

## 4 项 spot-check 结果（2026-05-09）

| # | 检查项 | 结果与修订 |
|---|---|---|
| 1 | IVE `classify_case` sync `llm.invoke` | ✓ src/cryptotrader/learning/evolution/ive.py:247 sync 调用确认 |
| 2 | SkillsGrid 缺 `triggers_keywords` | ✓ web/src/pages/memory/components/SkillsGrid.tsx 仅渲染 regime_tags |
| 3 | `skill_metadata_inference` 缺 failure flag | ✓ except 兜底但未写 metadata 字段 |
| 4 | LLM cache usage 提取位置 | ⚠️ 修订 — log_llm_usage 已读 cache_read 写 structlog；缺 cache_creation；未写 OTel span。FR-Z7~9 需具体化（见 Decision 1） |

## 3 项 clarify 决策（来自 spec.md Clarifications 段）

| # | Question | Answer |
|---|---|---|
| C1 | FR-Z18 metric 聚合源 | 复用 spec 015 既有 `/metrics` Prometheus endpoint，新增 2 metric 由 OTel span attr 进程内聚合 |
| C2 | SC-Z3 LLM call 数 | ≥ 4 agent LLM 点必须含 cache 字段（verdict 因 weighted-downgrade 可能跳过 LLM 不计入） |
| C3 | hit_rate read+creation=0 | 仍写 3 字段全 0，保持字段一致性 |

## 实施细节决策

### Decision 1：log_llm_usage OTel span attr 写入路径

**Decision**：FR-Z7~9 改造方式：

```python
# src/cryptotrader/agents/base.py
from opentelemetry import trace as _otel_trace

def log_llm_usage(response: AIMessage, *, caller: str) -> None:
    if not isinstance(response, AIMessage):
        return
    usage = response.usage_metadata or {}
    if not usage:
        return

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    model_name = (response.response_metadata or {}).get("model_name", "unknown")

    cache_read = usage.get("cache_read_input_tokens", 0)
    if not cache_read:
        prompt_details = usage.get("input_token_details") or {}
        cache_read = prompt_details.get("cached", 0) if isinstance(prompt_details, dict) else 0
    cache_creation = usage.get("cache_creation_input_tokens", 0)

    total_cache = cache_read + cache_creation
    cache_hit_rate = (cache_read / total_cache) if total_cache > 0 else 0.0

    # spec 020a 新增：OTel span attr
    span = _otel_trace.get_current_span()
    if span and span.is_recording():
        span.set_attribute("llm.cache.read_tokens", cache_read)
        span.set_attribute("llm.cache.creation_tokens", cache_creation)
        span.set_attribute("llm.cache.hit_rate", cache_hit_rate)

    # 既有 structlog 路径保留
    _structlog.info(
        "llm_usage",
        caller=caller,
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        prompt_cache_hit=cache_read > 0,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        cache_hit_rate=cache_hit_rate,
    )
```

**Rationale**：
- OTel span 当前作用域 = LLM call span（spec 010 已设置 LLM call 包装）；attr 注入到该 span
- structlog 路径保留作为 dev 机调试 + 历史日志兼容
- `is_recording()` guard 防止 OTel SDK 未初始化（test 环境）抛异常
- 边界：read + creation 同时为 0 时 hit_rate = 0（与 FR-Z8 clarify 一致）

### Decision 2：Prometheus 2 metric 聚合实现

**Decision**：FR-Z18 改造方式：

```python
# src/api/routes/metrics.py
from prometheus_client import Gauge

# spec 020a 新增 2 metric（添加到既有 REGISTRY）
LLM_CACHE_HIT_RATE_GAUGE = Gauge(
    "llm_cache_hit_rate_24h_avg",
    "LLM prompt cache hit rate, 24h sliding window average",
)
IVE_CLASSIFY_FAILURE_RATE_GAUGE = Gauge(
    "ive_classify_failure_rate_1h_avg",
    "IVE classify_case failure rate, 1h sliding window average",
)
```

聚合数据来源：进程内维护 2 个 ring buffer（`collections.deque(maxlen=N)`）：

- LLM cache：每次 LLM call 完成后 push (timestamp, hit_rate)；24h sliding window
- IVE failure：每次 IVE classify_case 完成后 push (timestamp, success/failure)；1h sliding window

`/metrics` endpoint 调用前先 trigger gauge 更新（lazy compute）。

```python
# src/cryptotrader/observability/cache_metrics.py（新模块）
from collections import deque
from time import time
from threading import Lock

class CacheMetricsAggregator:
    def __init__(self, window_seconds: int = 86400):
        self._window = window_seconds
        self._buffer: deque[tuple[float, float]] = deque()
        self._lock = Lock()

    def record(self, hit_rate: float) -> None:
        with self._lock:
            now = time()
            self._buffer.append((now, hit_rate))
            self._evict_expired(now)

    def average(self) -> float:
        with self._lock:
            self._evict_expired(time())
            if not self._buffer:
                return 0.0
            return sum(r for _, r in self._buffer) / len(self._buffer)

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._window
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()
```

**Rationale**：
- ring buffer 进程内 lock 保护，无 IO 开销
- 单进程内存上限：cache 24h × ~5 LLM/cycle × 24 cycle/day = ~120 entries ≈ 几 KB；IVE 1h × ~10/cycle = ~10 entries
- 不引入 Prometheus exporter 单独 pipeline；与 spec 015 既有 `/metrics` endpoint 合流
- 非生产 process 重启 ring buffer 清零（acceptable，dashboard 接受 24h warm-up window）

### Decision 3：staging_validate 脚本结构

**Decision**：FR-Z1~3 改造方式：

```python
# scripts/staging_validate.py
import argparse
import sys
import time
from typing import Callable

class StepResult:
    def __init__(self, idx: int, name: str, status: str, duration_ms: int, error: str = ""):
        self.idx, self.name, self.status, self.duration_ms, self.error = idx, name, status, duration_ms, error

    def fmt(self) -> str:
        line = f"[step {self.idx}] {self.name}: {self.status} {self.duration_ms}ms"
        return line if not self.error else f"{line}\n  ERROR: {self.error}"

def run_step(idx: int, name: str, fn: Callable[[], None]) -> StepResult:
    start = time.time()
    try:
        fn()
        return StepResult(idx, name, "PASS", int((time.time() - start) * 1000))
    except Exception as e:
        return StepResult(idx, name, "FAIL", int((time.time() - start) * 1000), str(e))

def main(dry_run: bool = True) -> int:
    steps = [
        (1, "migrate_017_to_018 dry-run", lambda: _migrate("017_to_018", dry_run)),
        (2, "migrate_018_to_019 dry-run", lambda: _migrate("018_to_019", dry_run)),
        (3, "single cycle smoke (mocked LLM)", _run_smoke_cycle),
        (4, "OTel telemetry 8+3 fields", _check_otel_fields),
        (5, "EvolvingSkillProvider retrieval ≥1 hit", _check_retrieval),
    ]
    results = [run_step(idx, name, fn) for idx, name, fn in steps]
    for r in results:
        print(r.fmt())
    failed = [r for r in results if r.status == "FAIL"]
    return 1 if failed else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
```

**Rationale**：
- 简单 step list pattern，无需 framework
- mocked LLM via `unittest.mock.patch("langchain_openai.ChatOpenAI.ainvoke")` (spec 014 测试 pattern)
- step 4 通过 OTel test harness（in-memory exporter）assert 字段
- step 5 instantiate EvolvingSkillProvider + 调 get_available_skills(...) 检查返回 ≥ 1

### Decision 4：rollback-trilogy.md 文档结构

**Decision**：FR-Z4~6 文档结构：

```markdown
# Trilogy Rollback Runbook

## 适用范围
spec 017b / 018 / 019 / 020a 任一段落异常时回退步骤

## 紧急联系
- Owner: TBD
- Slack: #ops

## Spec 020a 回退（最简，本 spec）
### Step 1: git revert
git revert <020a SHA>
### Step 2: 验证
grep -n "llm.invoke" src/cryptotrader/learning/evolution/ive.py  # 应返回 sync 路径
### Known data loss
无（本 spec 无 schema 变更）

## Spec 019 回退
### Step 1: git revert 3fbf941
git revert 3fbf941
### Step 2: DB 回退
rm -rf agent_skills/*/SKILL.md.draft
### Step 3: 验证
pytest tests/test_e2e_skill_evolution.py
### Known data loss
- 全部 .draft 文件
- skill_set_hash 重新计算（spec 019 引入）

## Spec 018 回退
### Step 1: git revert 458a0f2 14afc50 1c0302d
### Step 2: DB 回退
DROP TABLE agent_memory_archived;
### Step 3: 验证
pytest tests/test_e2e_memory_evolution.py
### Known data loss
- archived rules 数据（spec 018 archived FSM 状态）
- IVE classify_case 结果

## Spec 017b 回退
### Step 1: git revert 5b65a4a 18e231e
### Step 2: 数据回退
git checkout HEAD~1 -- config/agents/  # 恢复配置文件
### Step 3: 验证
pytest tests/test_e2e_prompt_externalization.py
### Known data loss
- 配置驱动 prompt 历史（恢复硬编码 ROLE）
```

**Rationale**：
- 倒序排列（最近 spec 在前，回退顺序从最远开始）
- 每段含 3 step + known data loss
- pytest 验证比 grep 更可靠

### Decision 5：IVE async 调用方梳理

**Decision**：FR-Z11 调用方修正：

通过 grep `classify_case\|ive.classify` 发现调用点：
- `src/cryptotrader/nodes/evaluate.py:evaluate_node()` — 已 async，改 await ✓
- `tests/test_ive.py` — 改 pytest.mark.asyncio + await
- `tests/test_e2e_memory_evolution.py` — 已通过 evaluate_node 调用，自动联动

落地时 grep 全 repo 确认无遗漏。

**Rationale**：
- 调用图小（2 直接调用方），改 await 风险低
- spec 018 e2e 测试已覆盖 evaluate_node 路径，async 切换有现成回归覆盖

## Phase 0 检查项

- [x] 所有 NEEDS CLARIFICATION 已解决
- [x] 所有 dependency 已识别 best practice
- [x] 所有 integration 已找到 pattern

Phase 0 输出完成。
