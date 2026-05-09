# Quickstart：Trilogy Ops（spec 020a）

本文档展示 spec 020a 落地后的开发者使用入口。

## 落地后目录结构

```
scripts/
└── staging_validate.py                     # NEW (本 spec)

docs/
└── rollback-trilogy.md                     # NEW (本 spec)

src/cryptotrader/
├── agents/base.py                          # MODIFY: log_llm_usage cache_creation + OTel span attr
├── learning/evolution/
│   ├── ive.py                              # MODIFY: classify_case async
│   └── skill_metadata_inference.py         # MODIFY: inference_failed
├── learning/skill_proposal.py              # MODIFY: frontmatter inference_failed
└── observability/                          # NEW 子包
    ├── cache_metrics.py                    # NEW: SlidingWindowMetric for cache
    └── ive_metrics.py                      # NEW: SlidingWindowMetric for IVE

src/api/routes/metrics.py                   # MODIFY: 加 2 个 Gauge

web/src/pages/
├── memory/components/SkillsGrid.tsx        # MODIFY: triggers_keywords badges
└── metrics/index.tsx                       # MODIFY: 加 2 个 panel

tests/
├── test_staging_validate.py                # NEW
├── test_llm_usage_cache_attr.py            # NEW
├── test_ive_async.py                       # MODIFY (既有 test_ive.py)
├── test_skill_proposal_metadata_inference.py  # MODIFY
├── test_metrics_endpoint_cache.py          # NEW
└── test_e2e_trilogy_ops.py                 # NEW
```

## 开发者使用场景

### 场景 1：Staging 部署前 smoke check

```bash
# dev 机
python scripts/staging_validate.py --dry-run

# 期望输出
# [step 1] migrate_017_to_018 dry-run: PASS 152ms
# [step 2] migrate_018_to_019 dry-run: PASS 89ms
# [step 3] single cycle smoke (mocked LLM): PASS 1245ms
# [step 4] OTel telemetry 8+3 fields: PASS 12ms
# [step 5] EvolvingSkillProvider retrieval ≥1 hit: PASS 3ms

echo $?  # 应为 0
```

### 场景 2：观察 cache hit rate

```bash
# 触发 1 cycle 后
curl http://localhost:8000/metrics | grep llm_cache_hit_rate
# llm_cache_hit_rate_24h_avg 0.0  # cycle 跑前
# llm_cache_hit_rate_24h_avg 0.18 # cycle 跑后（warm-up 期间偏低）

curl http://localhost:8000/metrics | grep ive_classify_failure
# ive_classify_failure_rate_1h_avg 0.0
```

或前端 `/metrics` 页查看 panel。

### 场景 3：生产事故 rollback

```bash
# 阅读 runbook
cat docs/rollback-trilogy.md

# 按 spec 段执行（如 spec 019 异常）
git revert 3fbf941
rm -rf agent_skills/*/SKILL.md.draft

# 验证
pytest tests/test_e2e_skill_evolution.py -v --no-cov
```

### 场景 4：本地单测 IVE async

```python
import pytest
from cryptotrader.learning.evolution.ive import classify_case

@pytest.mark.asyncio
async def test_classify_case_async():
    result = await classify_case(case_data, llm=mock_llm)
    assert result.classification in {"implementation", "fundamental", "noise"}
```

### 场景 5：本地单测 cache attr 注入

```python
from unittest.mock import patch
from langchain_core.messages import AIMessage
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

def test_log_llm_usage_writes_cache_attr():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    msg = AIMessage(
        content="test",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 80,
            "cache_creation_input_tokens": 20,
        },
    )

    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("llm.call") as span:
        log_llm_usage(msg, caller="test")

    spans = exporter.get_finished_spans()
    attrs = spans[0].attributes
    assert attrs["llm.cache.read_tokens"] == 80
    assert attrs["llm.cache.creation_tokens"] == 20
    assert attrs["llm.cache.hit_rate"] == 0.8
```

### 场景 6：propose_new_skill LLM 失败 .draft 生成

```bash
# 触发 propose
arena skills propose-new --scope agent:tech --force-llm-failure

# 查看 .draft frontmatter
cat agent_skills/<proposed_name>/SKILL.md.draft | head -10
# ---
# regime_tags: []
# triggers_keywords: []
# importance: 0.5
# confidence: 0.5
# inference_failed: true
# ---
```

### 场景 7：SkillsGrid triggers_keywords 渲染

```bash
# 启动前端
cd web && pnpm dev

# 访问 http://localhost:5173/memory
# 滚到 Skills Grid section
# 看每张卡：
# - regime_tags badges（主色）
# - triggers_keywords badges（muted 色，最多 5 + "+N more"）
```

## 验证清单（C4 完成后跑）

```bash
# SC-Z1: staging_validate
python scripts/staging_validate.py --dry-run
echo $?  # 应为 0

# SC-Z2: rollback runbook
test -f docs/rollback-trilogy.md && grep -c "## Spec" docs/rollback-trilogy.md
# 应 ≥ 4（含 020a + 019 + 018 + 017b 段）

# SC-Z3: OTel cache attr
uv run python -m pytest tests/test_e2e_trilogy_ops.py -v --no-cov

# SC-Z4: IVE async grep
grep -n "llm.invoke" src/cryptotrader/learning/evolution/ive.py
# 期望：返回空

# SC-Z5: SkillsGrid triggers badges
grep "triggers_keywords" web/src/pages/memory/components/SkillsGrid.tsx
# 期望：≥ 1 hit
cd web && pnpm vitest run src/pages/memory/components/SkillsGrid.test.tsx --reporter=verbose

# SC-Z6: skill_proposal failure flag
uv run python -m pytest tests/test_skill_proposal_metadata_inference.py::test_llm_failure_writes_flag -v --no-cov

# SC-Z7: dashboard 2 panel manual smoke
cd web && pnpm dev
# 浏览器访问 /metrics，确认 2 个新 panel

# SC-Z8: 全套回归
uv run python -m pytest tests/ --no-cov 2>&1 | tail -3
# 期望：≥ 2339 passed / 0 failed

# SC-Z11: commit count
git log --oneline 021-trilogy-ops..main | wc -l
# 期望：≤ 4
```

## 与 spec 020b 的衔接

spec 020b（reflect daemon + git lineage）启动后：

```python
# 020b 加 daemon
from cryptotrader.ops.daemon import ReflectDaemon

ReflectDaemon(
    cache_metrics=cache_aggregator,  # 复用本 spec aggregator
    interval_seconds=86400,
).start()

# 020b 复用本 spec rollback runbook
# docs/rollback-trilogy.md 加 spec 020b 段
```

均不破坏本 spec 接口契约。
