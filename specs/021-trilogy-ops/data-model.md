# Data Model: Spec 020a — Trilogy Ops

本 spec **无新数据 schema 变更 / 无 migrate 脚本**。仅扩展既有运行时 entity 字段（OTel span attr / metadata frontmatter / structlog event）。

## 运行时 Entity（无持久化）

### 1. CacheTelemetryAttr（OTel span attribute set）

**位置**：每个 LLM call OTel span（spec 010 既有）

**字段（spec 020a 新增）**：

| 字段名 | 类型 | 取值范围 | 说明 |
|---|---|---|---|
| `llm.cache.read_tokens` | int | ≥ 0 | 从 cache 命中读取的 token 数（来自 `usage.cache_read_input_tokens`） |
| `llm.cache.creation_tokens` | int | ≥ 0 | 写入 cache 的 token 数（来自 `usage.cache_creation_input_tokens`） |
| `llm.cache.hit_rate` | float | [0.0, 1.0] | read / (read + creation)；分母 0 时为 0.0 |

**写入路径**：`src/cryptotrader/agents/base.py:log_llm_usage()` 在 `acompletion_with_fallback` 等所有 LLM call 调用后写入

**Validation rules**：
- `read_tokens + creation_tokens = 0` 时仍写 3 字段全 0（保持字段一致性，FR-Z8 clarify）
- OTel SDK 未初始化时（`span.is_recording()` False）跳过写入不抛异常

---

### 2. SkillMetadata.inference_failed（既有 dict 字段扩展）

**位置**：
- `src/cryptotrader/learning/evolution/skill_metadata_inference.py:infer_skill_metadata()` 返回值
- `agent_skills/<name>/SKILL.md.draft` frontmatter

**字段（spec 020a 新增）**：

| 字段名 | 类型 | 取值范围 | 说明 |
|---|---|---|---|
| `inference_failed` | bool | True \| False | LLM 推断 except 兜底时为 True；正常路径为 False（默认） |

**触发条件**：
- True：`infer_skill_metadata()` 内部 LLM 调用抛异常（OpenAI API 错误 / JSON 解析失败 / 超时等）走 except 路径
- False：LLM 推断成功 + JSON parse 成功

**与既有字段的关系**：
- `regime_tags` / `triggers_keywords` / `importance` / `confidence` 在 inference_failed=True 时使用默认值（`[] / [] / 0.5 / 0.5`）
- 与 spec 019 既有 metadata schema 兼容（仅加 1 字段，不改既有字段语义）

---

### 3. SlidingWindowMetric（in-process aggregator）

**位置**：`src/cryptotrader/observability/cache_metrics.py`（新模块）+ `src/cryptotrader/observability/ive_metrics.py`（新模块）

**字段**：

| 字段名 | 类型 | 说明 |
|---|---|---|
| `_buffer` | `deque[tuple[float, T]]` | (timestamp, value) ring buffer |
| `_window` | int (seconds) | sliding window 大小（86400 for cache / 3600 for IVE） |
| `_lock` | `threading.Lock` | 并发写保护 |

**操作**：
- `record(value: T)`：push 当前 timestamp + value，evict 过期 entry
- `average() -> float`：计算 window 内 value 平均
- `_evict_expired(now: float)`：丢弃 timestamp < now - window 的 entry

**Validation rules**：
- buffer 空时 `average()` 返回 0.0
- 进程重启时 buffer 清空（acceptable，dashboard 接受 24h warm-up）

---

### 4. ValidationStep（staging_validate 内部 dataclass）

**位置**：`scripts/staging_validate.py`

**字段**：

| 字段名 | 类型 | 说明 |
|---|---|---|
| `idx` | int | 步骤序号（1-N） |
| `name` | str | 步骤名（如 "migrate_017_to_018 dry-run"） |
| `status` | Literal["PASS", "FAIL"] | 执行结果 |
| `duration_ms` | int | 步骤耗时（毫秒） |
| `error` | str | FAIL 时的错误信息（PASS 时为空） |

**输出格式**（FR-Z3）：
```
[step 1] migrate_017_to_018 dry-run: PASS 152ms
[step 2] migrate_018_to_019 dry-run: PASS 89ms
[step 3] single cycle smoke (mocked LLM): FAIL 4231ms
  ERROR: Mock LLM did not return usage_metadata
[step 4] OTel telemetry 8+3 fields: SKIP
[step 5] EvolvingSkillProvider retrieval ≥1 hit: SKIP
```

**Exit code**（FR-Z2 f）：
- 全 PASS → exit 0
- 任一 FAIL → exit 1

---

## 既有 entity 字段映射（不变）

下列 entity 在 spec 020a 中**字段不变**，仅说明依赖关系：

- `AIMessage.usage_metadata`（LangChain 既有）：含 `cache_read_input_tokens` / `cache_creation_input_tokens` 两 key（langchain-openai >= 1.1.10 / Anthropic provider）
- `Skill` dataclass（spec 019）：6 字段全部不变
- `PatternRecord` / `CaseRecord`（spec 018）：字段不变
- `OpenTelemetry Span`（spec 010）：本 spec 仅扩展 attribute 集合，不改 span 结构

---

## State Transitions（无）

本 spec 无 FSM / 状态机变化。

## 数据流向

```
LLM call (4 agent + verdict)
   ↓ AIMessage.usage_metadata
log_llm_usage()
   ↓ 写入 OTel span attr (3 cache fields)
   ↓ structlog event (4+3 fields)
CacheMetricsAggregator.record(hit_rate)
   ↓ deque ring buffer (24h)

/metrics endpoint 触发
   ↓ Gauge.set(aggregator.average())
   ↓ generate_latest()
prometheus output

dashboard `/metrics` 页
   ↓ React Query 拉取 prometheus output
   ↓ panel render
```

```
IVE classify_case
   ↓ try / except
IveMetricsAggregator.record(success: bool)
   ↓ deque ring buffer (1h)
   ↓ Gauge.set(failure_rate)
```
