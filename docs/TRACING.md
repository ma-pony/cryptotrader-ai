# Request Tracing

两层追踪体系：(1) **structlog contextvars** 给每个请求 / CLI / 调度任务一个 `trace_id`；(2) **node trace registry** 把 graph 节点级 timeline 写入 journal commit。

## 1. trace_id（请求/任务粒度）

基于 `structlog.contextvars` 全异步安全。每个 HTTP 请求 / CLI 命令 / scheduler cycle 自动生成或继承一个 trace_id，所有同一请求内的日志事件都会自动带上它。

### 使用方式

```bash
# HTTP — 自动生成
curl http://localhost:8003/api/portfolio/snapshot
# 响应 X-Trace-ID 头返回 trace_id

# 透传客户端 trace_id
curl -H "X-Trace-ID: my-id" http://localhost:8003/api/decisions
```

CLI 和 scheduler 同样自动生成；scheduler 的每个 cycle 都是新 trace_id。

### 日志格式

所有结构化日志统一 JSON：

```json
{
  "event": "risk_gate_rejected",
  "level": "warning",
  "timestamp": "2026-04-29T01:34:42.091Z",
  "trace_id": "0dbb44cc-2790-42c1-b0d1-05c879aa0999",
  "pair": "BTC/USDT",
  "check_name": "cooldown",
  "reason": "Cooldown active for BTC/USDT"
}
```

代码侧：

```python
import structlog
logger = structlog.get_logger()
logger.info("processing_trade", pair="BTC/USDT", action="long")
```

## 2. Node Trace Registry（节点级 timeline）

journal commit 需要每个图节点的执行时间、摘要、聚合 latency_breakdown。问题：LangGraph 的 `record_trade` 节点在 graph **内部** 运行，但 trace 由 graph **外部** 的 runner 累积。两者不能通过 state delta 共享一个累积中的列表（reducer 会覆盖）。

**方案**：`tracing._node_trace_registry: dict[trace_id, list[entry]]` 内存 registry，runner 三段式维护：

```python
# scheduler 路径 (run_graph_traced) 和 chat 路径 (analysis_runner) 都遵循:
trace_register(trace_id)
try:
    async for chunk in graph.astream(...):
        for node_name, update in chunk.items():
            trace_append(trace_id, {
                "node": node_name,
                "duration_ms": elapsed_since_last_chunk,
                "ts": time.time(),
                "summary": _summarize_node_output(node_name, update),
            })
finally:
    trace_unregister(trace_id)  # 清理避免内存泄漏
```

journal 节点在 graph 内部读取：

```python
# nodes/journal.py
from cryptotrader.tracing import trace_get
node_trace = state["data"].get("node_trace") or trace_get(state["metadata"]["trace_id"])
```

### 实测产出

`/api/decisions/{commit_hash}` 返回的 `node_timeline` 字段示例：

```json
{
  "node_timeline": [
    {"node": "init_decision",    "start_ms":      0, "duration_ms":      5},
    {"node": "collect_data",     "start_ms":      5, "duration_ms":  12918},
    {"node": "tech_agent",       "start_ms":  23003, "duration_ms":    407},
    {"node": "debate_round_1",   "start_ms":  50210, "duration_ms":  23783},
    {"node": "verdict",          "start_ms": 103550, "duration_ms":  11993},
    {"node": "execute",          "start_ms": 116860, "duration_ms":    820}
  ],
  "latency_breakdown": {
    "data_ms": 23000, "agents_ms": 27200, "debate_ms": 53300,
    "verdict_ms": 12000, "risk_ms": 1300, "execute_ms": 820,
    "total_ms": 117700
  }
}
```

journal 也存了每节点 `summary`（如 `"price=$76,475 vol=0.0028"`、`"chain_agent: bearish 60%"`），通过 CLI 直查 commit 可见，未来 API 暴露可直接消费。

## 3. 注意事项

- **Py 3.10 兼容性**：`asyncio.TimeoutError is not builtins.TimeoutError`。SSE keepalive、scheduler timeout 必须 `except asyncio.TimeoutError`，否则不触发。
- **trace_id 来源优先级**：state `metadata.trace_id` > structlog contextvars。
- **registry 必须在 finally 清理**：避免长进程内存泄漏。
- **summary 在 chat 和 scheduler 路径行为一致**：均通过 `tracing._summarize_node_output()` 计算。
