# Request Tracing

使用 **structlog** 为每个请求自动生成和传播 trace ID，便于日志追踪和问题排查。

## 功能特性

- ✅ 自动生成唯一 trace ID (UUID)
- ✅ 异步安全（基于 contextvars）
- ✅ 结构化 JSON 日志输出
- ✅ 自动包含时间戳和日志级别
- ✅ 支持 HTTP 头传递 trace ID

## 使用方式

### API 请求

每个 HTTP 请求自动注入 trace ID：

```bash
# 自动生成 trace ID
curl http://localhost:8003/analyze

# 或传递自定义 trace ID
curl -H "X-Trace-ID: my-custom-id" http://localhost:8003/analyze
```

响应头会返回 trace ID：
```
X-Trace-ID: 550e8400-e29b-41d4-a716-446655440000
```

### CLI 命令

每次运行自动生成 trace ID：

```bash
arena run --pair BTC/USDT
# 输出: Arena analyzing BTC/USDT mode=paper trace=550e8400-...
```

### Scheduler

定时任务每次执行自动生成新的 trace ID，可通过 status 查看：

```bash
arena scheduler status
# 显示每个交易对的最新 trace_id
```

## 日志格式

启用 structlog 后，日志输出为 JSON 格式：

```json
{
  "event": "Risk check failed",
  "level": "warning",
  "timestamp": "2026-03-04T13:08:16.559Z",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "pair": "BTC/USDT",
  "check": "MaxPositionSize",
  "reason": "Position size exceeds 10% limit"
}
```

## 代码示例

在任何模块中使用 structlog：

```python
import structlog

logger = structlog.get_logger()

# 日志会自动包含 trace_id
logger.info("processing_trade", pair="BTC/USDT", action="long")
logger.warning("risk_check_failed", check="MaxPositionSize")
logger.error("execution_failed", error=str(e))
```

## 依赖

```bash
uv add structlog
```
