# SSE Events Contract

**Feature**: `001-frontend-rewrite-langalpha-port`
**Date**: 2026-04-16
**Source FRs**: FR-602, FR-809
**Endpoint**: `POST /api/chat/stream`
**Transport**: `text/event-stream`（SSE）；通过移植自 LangAlpha 的 `streamFetch` 客户端消费

> **范围**：本契约仅覆盖 ChatAgent 页面（P2，FR-600~604）的 SSE 事件流。其它后端不使用 SSE（Backtest 长任务用轮询，FR-302 / D-9）。

---

## 1. SSE 帧格式

每个事件遵循 SSE 标准：

```
event: <event_name>
data: <JSON payload>

```

事件之间以空行分隔；`data` 字段必须为单行 JSON（多行需序列化为 `\n` escape）。

---

## 2. 事件类型（共 5 种）

> useChatMessages 简化硬约束（NFR-M-007）：仅处理这 5 个事件，多余事件类型必须丢弃并记录警告。

### 2.1 `message_chunk` — 助手消息流式块

```
event: message_chunk
data: {"id":"msg_001","content":"让我分析一下 ","done":false}
```

**字段**：
- `id` (string)：消息 id（同一条消息所有 chunk 共享）
- `content` (string)：本 chunk 文本（前端追加到 `ChatMessage.content_chunks`）
- `done` (boolean)：是否最后一块；`true` 时本消息流结束

**前端处理**：
- 找到 `messages.find(m => m.id === id)` 或新建
- 追加 `content` 到 `content_chunks`
- 不重写（NFR-M-007 约定）

---

### 2.2 `tool_call` — 助手发起工具调用

```
event: tool_call
data: {
  "call_id": "call_42",
  "msg_id": "msg_001",
  "tool_name": "get_decision_commit",
  "args": {"commit_hash": "a1b2c3d"}
}
```

**字段**：
- `call_id` (string)：工具调用 id
- `msg_id` (string)：归属的 assistant 消息 id
- `tool_name` (string)：工具名（白名单校验）
- `args` (object)：工具参数（透传，前端不解析）

**前端处理**：
- UI 显示一个"工具调用中…"占位
- 等待对应 `tool_result` 事件

---

### 2.3 `tool_result` — 工具调用返回

```
event: tool_result
data: {
  "call_id": "call_42",
  "result": {"...": "..."},
  "error": null
}
```

**字段**：
- `call_id` (string)：与 `tool_call` 对应
- `result` (unknown | null)：工具返回（成功时）
- `error` (string | null)：错误消息（失败时）

**前端处理**：
- 匹配 `call_id`，把占位替换为 result 渲染
- 错误时显示红色警告 + error 文本

---

### 2.4 `inline_widget` — 内联可视化 widget

```
event: inline_widget
data: {
  "widget_id": "w_001",
  "msg_id": "msg_001",
  "type": "chart",
  "payload": {
    "chart_type": "equity_curve",
    "points": [{"ts": "2026-04-16T13:00:00Z", "equity": 10000}]
  }
}
```

**字段**：
- `widget_id` (string)：唯一 id
- `msg_id` (string)：归属消息
- `type` (`"chart" | "table" | "verdict" | "markdown"`)：widget 类型
- `payload` (object)：渲染数据（必须可 JSON.stringify；含 NaN/Infinity 由 streamFetch 的 JSON.parse patch 处理）

**前端处理**：
- 在消息流中插入 `<InlineWidget>` 组件
- iframe sandbox 注入 24+ CSS 主题变量（NFR-S-003）
- iframe `sandbox="allow-scripts"`，**不带** `allow-same-origin`

---

### 2.5 `verdict` — 多代理决策最终裁决

```
event: verdict
data: {
  "msg_id": "msg_001",
  "action": "long",
  "size": 0.5,
  "confidence": 0.72,
  "reasoning": "Markdown 推理…",
  "trace_id": "abc123"
}
```

**字段**：与 `Verdict` data-model 对齐（`source` 在此场景固定为 `ai`，省略）
- `msg_id` (string)
- `action` (`"long" | "short" | "hold"`)
- `size` (number, 0~1)
- `confidence` (number, 0~1)
- `reasoning` (string, Markdown)
- `trace_id` (string?, 可选)

**前端处理**：
- 渲染为 `VerdictCard`（与 Decisions 页详情复用）
- 若 `trace_id` 存在 + `OTLP_ENDPOINT` 配置，显示 Jaeger 链接（同 FR-203）

---

## 3. 流终止

后端在所有事件发送完毕后发送：
```
event: done
data: {}

```

`done` 事件**不在** 5 种处理列表中，但前端 `streamFetch` 收到后必须关闭 reader 并标记会话结束。

---

## 4. 错误处理

### 4.1 HTTP 层错误

streamFetch 在初次握手就识别：

| HTTP | 处理 |
|------|------|
| 401 | Toast "X-API-Key 无效"，停止流 |
| 429 | exponential backoff 重连（最多 3 次） |
| 413 | Toast "请求过大"，丢弃当前会话最后一条 user 消息 |
| 404 | Toast "聊天接口不可用"，禁用 chat 页面 |
| 500/503 | Toast "服务暂时不可用，请稍后重试" |

### 4.2 流中错误事件

```
event: error
data: {"detail": "LLM provider 超时", "code": "LLM_TIMEOUT", "trace_id": "abc"}
```

前端处理：在当前 assistant 消息末尾追加红色错误块，停止流。

---

## 5. 取消

前端通过 `AbortController.signal` 中止 fetch；后端检测连接关闭后清理 LLM 请求与节点状态。

---

## 6. 调试

streamFetch 支持 `debug: true`（NFR-O-004），dev 模式下：
- 每条 SSE chunk 打印到 console
- 包含解析后的事件类型 + payload
- 不在 prod 启用（避免敏感数据泄漏）

---

## 7. 实施清单

后端 (`src/api/routes/chat.py`)：
- [ ] 接收 `POST /api/chat/stream`
- [ ] 触发 `build_trading_graph` 或专用 chat graph
- [ ] 把图节点输出适配为 5 种 SSE 事件
- [ ] 流完发送 `done`
- [ ] 错误 → `error` 事件
- [ ] 客户端断连 → 清理 LLM call

前端 (`web/src/hooks/useChatMessages.ts`)：
- [ ] ≤ 500 行硬限（NFR-M-007）
- [ ] 处理 5 种事件 + `done` + `error`
- [ ] 维护消息流状态（Zustand `useChatStore`）
- [ ] 支持 `AbortController` 取消
- [ ] 持久化到 IndexedDB（FR-604）
