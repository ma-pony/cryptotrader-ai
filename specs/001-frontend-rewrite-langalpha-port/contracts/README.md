# Interface Contracts

**Feature**: `001-frontend-rewrite-langalpha-port`
**Date**: 2026-04-16

本目录定义前后端之间、前端与外部 widget 之间的接口契约。每个文件聚焦一类合约：

| 文件 | 内容 |
|------|------|
| [http-endpoints.md](./http-endpoints.md) | FastAPI REST endpoints（FR-800~810），含路径、请求、响应、错误码 |
| [sse-events.md](./sse-events.md) | ChatAgent SSE 事件协议（FR-602 / FR-809） |
| [ui-routes.md](./ui-routes.md) | 前端路由表 + 每路由的组件契约（lazy chunk、状态依赖、URL 同步） |

## 约定

- **数据形态**：以 [data-model.md](../data-model.md) 为单一事实来源；本目录的 schema 引用该文件
- **HTTP 方法**：REST 资源遵循 RFC 7231 语义（GET 幂等、POST 创建/触发、DELETE 删除）
- **错误响应**：统一 `ApiError` 形态（`{ detail: string, code?: string, trace_id?: string }`）；HTTP 状态码遵循 FastAPI 默认
- **认证**：所有 endpoint 走 `X-API-Key` header；后端未配置 API_KEY 时跳过校验（开发模式，NFR-S-001）
- **i18n**：错误 `detail` 字段返回中文（项目默认）；前端可选择本地化覆盖

## 验证

每个 endpoint 必须满足：
1. 后端 pytest 覆盖（happy path + 至少 1 个 error path）
2. 前端 zod schema 解析成功（`*.schema.ts` 与 pydantic 字段对齐）
3. 在 `docker compose up -d` 后 `curl` 验通（A-7 纪律）
