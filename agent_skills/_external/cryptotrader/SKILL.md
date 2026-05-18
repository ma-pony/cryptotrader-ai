---
name: cryptotrader-ai
description: CryptoTrader AI 多智能体加密货币交易系统的外部集成引导技能。包含本地安装说明、子技能路由表、API 认证方式及未来 JWT 设计存档。
version: "1.0"
---
# CryptoTrader AI — 外部集成引导

外部 agent（Codex / Cursor / Claude Code 等）通过一行 prompt 即可接入
CryptoTrader AI：

```
Read https://<your-host>/skill/cryptotrader
```

## 本地快速安装

```bash
# 克隆并启动所有服务（需要 Docker + uv）
git clone https://github.com/your-org/cryptotrader-ai.git
cd cryptotrader-ai
cp .env.example .env        # 填入 API_KEY / OPENAI_API_KEY
docker compose up -d        # postgres / redis / api / web / scheduler

# 验证 API 可用
curl -s -H "X-API-Key: $API_KEY" http://localhost:8000/health | jq .
```

## 子技能路由表

| 子技能 | 端点 | 说明 |
|---|---|---|
| `verdict-feed` | `GET /skill/verdict-feed` | 交易决策流 — 最近 verdict + 实时订阅 |
| `market-intel` | `GET /skill/market-intel` | 市场情报 — 4 agent 分析结果 + snapshot |
| `evolution-insights` | `GET /skill/evolution-insights` | 进化系统产出 — skill 提案 + pattern 统计 |
| `execution-replay` | `GET /skill/execution-replay` | 执行回放 — journal 事件 + OCO 状态查询 |

每个子技能文档均可通过 `GET /skill/<name>?format=markdown` 或 `?format=json`
独立获取。

## API 认证

所有受保护端点需在请求头携带 API Key：

```http
X-API-Key: <your-api-key>
```

```bash
# 示例：获取最新 verdict
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/api/verdicts/recent?limit=5"
```

环境变量 `AUTH_MODE=disabled` 可在本地开发时绕过认证（每次请求会产生
WARNING 日志）。生产环境必须设置 `AUTH_MODE=enabled` + 强随机 `API_KEY`。

## 格式选项

`GET /skill/<name>` 支持两种响应格式：

- `?format=markdown`（默认）— 返回原始 Markdown，`Content-Type: text/markdown`
- `?format=json` — 返回 `SkillRecord` JSON，包含 `name`、`description`、`body`
  字段及已解析的 YAML frontmatter

```bash
# 以 JSON 格式获取本技能
curl -H "X-API-Key: $API_KEY" \
     "http://localhost:8000/skill/cryptotrader?format=json"
```

## 设计存档：未来 Per-Agent JWT 方案

> **design only, not implemented**
>
> 当前认证为单一共享 `API_KEY`（Bearer Token via `X-API-Key` header）。
> 未来可升级为 per-agent JWT 方案：
>
> - 每个外部 agent 在首次握手时通过 `POST /auth/agent-token` 交换短期 JWT
> - JWT payload 携带 `agent_id`、`allowed_skills` 列表、`exp` (15 min TTL)
> - `/skill/<name>` handler 验证 JWT 中的 `allowed_skills` 白名单
> - `ExternalSkillFetchAggregator` 以 `agent_id` 为维度统计调用频率
> - refresh token 机制：JWT 到期前 agent 自动续签，避免中断
>
> 此设计已在 spec 025 data-model.md 第 3 节存档，待规模扩展时实施。
