# Contract：Skill API Routes

**模块路径**：`src/api/routes/memory.py`（spec 018 既有；本 spec 扩展）

**注册位置**：`src/api/main.py` 已 register memory router with prefix `/api/memory`（spec 018 已落地）；本 spec 仅在同 router 内加 4 个新 endpoints。

## 4 个新 Endpoints

### 1. `GET /api/memory/skills`

**Query 参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `agent` | str | ✗ | tech / chain / news / macro；缺省返回所有 |

**Response 200**：

```json
{
  "items": [
    {
      "name": "chain-analysis",
      "scope": "agent:chain",
      "version": "1.0",
      "regime_tags": [],
      "triggers_keywords": ["funding rate", "exchange flow", "..."],
      "importance": 0.7,
      "confidence": 0.7,
      "access_count": 23,
      "last_accessed_at": "2026-05-09T12:34:56Z",
      "manually_edited": false,
      "description": "On-chain analysis skill ..."
    }
  ],
  "total": 5
}
```

**Response 400**：参数验证失败（如 agent 不是合法 agent_id）

### 2. `GET /api/memory/skills/{name}`

**Path 参数**：`name: str` —— skill name

**Response 200**：

```json
{
  "name": "chain-analysis",
  "scope": "agent:chain",
  "version": "1.0",
  "regime_tags": [],
  "triggers_keywords": [...],
  "importance": 0.7,
  "confidence": 0.7,
  "access_count": 23,
  "last_accessed_at": "...",
  "manually_edited": false,
  "description": "...",
  "body": "# On-Chain Analysis Agent Skill\n\n## Agent Role\n..."
}
```

**Response 404**：skill name 不存在

### 3. `GET /api/memory/skill-access`

**Query 参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `since` | ISO8601 | ✗ | 起始时间；缺省 24h 前 |
| `agent` | str | ✗ | 过滤特定 agent 的 access 事件 |

**Response 200**：

```json
{
  "items": [
    {
      "skill_name": "chain-analysis",
      "scope": "agent:chain",
      "access_count": 23,
      "last_accessed_at": "2026-05-09T12:34:56Z"
    }
  ],
  "total": 5
}
```

**Note**：本 endpoint 实质返回当前 skill access 状态；如需细粒度"每次 access 时间戳"，需要 spec 020 daemon 加 access log（OOS）。本 spec 仅返回 SKILL.md frontmatter 中 access_count + last_accessed_at。

### 4. `GET /api/memory/skill-proposals`

**Query 参数**：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `since` | ISO8601 | ✗ | 起始时间；缺省 7 天前 |

**Response 200**：

```json
{
  "items": [
    {
      "name": "macro-fed-rate-cut-bull",
      "draft_path": "agent_skills/macro-fed-rate-cut-bull/SKILL.md.draft",
      "created_at": "2026-05-09T08:00:00Z",
      "llm_inferred_metadata": {
        "regime_tags": ["high_funding"],
        "triggers_keywords": ["fed", "rate cut", "bull"],
        "importance": 0.6,
        "confidence": 0.6
      },
      "llm_call_failed": false,
      "user_saved": false
    }
  ],
  "total": 1
}
```

**Response 字段说明**：
- `draft_path`：`.draft` 文件相对路径
- `created_at`：从 `.draft` 文件 mtime 取
- `llm_inferred_metadata`：从 `.draft` frontmatter 读
- `user_saved`：`.draft` 同名 SKILL.md 是否已存在（即用户是否已 manual save）

## 错误处理

- IO 异常返回 500 + `{"error": "memory_io_error", "detail": "..."}`
- 参数解析错误返回 400 + `{"error": "invalid_query", "detail": "..."}`
- 资源不存在返回 404 + `{"error": "not_found", "detail": "..."}`

## 鉴权

沿用 spec 015 既有 X-API-Key header（同 spec 018 endpoints）。

## Caching

- `/skills` / `/skills/{name}` / `/skill-access`：`Cache-Control: max-age=30`
- `/skill-proposals`：`Cache-Control: max-age=300`（变化频率低）

## 单测要求

参考 spec.md SC-W11：`tests/test_api_memory_skills.py` ≥ 8 用例 PASS：
1. `GET /api/memory/skills?agent=tech` 返回 200 + JSON list
2. `GET /api/memory/skills/{name}` 返回 200 + 详情含 body
3. `GET /api/memory/skills/unknown` 返回 404
4. `GET /api/memory/skill-access?since=...` 返回 200 + access 事件
5. `GET /api/memory/skill-proposals?since=...` 返回 200 + proposal 历史
6. 缺鉴权 header 返回 401
7. 错误参数返回 400
8. Response 含 Cache-Control header

## 集成 Frontend

`web/src/pages/memory/queries.ts` 加 4 hooks：
```ts
useSkills({ agent }, { staleTime: 30000 })
useSkillByName({ name }, { staleTime: 30000 })
useSkillAccess({ since, agent }, { staleTime: 30000 })
useSkillProposals({ since }, { staleTime: 300000 })
```

复用 spec 018 既有 React Query / api client 模式。
