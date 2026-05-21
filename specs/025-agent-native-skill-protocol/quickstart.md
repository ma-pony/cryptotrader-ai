# Quickstart: Agent-Native Skill Protocol Layer

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

## Prerequisites

- arena serve 已启动：`uv run arena serve --port 8003`
- `API_KEY` 环境变量已设置（或 `.env` 含 `API_KEY=...`）
- 当前分支：`025-agent-native-skill-protocol`（branch checkout 不影响运行中的服务）
- 已有 ≥ 3 patterns 在 disk（spec 021 已落地）：
  ```bash
  find agent_memory/{tech,chain,news,macro}/patterns -name "*.md" -not -name ".gitkeep" 2>/dev/null | wc -l
  # 应输出 ≥ 3
  ```

---

## 验证 SC-Q1 — Demo Client 端到端（< 30s）

```bash
export API_KEY="<your-key>"
time uv run python scripts/demo_external_client.py
```

预期输出：
```
[1/4] Fetching bootstrap SKILL.md from /skill/cryptotrader ... OK (1.2 KB)
[2/4] Fetching child skill /skill/verdict-feed ... OK (2.1 KB)
[3/4] Fetching patterns from /api/memory/patterns ... OK (3 patterns)
[4/4] Polling heartbeat from /api/events/heartbeat?since=2026-05-18T00:00:00Z ... OK (12 events)
End-to-end: 4.3s ✓
```

**Pass criteria**：exit 0 + total time < 30s + 4 步全 OK。

---

## 验证 SC-Q2 — 5 个 _external/SKILL.md 文件 + YAML frontmatter

```bash
# 文件存在
ls agent_skills/_external/{cryptotrader,verdict-feed,market-intel,evolution-insights,execution-replay}/SKILL.md
# 应输出 5 行

# YAML frontmatter valid
for f in agent_skills/_external/*/SKILL.md; do
    head -5 "$f" | python3 -c "
import sys, yaml
content = sys.stdin.read()
fm = content.split('---')[1] if '---' in content else ''
data = yaml.safe_load(fm)
assert 'name' in data and 'description' in data, f'Missing required fields in $f'
print(f'$f: name={data[\"name\"]} OK')
"
done
# 应输出 5 行 "name=... OK"
```

---

## 验证 SC-Q3 — Patterns API（**闭 spec 021 T021**）

```bash
curl -s -H "Authorization: Bearer $API_KEY" \
  http://localhost:8003/api/memory/patterns \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'total: {data[\"total\"]}')
print(f'items: {len(data[\"items\"])}')
assert data['total'] >= 3, f'SC-Q3 FAIL: total={data[\"total\"]} < 3'
print('SC-Q3 PASS ✓')
"
```

按 agent 过滤：
```bash
curl -s -H "Authorization: Bearer $API_KEY" \
  "http://localhost:8003/api/memory/patterns?agent=tech" \
  | jq '.items | map(.agent) | unique'
# 应输出 ["tech"]
```

---

## 验证 SC-Q4 — Heartbeat Events

```bash
# 初次 poll
curl -s -H "Authorization: Bearer $API_KEY" \
  "http://localhost:8003/api/events/heartbeat?since=2026-05-18T00:00:00Z&limit=50" \
  | jq '{count: (.items | length), next_cursor: .next_cursor}'
# 应输出 count >= 0, next_cursor non-null 或 null

# Cursor 续传
NEXT=$(curl -s -H "Authorization: Bearer $API_KEY" \
  "http://localhost:8003/api/events/heartbeat?since=2026-05-18T00:00:00Z" \
  | jq -r '.next_cursor')

curl -s -H "Authorization: Bearer $API_KEY" \
  "http://localhost:8003/api/events/heartbeat?cursor=$NEXT" \
  | jq '.items | length'
# 应输出 0（无新事件）或更多
```

按 types 过滤：
```bash
curl -s -H "Authorization: Bearer $API_KEY" \
  "http://localhost:8003/api/events/heartbeat?types=verdict,trade&limit=20" \
  | jq '.items | map(.event_type) | unique'
# 应输出 ["trade", "verdict"] 子集
```

---

## 验证 SC-Q5 — OpenAPI 静态化

```bash
# 6 文件存在
ls docs/api/{verdict,market,events,execution,memory,openapi}.yaml
# 应输出 6 行

# OpenAPI 3.0+ 校验
pip install openapi-spec-validator  # 若未安装
for f in docs/api/*.yaml; do
    openapi-spec-validator "$f" && echo "$f OK"
done
```

---

## 验证 SC-Q6 — Pre-commit Hook Auto-Regen

```bash
# 修改 routes 触发 hook
echo "# comment" >> src/api/routes/events.py
git add src/api/routes/events.py
git commit -m "test: trigger openapi regen"

# 验证 events.yaml 被自动 stage
git diff HEAD~1 --name-only | grep -E "docs/api/events\.yaml"
# 应输出 docs/api/events.yaml

# 回退测试改动
git reset --hard HEAD~1
```

---

## 验证 SC-Q7 — Regression Tests

```bash
uv run python -m pytest tests/ --no-cov 2>&1 | tail -3
# 应输出 "≥ 2476 passed / 0 failed"
```

---

## 验证 SC-Q8 — Prometheus Gauge

```bash
# 触发 ≥ 1 次 heartbeat poll + ≥ 1 次 external skill fetch
curl -s -H "Authorization: Bearer $API_KEY" http://localhost:8003/skill/cryptotrader > /dev/null
curl -s -H "Authorization: Bearer $API_KEY" http://localhost:8003/api/events/heartbeat > /dev/null

# Scrape metrics
curl -s http://localhost:8003/metrics | grep -E \
  "events_heartbeat_poll_count_24h|events_heartbeat_poll_lag_seconds|external_skill_fetch_count_24h"
# 应输出 3 行，每行值 >= 0
```

---

## 验证 SC-Q9 — Spex Review Gates

```bash
# spec review
# 执行 /spex:review-spec — 应无 P0/P1 issues

# plan review（生成 REVIEW-PLAN.md）
# 执行 /spex:review-plan — 应生成 specs/025-agent-native-skill-protocol/REVIEW-PLAN.md
ls specs/025-agent-native-skill-protocol/REVIEW-PLAN.md
```

---

## 验证 SC-Q10 — 单 PR ≤ 4 Commits

```bash
git log --oneline main..025-agent-native-skill-protocol | wc -l
# 应输出 ≤ 5 行（4 implementation commits + 1 docs commit）
```

预期 commits：
- C1：skill 重组 + memory/patterns endpoint（闭 T021）
- C2：events heartbeat + journal extends + 3 gauge + SQL view
- C3：OpenAPI 静态化 + 5 SKILL.md + /skill/<name>
- C4：E2E + demo client + ruff/pytest gate

---

## Troubleshooting

| 症状 | 原因 | 解决 |
|---|---|---|
| `/skill/<name>` 404 | `agent_skills/_external/<name>/SKILL.md` 不存在或重名 | `ls agent_skills/_external/` 验证 |
| `/api/memory/patterns` total=0 | 当前没有 patterns（cases < 5/agent threshold） | 等 trading cycle 跑够 + `arena evolution-daemon --once` 触发 |
| `/api/events/heartbeat` 返回空 | `since` 太早 / 表中无对应 event_type | 改 `since` 或检查 journal 表 |
| demo client 跑 30s+ | 网络问题 / arena serve 卡顿 | 看 arena serve log |
| openapi-spec-validator FAIL | yaml schema 不规范 | 跑 `python scripts/export_openapi.py` 重生成 |
