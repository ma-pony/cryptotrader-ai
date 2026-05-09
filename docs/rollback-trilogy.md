# Trilogy Rollback Runbook

**适用范围**：spec 017b / 018 / 019 / 020a 任一段落出现生产异常时的回退步骤

**目标 RTO**：≤ 30 分钟完成任一 spec 回退

## 紧急联系

- Owner: TBD
- Slack: #ops-crypto
- 值班记录: docs/DEPLOYMENT.md

---

## 回退顺序原则

**必须按照从最新到最旧的顺序回退**（倒序）：020a → 019 → 018 → 017b。
不可跳过中间 spec 直接回退（spec 间有数据依赖）。

---

## Spec 020a 回退（最简，本 spec）

本 spec 无 schema 变更，回退最简单。

### Step 1: git revert

```bash
# 找到 020a PR 的 commit SHA
git log --oneline | grep "trilogy-ops\|020a"

# revert（替换 <020a-SHA> 为实际 SHA）
git revert <020a-SHA> --no-edit
git push origin main
```

### Step 2: 验证异步化已回退

```bash
# 应能找到 sync 路径（revert 后 classify_case 恢复为 sync def）
grep -n "def classify_case" src/cryptotrader/learning/evolution/ive.py
# 期望输出：def classify_case（无 async 关键字）

# 验证 OTel cache attr 不再写入
python -c "from cryptotrader.agents.base import log_llm_usage; print('OK')"
```

### Step 3: 验证核心功能不受影响

```bash
uv run python -m pytest tests/test_ive.py -v --no-cov
```

### Known data loss

**无**（本 spec 无 schema 变更，无持久化数据变更）。

- `scripts/staging_validate.py` 删除（不影响生产 cycle）
- `docs/rollback-trilogy.md` 删除（本文件，不影响生产 cycle）
- `log_llm_usage()` 恢复仅记录 `cache_read_input_tokens`（丢失 cache hit rate 观测数据）
- IVE `classify_case` 恢复 sync（不影响功能，恢复 event loop 阻塞）
- SkillsGrid 恢复无 `triggers_keywords` badges（不影响数据完整性）
- `propose_new_skill` `inference_failed` 字段消失（不影响 .draft 写入流程）

---

## Spec 019 回退

**前置条件**：已完成 020a 回退（见上一节）。

### Step 1: git revert spec 019 commit

```bash
# spec 019 commit SHA
git revert 3fbf941 --no-edit
git push origin main
```

### Step 2: 数据回退（删除 .draft 文件）

```bash
# 删除所有 propose_new_skill 生成的 .draft 文件
find agent_skills/ -name "SKILL.md.draft" -delete

# 验证删除结果
find agent_skills/ -name "SKILL.md.draft" | wc -l
# 期望：0
```

### Step 3: 验证 skill evolution 回退

```bash
uv run python -m pytest tests/test_e2e_skill_evolution.py -v --no-cov
# 注意：test_e2e_skill_evolution 中依赖 spec 019 功能的用例会失败，此为预期
# 目标：测试套件不因 import error 崩溃

# 验证 EvolvingSkillProvider 回退到 spec 017a 版本
python -c "from cryptotrader.agents.prompt_builder import PromptBuilder; print('OK')"
```

### Known data loss

- **全部 `.draft` 文件**（`agent_skills/<name>/SKILL.md.draft`）— propose_new_skill 生成的提议文件全部丢失
- **`skill_set_hash` 字段**（spec 019 引入，回退后 SKILL.md frontmatter 中该字段残留但不再由系统维护）
- **LLM 推断的 `regime_tags` / `triggers_keywords` / `inference_failed` 字段**（写入到 .draft 文件，随文件删除一并丢失）
- **EvolvingSkillProvider 检索历史**（内存中，进程重启即消失；无持久化影响）

---

## Spec 018 回退

**前置条件**：已完成 020a + 019 回退。

### Step 1: git revert spec 018 commits（共 3 个）

```bash
# spec 018 引入 3 个 commit（按时间倒序 revert）
git revert 1c0302d --no-edit  # 最新 commit 先 revert
git revert 14afc50 --no-edit
git revert 458a0f2 --no-edit
git push origin main
```

### Step 2: DB 回退（清理 spec 018 新增表）

```bash
# 连接 SQLite（默认路径 ~/.cryptotrader/market_data.db）
sqlite3 ~/.cryptotrader/market_data.db << 'EOF'
-- spec 018 新增的 archived rules 表
DROP TABLE IF EXISTS agent_memory_archived;
-- spec 018 扩展的 experience_json 列（不可 DROP COLUMN，需重建表）
-- 如需回到旧 schema，备份后删除整个 agent_memory 表重建
-- WARNING: 会丢失所有 patterns 数据
-- DROP TABLE IF EXISTS agent_memory;
PRAGMA integrity_check;
EOF

# 重置 IVE 分类记录（cases 目录中的 ive_classification 字段）
# 如有必要，手动删除 cases/*.md 中的 ## IVE Classification 段落
find agent_memory/ -name "*.md" -path "*/cases/*" | xargs grep -l "IVE Classification" | wc -l
# 确认 IVE 记录数量，酌情删除
```

### Step 3: 验证 memory evolution 回退

```bash
uv run python -m pytest tests/test_e2e_memory_evolution.py -v --no-cov
# 注意：spec 018 功能测试会失败，此为预期

# 验证 evaluate_node 可以导入（兜底返回 {}）
python -c "from cryptotrader.nodes.evolution import evaluate_node; print('OK')"
```

### Known data loss

- **`agent_memory_archived` 表数据**（所有已归档的规则，DROP TABLE 后永久丢失）
- **`experience_json` 列数据**（spec 018 新增的结构化经验记忆数据）
- **IVE failure classification 记录**（`cases/*.md` 中的 `## IVE Classification` 段落）
- **`archived rules` 数据**（spec 018 FSM archived 状态规则全部丢失）
- **`fundamental_failure_streak` 计数**（内存中，随 evaluate_node 逻辑一起消失）
- 注：若不执行 DB 回退（仅 git revert），`agent_memory_archived` 表会残留但不再被读写

---

## Spec 017b 回退

**前置条件**：已完成 020a + 019 + 018 回退。

### Step 1: git revert spec 017b commits（共 2 个）

```bash
# spec 017b 引入 2 个 commit（倒序 revert）
git revert 18e231e --no-edit
git revert 5b65a4a --no-edit
git push origin main
```

### Step 2: 配置文件数据回退

```bash
# 恢复 agents 配置到 spec 017b 之前的版本
# （spec 017b 把 agent role 从硬编码迁移到 config/agents/*.yaml）
git checkout HEAD~1 -- config/agents/ 2>/dev/null || \
  echo "WARN: config/agents/ 不存在于旧版本，跳过此步"

# 如 config/agents/ 确实存在，验证恢复
ls config/agents/ 2>/dev/null
```

### Step 3: 验证 prompt externalization 回退

```bash
uv run python -m pytest tests/test_e2e_prompt_externalization.py -v --no-cov
# 注意：spec 017b 功能测试会失败，此为预期

# 验证 PromptBuilder 可以导入（回退到 spec 017a 基础版本）
python -c "from cryptotrader.agents.prompt_builder import PromptBuilder; print('OK')"

# 验证 4 agent 可以初始化（不调用 LLM）
python -c "
from cryptotrader.agents.tech import TechAgent
from cryptotrader.agents.chain import ChainAgent
print('agents OK')
"
```

### Known data loss

- **配置驱动 prompt 历史**（恢复到硬编码 ROLE 字符串，config/agents/*.yaml 中的历史修订记录丢失）
- **spec 017b 引入的 prompt template 版本**（YAML 配置中的 template 字段全部丢失）
- **PromptBuilder 的 externalized prompt 缓存**（内存中，无持久化影响）
- **agent config 热重载历史**（spec 017b 支持的配置更新记录消失）
- 注：spec 017a（PromptBuilder 基建）仍保留，不在此 rollback 范围

---

## 快速参考：commit SHA 汇总

| Spec  | Commit SHA(s)                   | 说明              |
|-------|----------------------------------|-------------------|
| 020a  | 待补充（PR 合并后填入）          | trilogy ops 收尾  |
| 019   | 3fbf941                          | skill evolution   |
| 018   | 458a0f2, 14afc50, 1c0302d        | memory evolution  |
| 017b  | 5b65a4a, 18e231e                 | agent 集成切换    |
| 017a  | cfd3acc                          | PromptBuilder 基建（不回退） |

---

## 回退后监控 checklist

1. `curl http://localhost:8000/health` → 200 OK
2. `curl http://localhost:8000/metrics | grep ct_llm_calls` → 有数据
3. 观察 1 个完整 cycle 日志（`arena scheduler start --once`）
4. 确认 `evaluate_node` 不抛 ImportError
5. 确认前端 `/memory` 页可以加载（无 JS 报错）
