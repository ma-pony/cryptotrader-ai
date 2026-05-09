# Quickstart：Skill Evolution（spec 019）

本文档展示 spec 019 落地后的开发者使用入口。

## 落地后目录结构

```
agent_skills/
├── chain-analysis/SKILL.md           # 含 6 新字段（spec 014 既有 + 本 spec 加）
├── macro-analysis/SKILL.md           # 同上
├── news-analysis/SKILL.md            # 同上
├── tech-analysis/SKILL.md            # 同上
├── trading-knowledge/SKILL.md        # 同上（shared scope）
└── <new-proposed>/SKILL.md.draft     # 由 propose_new_skill 生成（含 LLM 推断 metadata）

src/cryptotrader/learning/evolution/  # spec 018 已建子包；本 spec 加 3 新文件
├── (spec 018 既有)
│   ├── fsm.py / pareto.py / ive.py
│   ├── provider.py / _io.py
│   └── ...
├── idf.py                            # NEW (本 spec C2)
├── skill_metadata_inference.py       # NEW (本 spec C2)
└── skill_provider.py                 # NEW (本 spec C3) — EvolvingSkillProvider

src/api/routes/memory.py              # MODIFY — 加 4 skills endpoints

web/src/pages/memory/                 # spec 018 既有
├── MemoryPage.tsx                    # MODIFY — 加 SkillsGrid section
├── components/SkillsGrid.tsx         # NEW
├── components/{RulesGrid,...}.tsx    # spec 018 既有
└── queries.ts                        # MODIFY — 加 4 hooks

scripts/
└── migrate_018_to_019.py             # NEW (含 5 skill 硬编码 mapping)
```

## 开发者使用场景

### 场景 1：观察 skill 进化状态

打开 web UI `/memory`，滚到 **Skills Grid** section（最后一个 section），看 4 agent + shared 共 5 skill 的 importance / access_count / regime_tags。

或调 API：
```bash
curl 'http://localhost:8000/api/memory/skills?agent=chain'
```

### 场景 2：查看 skill 详情含 body

```bash
curl 'http://localhost:8000/api/memory/skills/chain-analysis'
```

或前端 SkillsGrid 点击 skill 展开。

### 场景 3：查看最近 skill_proposal 创建的 .draft

```bash
curl 'http://localhost:8000/api/memory/skill-proposals?since=2026-05-09T00:00:00Z'
```

返回 list 含 `draft_path` / `llm_inferred_metadata` / `user_saved` 字段。

### 场景 4：手动触发 propose_new_skill

```bash
arena skills propose-new --scope agent:tech
```

完成后查看：
```bash
ls agent_skills/<proposed_name>/SKILL.md.draft
head -20 agent_skills/<proposed_name>/SKILL.md.draft
```

frontmatter 应含 LLM 推断的 regime_tags / triggers_keywords。

### 场景 5：实施迁移（生产环境）

```bash
# 1. 备份
cp -r agent_skills agent_skills.backup_pre_019_$(date +%Y%m%d_%H%M%S)

# 2. dry-run 验证
python scripts/migrate_018_to_019.py --dry-run

# 3. 实跑
python scripts/migrate_018_to_019.py

# 4. 部署 spec 019 代码
git pull && arena scheduler restart

# 5. 监控第 1 个 cycle：retrieval telemetry
tail -f logs/arena.log | grep -E "skill.retrieval|skill.proposal"
```

### 场景 6：本地单测 EvolvingSkillProvider

```python
import pytest
from pathlib import Path
from cryptotrader.learning.evolution.skill_provider import EvolvingSkillProvider

@pytest.fixture()
def provider(tmp_path):
    # tmp_path 含 fixture skill 文件
    return EvolvingSkillProvider(skill_root=tmp_path)

def test_get_available_skills_uses_d_rt_01(provider):
    skills = provider.get_available_skills(agent_id="tech", snapshot={"funding_rate": 0.0005})
    assert len(skills) <= 5  # top-k
    # 第一个 skill 应是 score 最高的
    assert skills[0].name in ["tech-analysis", "trading-knowledge"]
```

### 场景 7：本地单测 IDF 算法

```python
from cryptotrader.learning.evolution.idf import compute_idf, score_skill, extract_query_keywords

corpus = [
    ["funding rate", "OI", "whale"],
    ["fed", "dxy", "macro"],
    ["rsi", "macd", "trend"],
]
idf = compute_idf(corpus)
# log(3/1) ≈ 1.099
assert idf["funding rate"] > 0
assert idf["rsi"] > 0

query = extract_query_keywords({"funding_rate": 0.0005, "rsi": 75})
# 应含 "funding_rate" 和 "rsi"

score = score_skill(["funding rate", "OI"], query, idf)
# "funding rate" 在 query 中 → 加 idf["funding rate"]
assert score > 0
```

## 验证清单（C4 完成后跑）

```bash
# SC-W1 / W2: 5 SKILL.md 含新字段
ls agent_skills/*/SKILL.md | head -5 | xargs -I {} grep -l "importance:" {}

# SC-W3 / W4: 单测 PASS
uv run python -m pytest tests/test_migrate_018_to_019.py tests/test_idf.py tests/test_skill_metadata_inference.py tests/test_evolving_skill_provider.py tests/test_load_skill_tool.py tests/test_skill_proposal_metadata_inference.py -v --no-cov

# SC-W5: DefaultSkillProvider 退役
grep -rn "class DefaultSkillProvider" src/cryptotrader/
# 期望：返回空

# SC-W6: nodes/agents.py 用 EvolvingSkillProvider
grep "_skill_provider = EvolvingSkillProvider" src/cryptotrader/nodes/agents.py

# SC-W8: load_skill_tool 不再直接读文件
grep -n "open(.*SKILL.md" src/cryptotrader/agents/skills/tool.py
# 期望：返回空

# SC-W11: API 测试
uv run python -m pytest tests/test_api_memory_skills.py -v --no-cov

# SC-W12: 前端测试
cd web && pnpm vitest run pages/memory --reporter=verbose

# SC-W14: E2E
uv run python -m pytest tests/test_e2e_skill_evolution.py -v --no-cov

# SC-W15: 全套回归
uv run python -m pytest tests/ --no-cov 2>&1 | tail -3
```

## 与 spec 020 的衔接

spec 020（Ops 子域）启动后无需改 spec 019 代码：

```python
# spec 020 加 daemon
from cryptotrader.ops.daemon import SkillEvolutionDaemon

SkillEvolutionDaemon(
    skill_provider=_skill_provider,  # 本 spec 实例
    interval_seconds=86400,           # 1/day 跑高级进化
).start()

# spec 020 加 cache
prompt_cache_config = {
    "skill_set_hash": _skill_provider.compute_set_hash(),
    "cache_invalidate_on": "skill_proposal",
}
```

均不破坏本 spec 接口契约。
