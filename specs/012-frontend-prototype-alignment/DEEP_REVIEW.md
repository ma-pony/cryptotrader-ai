# Deep Review Report · frontend-prototype-alignment

**Date**: 2026-04-24
**Scope**: 本会话后端新增/修改的 15 个文件（9 核心 + 6 API 路由） + 3 个新测试文件
**Agents**: Security · Performance · Architecture · Testing · Correctness
**Auto-fix**: disabled (per user request, 仅出报告)
**Status**: ⚠️ **Issues Found** — 8 Critical / 28 Important / 15 Minor / 3 Nice-to-have

---

## 执行摘要

共 **54 个问题**，跨 5 维度。最关键的是 3 个性能 Critical（会把 risk/metrics/decisions 几个热路由从 ~100ms 拖到 ~1-2s）和 2 个正确性 Important（token 成本可能 17× 高估，SQLite DB 不迁移新列）。

**跨 agent 重合（同一问题被多个视角标记）**：
1. **token 前缀匹配顺序错**（Correctness Important + Performance Minor）→ `gpt-4o-mini-YYYY-MM-DD` 误匹配 `gpt-4o` 前缀，成本膨胀 17×
2. **`_build_bias` 每次 detail 扫 1000 commits**（Performance Critical + Security Minor DoS）
3. **Callback 注入耦合**（Architecture Critical + Correctness Minor）→ manifest path 跳过 callback，token 统计漏记
4. **新 API 字段无集成测试**（Testing Critical × 3）→ debate_turns / bias / latency_breakdown / correlation_groups / cost_14d 等全部无 response shape 断言

---

## 问题分类汇总

| 维度 | Critical | Important | Minor | Nice-to-have | 合计 |
|---|---|---|---|---|---|
| 🔒 Security | 0 | 3 | 2 | 1 | 6 |
| ⚡ Performance | 3 | 4 | 2 | 0 | 9 |
| 🏗️ Architecture | 2 | 8 | 3 | 0 | 13 |
| 🧪 Testing | 3 | 9 | 2 | 1 | 15 |
| ✅ Correctness | 0 | 4 | 6 | 1 | 11 |
| **合计** | **8** | **28** | **15** | **3** | **54** |

---

## 🔴 Critical 问题（8 个 — 建议立即修）

### C1 · 性能 · Risk 路由 7 个 await 串行
**文件**: `src/api/routes/risk.py:360`
7 个独立 compute/build 函数（daily_loss/drawdown/exposure/cvar/corr_groups/cooldowns/recent_blocks）被串行 await。每个 ~30-80ms DB round-trip × 7 = **210-560ms local / 400-1000ms cloud**。
**修法**: `asyncio.gather(*7 coroutines)` 一把拿。

### C2 · 性能 · `get_portfolio()` / `_load_snapshots` 冗余调用
**文件**: `src/api/routes/risk.py` + `src/api/routes/portfolio_v2.py`
同一个 /api/risk/status 请求里 `pm.get_portfolio()` 被调 **4 次**（daily_loss / total_exposure / correlation_groups / _known_pairs）。`_load_snapshots` 在 portfolio route 被调 **3 次**（get_daily_pnl / get_drawdown / _compute_extras.sharpe）。
**修法**: 在 route handler 顶部 fetch 一次，作为参数下发。

### C3 · 性能 · `_build_bias` 每次 detail 请求扫 1000 commits
**文件**: `src/api/routes/decisions.py:610`
`get_decision` 每次 call `_build_bias(store)` → `detect_biases(store, days=30)` → `store.log(limit=1000)`。Bias 数据 30 天窗口，不随单个 commit 变化，但每次刷详情都全表扫一遍。**每请求 50-200ms 无用功**（Postgres JSONB）或 300-800ms（in-memory）。
**修法**: 模块级 `_bias_cache = (ts, BiasOut | None)`，TTL 60s；或拆到独立 `/api/decisions/bias-summary` lazy 端点。

### C4 · 架构 · `collect_snapshot` SRP 违反
**文件**: `src/cryptotrader/nodes/data.py:64`
既是数据采集节点，又负责 `start_ledger()` 观察性生命周期。如果未来有另一种图入口不走 `collect_snapshot`（已经有 `lite_graph` / `debate_graph`），就无法初始化 ledger，token 统计静默失效。
**修法**: 新建 `graph_init` 节点或 LangGraph middleware 做 ledger 初始化，`collect_snapshot` 只负责数据。

### C5 · 架构 · `create_llm` 强制注入 callback，无 opt-out
**文件**: `src/cryptotrader/agents/base.py:168`
callers 无法禁用 token tracker（例如 backtest cost-free 模式、单测场景）。必须 monkeypatch 模块才能跳过。如果 caller 自带 `callbacks`，会被自动追加一个第二 callback。
**修法**: `create_llm(..., track_tokens: bool = True)` kwarg；False 时跳过注入。

### C6 · 测试 · `_extract_usage` 4 个分支全无覆盖
**文件**: `src/cryptotrader/llm/token_tracker.py:113`
callback tests 只覆盖 happy AIMessage path。以下均无测试：
- `LLMResult(generations=[])` 空列表
- 首个 generation 不是 AIMessage（走 `llm_output.token_usage` fallback path）
- `input_token_details.cache_read > 0` 设 cache_hit=True
- `on_chat_model_end` 别名分发

### C7 · 测试 · Decision Detail 响应新字段 0 集成测试
**文件**: `tests/test_api_decisions_detail.py`
这次新加的 6 个字段（`debate_turns` / `debate_gate` / `consensus_metrics` / `latency_breakdown` / `token_usage` / `bias`）**无一被 response shape 断言**。`_build_bias` 的 <3 agents 分支、severity 分级、summary 启发式也全未测。回归风险极高。

### C8 · 测试 · Risk 5 个 compute helper + 3 个 build helper 无单测
**文件**: `src/api/routes/risk.py`
`_build_correlation_groups` / `_known_pairs` / `_build_cooldowns` / `_build_recent_blocks` / `_compute_drawdown_pct` / `_compute_daily_loss_pct` / `_compute_total_exposure_pct` / `_compute_cvar_95` 全部只有路由层隐式覆盖。关键边界：`ttl == -1`（key 存在无过期）、sample < 30 → None、tail_count=1、空 portfolio，都未测。

---

## 🟠 Important 问题（28 个 — 优先级高）

### 安全（3）

**I-S1 · Binance URL 查询参数注入**（`market.py:39`）
`pair_symbol` 直接拼 f-string 进 Binance URL。`pair = "BTCUSDT&period=1m&limit=500"` 会改变请求语义。**修**: allowlist regex `^[A-Z0-9]{2,20}$` + 用 httpx `params=` 而非 f-string。

**I-S2 · ccxt 任意类实例化**（`market.py:121`）
`getattr(ccxt, exchange, None)` 无白名单。攻击者可指定任何 ccxt 模块属性触发实例化 + fetch_ohlcv（变相选择连接的 exchange host）。**修**: 硬编码 `ALLOWED_EXCHANGES = {'binance','okx','bybit'}`。

**I-S3 · API_KEY 未设时匿名通过**（`api/dependencies.py:14`）
`API_KEY=""` 时 `verify_api_key` 直接 return，所有路由公开。**修**: 启动时 `logger.warning` + 可选 `require_api_key=True` 硬拒绝默认值。

### 性能（4）

**I-P1 · Metrics route 两次 journal 扫描串行**（`metrics.py:414`）→ 5000 行反序列化 ×2。gather + 合并一次 `store.log(limit=3000)` 共享。

**I-P2 · `_flush_pending` O(N²) list.remove**（`store.py:329`）→ 用 `flushed_hashes` set + 列表推导。

**I-P3 · `_known_pairs` 第 4 次 `get_portfolio()`**（`risk.py:120`）→ 参数化传入。

**I-P4 · Portfolio 路由 3 × `_load_snapshots`**（见 C2）。

### 架构（8）

**I-A1 · `_coerce_timestamp` / `_metrics_coerce_ts` 字节相同**（`portfolio_v2.py:85` + `metrics.py`）→ 抽到 `api/routes/_utils.py`。

**I-A2 · `build_commit` 21 参数 god function**（`journal/commit.py:27`）→ 引入 `CommitObservability` 数据类打包 8 个观察性字段。

**I-A3 · `_LATENCY_STAGE_MAP` 隐式耦合**（`nodes/journal.py:13`）→ 节点 rename 会静默分类到 "other"。用 `@node_logger(stage='agents')` 自注册，或加合约测试。

**I-A4 · `latency_breakdown` / `token_usage` = `dict[str, Any]`**（`models.py:325`）→ 改 TypedDict，API Out model 复用。

**I-A5 · `MODEL_COSTS` 硬编码在 LLM 模块**（`token_tracker.py:25`）→ 移到 `config/default.toml` `[llm.model_costs]`；模型名会漂移（`claude-opus-4-7` 当前不是真实模型名）。

**I-A6 · `_fetch_long_short` 重复 Binance httpx pattern**（`market.py:28`）→ 应在 `data/providers/binance.py` 里加 `fetch_long_short_ratio()`。

**I-A7 · `_agent_kind_from_name` 后端 + `normalizeKind` 前端双实现**（`chat.py:50`）→ 后端暴露 `/api/chat/agent-kinds`，前端消费。

**I-A8 · Prometheus bucket wire 格式泄露到 API**（`metrics.py:95`）→ 用 `histogram_percentile()` 封装，API 只吐 float。

### 测试（9）
详见 C6/C7/C8 + 以下补充：
- `_compute_extras` 4 字段全无测试（`portfolio_v2.py`）
- `_fetch_long_short` 完全无测试（`market.py`）
- `_llm_accounting_last_24h` / `_cost_14d_series` 无直接单测（`metrics.py`）
- `_agent_kind_from_name` 无单测（`chat.py`）
- `_classify_move` 边界未测（confidence > 1.0, empty before_dir）
- `RedisStateManager.ttl` sentinel 值（-2, -1）未测
- `_snapshot_token_usage` 两分支未测

### 正确性（4）

**I-C1 · Token 前缀匹配顺序 Bug**（`token_tracker.py:49`）
`gpt-4o-mini-YYYY-MM-DD` startswith `gpt-4o`（在 dict 中先于 `gpt-4o-mini`）→ 成本按 $2.50/$10 计而非 $0.15/$0.60 → **17× 高估**。**修**: `sorted(MODEL_COSTS, key=len, reverse=True)` 或长前缀优先。

**I-C2 · SQLite 从不迁移新列**（`store.py:92`）
`create_all` 非迁移；ALTER TABLE 只在 postgresql 分支跑。既有 SQLite DB 永远没有 `latency_breakdown` / `token_usage` 列，读取走 `getattr` 默认 `{}`，静默丢数据。**修**: 加 sqlite `ALTER TABLE ADD COLUMN` 分支或用 Alembic。

**I-C3 · `cache_hit_rate` 可超过 1.0**（`metrics.py:339`）
Anthropic prompt cache 单次调用可能多段 cache_read，`cache_hits` 按段累加，能超过 `calls`。**修**: `min(1.0, cache_hits / calls)`。

**I-C4 · `g_resp.json()` 无 status_code 检查**（`market.py:57`）
Binance 429/400 → `.json()` 可能 raise JSONDecodeError。**修**: 加 `and g_resp.status_code == 200` 守卫。

---

## 🟡 Minor 问题（15 个 — 建议修但非阻塞）

### 安全
- **S-m1** `metrics/summary` 无缓存，可 DoS 放大 journal 扫描（`metrics.py:302`）→ 60s TTL 缓存。
- **S-m2** `response.response_metadata` 非 dict 时 `.get()` 抛 AttributeError 被 broad except 吞（`token_tracker.py:124`）→ `isinstance(meta, dict)` 守卫。

### 性能
- **P-m1** `_match_cost` prefix scan O(N=11) per LLM response（`token_tracker.py:49`）→ 结果 memoize。
- **P-m2** `debate_round` 构造 `others` dict per-agent（`debate.py:163`）→ N=4 可忽略。

### 架构
- **A-m1** `drawdown`（fraction）vs `drawdown_pct`（percentage）命名不一致（`portfolio_v2.py:44` vs `risk.py:69`）→ 统一后缀。
- **A-m2** `_deserialize` / `_row_to_dc` 两套几乎相同的重建逻辑（`store.py:154, 229`）→ 抽 `_build_dc()`。
- **A-m3** `_CORR_GROUPS` 在 risk route 硬编码，与 `risk/checks/correlation.py` 定义重复（`risk.py:83`）→ 从检查模块导入。

### 测试
- **T-m1** `_llm_accounting_last_24h` 用 `datetime.now(UTC)` 无注入点，24h 边界时间敏感（`metrics.py:311`）→ 加 `now=None` kwarg。
- **T-m2** AIMessage mock 未用 Anthropic-style `input_token_details.cache_read`（`test_token_tracker.py:110`）→ 加 Anthropic 形状测试。

### 正确性
- **C-m1** `_debate_status` 对无 gate 的旧 row 返回 `'skipped'`，不在前端契约中（`decisions.py:227`）→ 改 `'no-debate'`。
- **C-m2** `start_ledger()` 在 backtest 模式每步重置（`data.py:66`）→ 根据 `backtest_mode` 跳过。
- **C-m3** `_classify_move` 空 `before_dir` 生成 `'让步（由转bullish）'`（`debate.py:46`）→ `before_dir = before_dir or 'neutral'`。
- **C-m4** `create_llm` manifest 分支跳过 callback（`agents/base.py:172`）→ 注入到 `_try_manifest_llm` 或提前。
- **C-m5** `total_trades` 不按 OrderStatus 过滤，PENDING/CANCELLED 也计数（`portfolio_v2.py:169`）→ 加 `status != CANCELLED` 过滤。
- **C-m6** `daily_loss_pct` 盈利日返回负值（`risk.py:230`）→ 文档化契约或 `max(0.0, v)`。

---

## 🟢 Nice-to-have（3）

- **N1** ALTER TABLE 字符串插值虽当前安全，应加注释阻止未来引入外部 col 名（`store.py:99`）。
- **N2** `_compute_cvar_95` tail_count=1 边界（正好 20 样本）未测（`risk.py:258`）。
- **N3** `+Inf` bucket `upper_bound_s=1e12` 与前端 cumulative 语义易误解（`metrics.py:291`）→ 加 `LatencyHistogramBucketOut.count` 字段注释。

---

## 建议修复顺序

**Phase 1（必修，< 2h）**：
1. **I-C1 token prefix 顺序** — 成本数据即刻不准
2. **I-C2 SQLite 迁移** — 静默数据丢失
3. **C1+C2 Risk/Portfolio 并行化 + 复用 portfolio** — 3×-5× 延迟改善
4. **C3 bias 60s 缓存** — 每次 detail 少扫 1000 commits
5. **I-C3 cache_hit_rate 上界** — 简单一行修复
6. **I-C4 market status_code 检查** — 简单守卫

**Phase 2（结构性，半天）**：
7. **C5 create_llm track_tokens 参数** + **C-m4 manifest 分支注入** — 完整 token 追踪
8. **C4 ledger 初始化移出 collect_snapshot** — 解耦
9. **I-A1 抽 `_utils.coerce_timestamp`** + **I-A2 CommitObservability** — 降低修改成本
10. **I-A4 TypedDict** — IDE 体验 + 类型安全

**Phase 3（测试补全，1-2 天）**：
11. **C6/C7/C8 + Important 测试条目** — 防回归

**Phase 4（架构/安全加固）**：
12. **I-S1/I-S2/I-S3 安全三件套**
13. **I-A3 _LATENCY_STAGE_MAP 自注册**
14. **I-A5 MODEL_COSTS 入 config**

---

## 规格遵守度评估

vs brainstorm refined concept（9 scope items）：

| 需求 | 满足 | 偏差 |
|---|---|---|
| DB 扩 latency_breakdown + token_usage | ✅ Postgres | ⚠️ SQLite 漏迁移（I-C2） |
| Turn 结构持久化 | ✅ | — |
| Token tracker | ✅ | ⚠️ manifest path 漏注入（C-m4） + 价格匹配 bug（I-C1） |
| Portfolio 4 字段 | ✅ | ⚠️ total_trades 包含未成交订单（C-m5） |
| Decisions list 扩展 | ✅ | ⚠️ 旧 row debate_status='skipped' 不符合前端契约（C-m1） |
| Decisions detail 扩展（含 bias） | ✅ | — |
| Risk 4 meters + 三个列表 | ✅ | ⚠️ daily_loss_pct 负值语义未文档化（C-m6） |
| Metrics 6 字段 | ✅ | ⚠️ cache_hit_rate 可 > 1.0（I-C3） |
| Chat agent_message SSE | ✅ | — |
| Market 两比率 | ✅ | ⚠️ URL 注入 + 无 status check（I-S1, I-C4） |
| 0 mock | ✅ | — |

**符合度 ~85%**（功能全覆盖，质量/边界有瑕疵）。

---

## 下一步

1. 本报告是只读产物，未自动修复任何问题
2. 如需自动修，运行 `/spex:deep-review --auto-fix --stop-on-critical` 或单独对每项用 `/spex:review-code`
3. 建议先 Phase 1（6 项，< 2h），验证后再推 Phase 2/3

**报告路径**：`.kiro/specs/frontend-prototype-alignment/DEEP_REVIEW.md`
