# Agent 反思优化系统 — 设计文档

> 每个 Agent 定期回顾自己的历史分析 + 实际结果，通过 LLM 生成策略备忘录，实现自我优化。

---

## 1. 问题背景

### 1.1 现有学习机制

系统已有两个学习层：

| 机制 | 文件 | 作用 | 局限 |
|------|------|------|------|
| 言语强化 | `learning/verbal.py` | 搜索 3 条相似历史条件，共享经验注入 | 所有 Agent 共享相同经验，无法个性化 |
| 统计校准 | `journal/calibrate.py` | 检测过度自信、方向偏好、准确率，生成校准警告 | 只能告诉 Agent "你有 bullish bias"，无法告诉它"你在低波动时期持续误判 RSI 超卖信号" |

### 1.2 缺失能力

- **没有 LLM 驱动的深度反思** -- Agent 无法回顾"我当时为什么判断错了"、"哪些信号实际有效"
- **校准只有定量统计，没有定性分析** -- 知道准确率 40%，但不知道错在哪里
- **经验是共享的，不是个性化的** -- Tech Agent 和 Macro Agent 看到同样的历史经验

### 1.3 目标

每个 Agent 定期回顾自己的历史分析 + 实际 PnL 结果，通过 LLM 生成**策略备忘录**（strategy memo），在后续分析中自动注入，实现自我优化。

---

## 2. 系统设计

### 2.1 核心思路

每 N 个交易周期（默认 20，即 ~3.3 天 @ 4h 间隔），对 4 个 Agent 各执行一次 LLM 反思调用：

- **输入**：该 Agent 最近 30 条有 PnL 结果的历史分析（direction, confidence, reasoning, key_factors + 实际 PnL + 当时市场环境）
- **输出**：3-5 条策略备忘录（哪些信号有效、哪些误导、什么偏差需要纠正）
- **持久化**：SQLite `~/.cryptotrader/agent_reflections.db`（复用现有 `~/.cryptotrader/` 模式）
- **注入**：下次分析时作为 `"Strategy memo (your own prior self-reflection):"` 追加到 experience 末尾

### 2.2 数据流

```
verbal_reinforcement 节点:
  1. get_experience()                    -- 共享历史经验（不变）
  2. detect_biases() -> corrections      -- 统计校准（不变）
  3. load_reflections()                  -- [新增] 从 SQLite 读取策略备忘录
  4. asyncio.ensure_future(              -- [新增] 后台触发 LLM 反思
       maybe_reflect(cycle_count)          （fire-and-forget，不阻塞交易周期）
     )

_run_agent() 注入（在 agents.py 中）:
  experience = verbal_experience
  + "\n\n" + bias_correction            -- 统计校准（不变）
  + "\n\n" + reflection_memo            -- [新增] 策略备忘录
```

### 2.3 在完整 pipeline 中的位置

```
                          交易管线
                            |
                            v
                       collect_data
                            |
                            v
                       update_pnl
                            |
                            v
                    verbal_reinforcement
                      |            |
           [不变]     |            |  [新增]
         experience   |            |  load_reflections()  <-- 快速 SQLite 读取
         corrections  |            |  maybe_reflect()     <-- 后台 LLM (不阻塞)
                      |            |
                      v            v
              +------ 注入到 Agent prompt ------+
              |                                  |
              v                                  v
      "Historical similar..."          "Strategy memo (your own
       + 校准警告                        prior self-reflection):
                                         1. RSI oversold + 低波动 = 误导..."
              |                                  |
              +----------------------------------+
                            |
                            v
                    4 Agent 并行分析
                            |
                            v
                        辩论 -> Verdict -> 风控 -> 执行
```

---

## 3. 反思 Prompt 设计

### 3.1 领域专属上下文

每个 Agent 看到自己领域的关键信号，引导它针对性地反思：

| Agent | 关键信号 |
|-------|----------|
| Tech | RSI（超卖<30, 超买>70）、MACD 交叉、SMA20/60 交叉、布林带宽度、ATR |
| Chain | funding rate 极端值、交易所净流量、鲸鱼转账、OI 变化、清算接近度 |
| News | 新闻情绪分数极端值、重大事件识别（监管、ETF、交易所事故）、噪音 vs 信号 |
| Macro | 联储利率变化、DXY 趋势、恐贪指数极端值、ETF 日流入>$200M、VIX 飙升 |

### 3.2 Prompt 结构

```
系统消息：
  你是 [领域] Agent。你正在回顾自己过去的分析记录和实际结果，
  以生成策略备忘录来改进未来的判断。

用户消息：
  你是 [领域] Agent。回顾你最近 [N] 次有结果的分析。

  [领域专属提示]

  你的历史分析：
  ---
  [日期] | direction=bullish confidence=0.75 | 结果: pnl=+2.30 | verdict=long
    Reasoning: RSI oversold at 28, MACD histogram turning positive...
    Key factors: [RSI超卖, MACD金叉]
    市场环境: price=95000 volatility=0.0320 funding=0.00150
  ---
  [...N 条...]

  任务：写一份简洁的策略备忘录（3-5 条要点），你的未来自己会在每次分析前阅读。
  1. 你领域中哪些信号在最近的分析中最有预测力？
  2. 哪些信号具有误导性或导致了错误判断？举具体例子。
  3. 你有什么系统性偏差需要纠正？
  4. 一条具体的规则或阈值调整，应用于下次分析。

  基于上方历史数据给出具体建议，不要泛泛而谈。
  输出纯文本，不需要 JSON。
```

### 3.3 LLM 参数

- **模型**：`config.reflection.model`（默认空 = 使用 `config.models.analysis`）
- **温度**：0.3（平衡创造性和稳定性）
- **超时**：120 秒
- **通过 `create_llm()` 工厂调用**，自动获得 fallback 和缓存

---

## 4. 持久化

### 4.1 SQLite Schema

**路径**：`~/.cryptotrader/agent_reflections.db`

```sql
CREATE TABLE IF NOT EXISTS agent_reflections (
    agent_id   TEXT PRIMARY KEY,     -- "tech_agent", "chain_agent", ...
    memo       TEXT NOT NULL,        -- 策略备忘录全文
    updated_at TEXT NOT NULL         -- ISO 格式时间戳
)
```

### 4.2 读写模式

- **读**：每个交易周期执行一次 `load_reflections()`，快速全表扫描（最多 4 行）
- **写**：每 N 个周期执行一次 `save_reflection()`，`INSERT OR REPLACE` 原子操作
- **覆盖策略**：每次反思覆盖上一次备忘录，不累积。避免 prompt 膨胀，保持备忘录始终基于最新数据
- **异步包装**：`sqlite3` 同步操作通过 `asyncio.to_thread()` 包装，不阻塞事件循环

---

## 5. 配置

### 5.1 TOML 配置

```toml
# config/default.toml
[reflection]
enabled = true              # 是否启用反思系统
every_n_cycles = 20          # 每 N 个交易周期执行一次反思
min_commits_required = 10    # 至少需要多少条有 PnL 的 commit 才执行
lookback_commits = 30        # 从 journal 取最近多少条 commit
model = ""                   # 空 = 用 models.analysis
```

### 5.2 数据类

```python
@dataclass
class ReflectionConfig:
    enabled: bool = True
    every_n_cycles: int = 20
    min_commits_required: int = 10
    lookback_commits: int = 30
    model: str = ""  # 空 = use models.analysis
```

---

## 6. 触发与执行

### 6.1 触发条件

`maybe_reflect()` 在每个交易周期的 `verbal_reinforcement` 节点中被调用，但只在满足以下条件时真正执行：

1. `config.reflection.enabled == True`
2. `cycle_count > 0`（跳过首次运行）
3. `cycle_count % every_n_cycles == 0`
4. journal 中有 PnL 的 commit >= `min_commits_required`
5. 每个 Agent 至少有 3 条分析记录

### 6.2 执行模式

**后台 fire-and-forget**：

```python
task = asyncio.ensure_future(maybe_reflect(store, cycle_count, config.reflection))
task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
```

- 4 个 Agent x LLM 调用可能需要 1-2 分钟，不能阻塞交易周期
- 下一个周期自动加载新的反思结果
- 一个周期的延迟完全可接受

### 6.3 cycle_count 传递

Scheduler 将当前周期数通过 `extra_metadata` 注入初始状态：

```python
initial = build_initial_state(
    pair,
    engine=config.engine,
    exchange_id=config.scheduler.exchange_id,
    config=config,
    extra_metadata={"cycle_count": self._cycle_count},
)
```

非 Scheduler 模式（CLI `arena run`）中 `cycle_count` 默认为 0，不会触发反思。

---

## 7. 与现有系统的关系

### 7.1 三层学习机制对比

```
层次 1: 言语强化 (verbal.py)
  ├── 类型：定量相似性匹配
  ├── 粒度：全局共享
  ├── 频率：每个周期
  └── 内容："在相似市场条件下，过去我们做多赚了 $320"

层次 2: 统计校准 (calibrate.py)
  ├── 类型：定量统计偏差检测
  ├── 粒度：按 Agent
  ├── 频率：每个周期
  └── 内容："你有 70% 做多倾向，过度自信（错误时平均 0.72 置信度）"

层次 3: 自我反思 (reflect.py) [新增]
  ├── 类型：定性 LLM 深度分析
  ├── 粒度：按 Agent + 按领域
  ├── 频率：每 20 个周期
  └── 内容："低波动时 RSI 超卖信号不可靠（3 次中 2 次误判），
            funding rate 反转才是更可靠的指标"
```

三层互补，不冲突：
- 层次 1 提供"发生了什么"
- 层次 2 提供"统计上你偏了多少"
- 层次 3 提供"为什么你会判断错，下次该怎么做"

### 7.2 注入顺序

Agent 最终收到的 experience 文本结构：

```
Historical similar conditions:           <-- verbal.py (层次 1)
  - BTC @ 2024-11-15: verdict=long, pnl=+$320
    Lesson: strong trend continuation was correct

Calibration warnings (YOUR track record): <-- calibrate.py (层次 2)
  OVERCONFIDENT -- avg confidence on wrong calls is 72%. Lower your confidence...

Strategy memo (your own prior self-reflection):  <-- reflect.py (层次 3)
  1. RSI oversold + 低波动 = 经常误导，3 次中 2 次判断错误
  2. MACD 交叉在日线级别最可靠，1h 级别噪音太多
  3. 我有在高波动时过度看空的倾向，应关注支撑位反弹
  4. 规则：波动率 < 0.02 时，RSI 超卖权重降低 50%
```

---

## 8. 文件清单

### 8.1 新建文件

| 文件 | 行数 | 作用 |
|------|------|------|
| `src/cryptotrader/learning/reflect.py` | ~180 | 反思系统核心：加载/保存/触发/执行 |
| `tests/test_reflect.py` | ~230 | 12 个测试覆盖所有功能路径 |

### 8.2 修改文件

| 文件 | 改动 |
|------|------|
| `src/cryptotrader/config.py` | 新增 `ReflectionConfig` 数据类，注册到 `AppConfig` 和 `_build_config()` |
| `config/default.toml` | 新增 `[reflection]` 配置段 |
| `src/cryptotrader/nodes/data.py` | `verbal_reinforcement()` 集成反思加载 + 后台触发 |
| `src/cryptotrader/nodes/agents.py` | `_run_agent()` 在 bias correction 后追加反思备忘录注入 |
| `src/cryptotrader/scheduler.py` | `_run_pair()` 通过 `extra_metadata` 传递 `cycle_count` |
| `pyproject.toml` | RUF001 per-file-ignore（reflect.py 中的中文 LLM prompt） |

### 8.3 公开 API

```python
# reflect.py 公开函数

async def load_reflections(db_path: Path | None = None) -> dict[str, str]
    """从 SQLite 读取所有 Agent 的策略备忘录。返回 {agent_id: memo_text}"""

async def save_reflection(db_path: Path, agent_id: str, memo: str) -> None
    """持久化一个 Agent 的反思结果（upsert）"""

async def maybe_reflect(
    store: JournalStore,
    cycle_count: int,
    config: ReflectionConfig,
    db_path: Path | None = None,
) -> dict[str, str]
    """检查是否到了反思周期。如果是，执行 LLM 反思并返回更新的备忘录"""

async def run_agent_reflection(
    agent_id: str,
    records: list[dict],
    model: str,
) -> str
    """对单个 Agent 执行 LLM 反思调用，返回备忘录文本"""
```

---

## 9. 设计决策

| 决策 | 理由 |
|------|------|
| **后台执行反思（fire-and-forget）** | 4 个 Agent x LLM 调用可能需要 1-2 分钟，不能阻塞交易周期。下一周期自动加载结果，一个周期的延迟可接受 |
| **SQLite 而非 PostgreSQL** | 复用 `~/.cryptotrader/` 模式（同 llm_cache.db, market_data.db）。零迁移成本，重启保活。并发安全（REPLACE INTO 原子操作） |
| **滚动覆盖而非累积** | 每次反思覆盖上一次备忘录。避免 prompt 膨胀，保持备忘录始终基于最新数据 |
| **保留 calibrate.py 不变** | calibrate 做定量统计（准确率、偏差率），reflect 做定性分析（为什么错、什么信号有效）。两者互补 |
| **agent_type 键一致** | 反思结果用 `"tech_agent"` 等键存储，与 calibrate.py 和 nodes/agents.py 键名一致 |
| **asyncio.to_thread 包装 sqlite3** | 不引入新依赖（aiosqlite），SQLite 操作极快（<1ms），`to_thread` 足够 |
| **反思不在 backtest 中运行** | CLI `arena run` 的 cycle_count 为 0，不触发反思。反思只在 Scheduler 长期运行时有意义 |

---

## 10. 测试

### 10.1 测试覆盖

```bash
uv run pytest tests/test_reflect.py -v
```

| 测试 | 验证内容 |
|------|----------|
| `test_load_save_reflections` | 写入 -> 读取 -> 内容匹配 |
| `test_load_reflections_empty` | 不存在的 DB 返回空 dict |
| `test_save_reflection_upsert` | 第二次写入覆盖第一次 |
| `test_format_commit_for_agent` | 正确提取指定 Agent 的分析 |
| `test_format_commit_missing_agent` | Agent 不在 commit 中返回 None |
| `test_build_reflection_prompt` | prompt 包含领域上下文和历史记录 |
| `test_maybe_reflect_skips_when_not_due` | cycle_count 不整除时跳过 |
| `test_maybe_reflect_skips_when_disabled` | 禁用时跳过 |
| `test_maybe_reflect_skips_insufficient_data` | commit 不足时跳过 |
| `test_maybe_reflect_runs_and_saves` | mock LLM -> 验证 SQLite 写入 |
| `test_maybe_reflect_partial_failure` | 单个 Agent 失败不影响其他 |
| `test_reflection_injected_in_experience` | `_run_agent()` 正确注入备忘录 |

### 10.2 手动验证

```bash
# 运行 20+ 个周期积累数据后：
ls ~/.cryptotrader/agent_reflections.db

sqlite3 ~/.cryptotrader/agent_reflections.db \
  "SELECT agent_id, length(memo), updated_at FROM agent_reflections;"
# 应看到 4 行（tech_agent, chain_agent, news_agent, macro_agent），memo 非空
```

---

## 11. 未来扩展（不在当前范围内）

- **按 pair 分别反思**：不同交易对可能需要不同策略，当前是全局反思
- **反思历史保留**：保留历史备忘录用于趋势分析（当前是覆盖式）
- **反思质量评估**：对比反思前后的准确率变化，验证反思有效性
- **用户可读反思报告**：在 dashboard 中展示每个 Agent 的当前策略备忘录
