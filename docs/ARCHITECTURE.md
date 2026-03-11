# CryptoTrader AI — 技术架构文档

---

## 1. 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | >=3.12 |
| 包管理 | uv | — |
| 构建系统 | Hatchling | — |
| LLM 编排 | LangChain + LangGraph | >=1.2.10, >=1.0.10 |
| LLM Provider | ChatOpenAI (兼容 API) | langchain-openai >=1.1.10 |
| 交易所连接 | ccxt | >=4.5.42 |
| 数据处理 | pandas + pandas-ta | >=2.3, >=0.4 |
| 数据验证 | Pydantic | >=2.12 |
| Web 框架 | FastAPI + Uvicorn | >=0.135 |
| 仪表板 | Streamlit | >=1.55 |
| CLI | Typer | >=0.24 |
| 数据库 | PostgreSQL 16 + SQLAlchemy | asyncpg >=0.29 |
| 缓存 / 状态 | Redis 7 | redis >=7.3 |
| 本地存储 | SQLite (数据 + LLM 缓存) | 内置 |
| 新闻 | feedparser (RSS) | >=6.0.12 |

## 2. 项目结构

```
cryptotrader-ai/
├── config/
│   ├── default.toml          # 主配置（执行模式、LLM 模型、风控参数）
│   ├── local.toml             # 本地覆盖（API keys，gitignored）
│   ├── risk.toml              # 风控参数
│   └── exchanges.toml.example # 交易所凭证模板
├── scripts/
│   └── run_backtest.py        # 高级回测脚本（比 CLI 更多参数）
├── src/
│   ├── cryptotrader/
│   │   ├── agents/            # 4 个专业 Agent
│   │   │   ├── base.py        # BaseAgent + ToolAgent + create_llm 工厂
│   │   │   ├── tech.py        # TechAgent（BaseAgent）
│   │   │   ├── chain.py       # ChainAgent（ToolAgent）
│   │   │   ├── news.py        # NewsAgent（ToolAgent）
│   │   │   ├── macro.py       # MacroAgent（BaseAgent）
│   │   │   ├── data_tools.py  # LangChain @tool 定义（6 Chain + 3 News）
│   │   │   ├── skills.py      # 渐进式知识加载
│   │   │   ├── tools.py       # 技能工具包装器
│   │   │   └── langchain_agents.py  # Supervisor 模式（备选）
│   │   ├── backtest/
│   │   │   ├── engine.py      # BacktestEngine（LLM + SMA 模式）
│   │   │   ├── cache.py       # OHLCV SQLite 缓存
│   │   │   ├── historical_data.py  # FnG/资金费率/BTC 主导率/FRED/期货量
│   │   │   └── result.py      # BacktestResult 数据类
│   │   ├── data/
│   │   │   ├── store.py       # 统一 SQLite 存储（61+ 数据源）
│   │   │   ├── snapshot.py    # SnapshotAggregator（数据聚合入口）
│   │   │   ├── market.py      # MarketCollector（ccxt OHLCV/Ticker）
│   │   │   ├── macro.py       # MacroCollector（FRED/CoinGecko/FnG/ETF）
│   │   │   ├── onchain.py     # OnchainCollector（5 个 provider 并行）
│   │   │   ├── news.py        # NewsCollector（RSS + 关键词情绪 + 社交）
│   │   │   ├── enhanced.py    # EnhancedDataProvider（OKX + 币安情绪）
│   │   │   ├── sync.py        # 批量历史同步（arena sync）
│   │   │   └── providers/     # 各数据源适配器
│   │   │       ├── binance.py
│   │   │       ├── coinglass.py
│   │   │       ├── cryptoquant.py
│   │   │       ├── defillama.py
│   │   │       ├── whale_alert.py
│   │   │       ├── sosovalue.py
│   │   │       └── rss_news.py
│   │   ├── debate/
│   │   │   ├── verdict.py     # AI Verdict（核心决策）
│   │   │   ├── researchers.py # Bull/Bear 对抗辩论
│   │   │   ├── challenge.py   # 交叉挑战 prompt 构建
│   │   │   └── convergence.py # 分歧度计算 + 收敛检测
│   │   ├── execution/
│   │   │   ├── simulator.py   # PaperExchange（模拟交易）
│   │   │   ├── exchange.py    # LiveExchange（实盘 ccxt）
│   │   │   └── order.py       # OrderManager（状态机）
│   │   ├── journal/
│   │   │   ├── store.py       # JournalStore（决策链）
│   │   │   ├── search.py      # 相似条件搜索
│   │   │   └── calibrate.py   # 偏差检测 + 校正生成
│   │   ├── learning/
│   │   │   ├── verbal.py      # 言语强化（历史经验注入）
│   │   │   └── reflect.py     # Agent 自我反思（LLM 策略备忘录）
│   │   ├── nodes/             # LangGraph 节点
│   │   │   ├── agents.py      # 4 Agent fan-out
│   │   │   ├── data.py        # 数据收集 + PnL 更新 + 趋势上下文
│   │   │   ├── debate.py      # 辩论轮次 + 收敛路由
│   │   │   ├── verdict.py     # Verdict + 风控检查
│   │   │   ├── execution.py   # 下单 + 止损 + 仓位更新
│   │   │   └── journal.py     # 日志记录
│   │   ├── portfolio/
│   │   │   └── manager.py     # PortfolioManager（仓位管理）
│   │   ├── risk/
│   │   │   ├── gate.py        # RiskGate（11 检查）
│   │   │   └── state.py       # RedisStateManager（状态 + 内存 fallback）
│   │   ├── graph.py           # 3 个图构建器
│   │   ├── state.py           # ArenaState TypedDict + build_initial_state
│   │   ├── models.py          # 所有数据模型
│   │   ├── config.py          # 配置加载 + 数据类
│   │   ├── db.py              # 共享 async DB session 工厂
│   │   └── graph_supervisor.py # Supervisor 图（备选架构）
│   ├── api/
│   │   ├── main.py            # FastAPI app（认证、限流、中间件）
│   │   └── routes/
│   │       ├── analyze.py     # POST /analyze
│   │       ├── health.py      # GET /health, /metrics
│   │       └── journal.py     # GET /journal/log, /journal/{hash}
│   ├── cli/
│   │   └── main.py            # Typer CLI（arena 命令）
│   └── dashboard/
│       └── app.py             # Streamlit 仪表板
├── tests/                     # 288 个测试，70% 覆盖率
├── pyproject.toml             # 项目配置 + 依赖
├── Makefile                   # 快捷命令
├── docker-compose.yml         # PostgreSQL 16 + Redis 7 + Scheduler 服务
└── CLAUDE.md                  # AI 编码指南
```

## 3. 数据模型

### 3.1 核心数据容器

```python
@dataclass
class DataSnapshot:
    """贯穿整个 pipeline 的中央数据容器"""
    timestamp: datetime
    pair: str                    # e.g. "BTC/USDT"
    market: MarketData           # OHLCV + ticker + funding + volatility
    onchain: OnchainData         # OI + 清算 + 鲸鱼 + DeFi TVL
    news: NewsSentiment          # 标题 + 情绪分数 + 关键事件
    macro: MacroData             # Fed/DXY/FnG/ETF/稳定币/哈希率
```

### 3.2 MarketData

```python
@dataclass
class MarketData:
    pair: str
    ohlcv: pd.DataFrame         # columns: timestamp, open, high, low, close, volume
    ticker: dict[str, Any]      # {"last": price, "baseVolume": vol}
    funding_rate: float          # 8h 资金费率
    orderbook_imbalance: float   # 买卖压力比
    volatility: float            # 收益率标准差
```

### 3.3 OnchainData

```python
@dataclass
class OnchainData:
    exchange_netflow: float = 0.0        # 正=流入(卖压) 负=流出(囤币)
    whale_transfers: list[dict] = []     # 大额转账列表
    open_interest: float = 0.0           # 未平仓合约
    liquidations_24h: dict = {}          # {volume_ratio, futures_volume, long_short_ratio,
                                         #  long_liquidations, short_liquidations}
    defi_tvl: float = 0.0               # DeFi 总锁仓量
    defi_tvl_change_7d: float = 0.0
    data_quality: dict[str, bool] = {}   # {has_oi, has_ls_ratio, has_etf, has_defi_tvl}
```

### 3.4 MacroData

```python
@dataclass
class MacroData:
    fed_rate: float = 0.0                # 联邦基金利率
    dxy: float = 0.0                     # 美元指数
    btc_dominance: float = 0.0           # BTC 市值占比 %
    fear_greed_index: int = 50           # 恐惧贪婪指数 0-100
    etf_daily_net_inflow: float = 0.0    # ETF 日净流入 $
    etf_total_net_assets: float = 0.0    # ETF 总净资产 $
    etf_cum_net_inflow: float = 0.0      # ETF 累计净流入 $
    vix: float = 0.0                     # 波动率指数
    sp500: float = 0.0                   # 标普 500
    stablecoin_total_supply: float = 0.0 # 稳定币总供应量
    btc_hashrate: float = 0.0           # 哈希率 GH/s
```

### 3.5 分析和决策模型

```python
@dataclass
class AgentAnalysis:
    agent_id: str                                      # "tech" | "chain" | "news" | "macro"
    pair: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float                                  # [0.0, 1.0]，data_sufficiency="low" 时 ≤ 0.3
    reasoning: str
    key_factors: list[str] = []
    risk_flags: list[str] = []
    data_points: dict[str, Any] = {}                   # 额外指标数据
    data_sufficiency: Literal["high", "medium", "low"] = "medium"
    is_mock: bool = False                              # LLM 调用失败时 True

@dataclass
class TradeVerdict:
    action: Literal["long", "short", "hold", "close"]
    confidence: float = 0.0
    position_scale: float = 0.0                        # [0.0, 1.0] → 直接控制仓位大小
    divergence: float = 0.0
    reasoning: str = ""
    thesis: str = ""                                   # 一句话交易论点
    invalidation: str = ""                             # 论点失效条件
```

### 3.6 执行和风控模型

```python
class OrderStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

# 合法状态转换（状态机）
VALID_TRANSITIONS = {
    PENDING: {SUBMITTED, CANCELLED, FAILED},
    SUBMITTED: {FILLED, PARTIALLY_FILLED, CANCELLED, FAILED},
    PARTIALLY_FILLED: {FILLED, CANCELLED},
}

@dataclass
class Order:
    pair: str
    side: Literal["buy", "sell"]
    amount: float                    # 必须 > 0
    price: float                     # 必须 >= 0
    order_type: Literal["market", "limit"] = "market"
    status: OrderStatus = PENDING

@dataclass
class DecisionCommit:
    """Git-like 不可变决策记录，形成链式结构"""
    hash: str                        # 16 位 hex
    parent_hash: str | None          # 前一条记录
    timestamp: datetime
    pair: str
    snapshot_summary: dict
    analyses: dict[str, AgentAnalysis]
    debate_rounds: int
    verdict: TradeVerdict | None
    risk_gate: GateResult | None
    order: Order | None
    pnl: float | None
    retrospective: str | None        # 事后分析文本
```

## 4. LangGraph 状态管理

```python
class ArenaState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]     # 追加式
    data: Annotated[dict[str, Any], merge_dicts]                 # 深度合并
    metadata: Annotated[dict[str, Any], merge_dicts]             # 深度合并
    debate_round: int
    max_debate_rounds: int
    divergence_scores: Annotated[list[float], operator.add]      # 追加式
```

### `data` 字典中的关键 key

| Key | 类型 | 何时设置 | 说明 |
|-----|------|----------|------|
| `snapshot` | `DataSnapshot` | collect_data | 市场快照 |
| `snapshot_summary` | `dict` | collect_data | 精简摘要 |
| `experience` | `str` | verbal_reinforcement | 历史经验文本 |
| `verdict_calibration` | `str` | verbal_reinforcement | 偏差校正文本 |
| `agent_reflections` | `dict[str, str]` | verbal_reinforcement | 各 Agent 策略备忘录 |
| `analyses` | `dict[str, dict]` | agent nodes | 4 个 Agent 分析结果 |
| `verdict` | `dict` | verdict node | 最终决策 |
| `risk_gate` | `dict` | risk_check | 风控结果 |
| `position_context` | `dict` | enrich_context / backtest | 当前持仓信息 |
| `trend_context` | `dict` | enrich_context | 价格趋势信息 |
| `order` | `dict` | place_order | 订单信息 |
| `stop_loss_triggered` | `bool` | check_stop_loss | 止损触发 |

### `metadata` 字典中的关键 key

| Key | 说明 |
|-----|------|
| `pair`, `engine`, `exchange_id` | 交易对、引擎、交易所 |
| `models` | `{tech_agent, chain_agent, news_agent, macro_agent}` 模型名 |
| `verdict_model`, `debate_model` | 特定阶段模型 |
| `llm_verdict` | `True` = AI 决策，`False` = 规则决策 |
| `backtest_mode` | `True` = ToolAgent 跳过实时 API |
| `debate_rounds` | 辩论轮数 |
| `cycle_count` | Scheduler 当前周期数（用于反思触发） |

`build_initial_state()` 工厂函数统一构造初始状态，确保 CLI/API/回测 三条路径一致。

## 5. LLM 架构

### 统一入口：`create_llm()` 工厂（`agents/base.py`）

```
所有 LLM 调用
      │
      ▼
create_llm(model, temperature, timeout, json_mode)
      │
      ├── 空模型 → 解析为 config.models.analysis 或 fallback
      ├── base_url / api_key → 从 config.llm 读取
      ├── streaming → 若 model ∈ config.llm.streaming_models
      ├── fallback → .with_fallbacks([fallback_llm])
      └── 缓存 → SQLiteCache at ~/.cryptotrader/llm_cache.db
```

### 10 个 LLM 调用点

| 位置 | 模型 | 温度 | 用途 |
|------|------|------|------|
| `BaseAgent.analyze()` | 各 agent 配置 | 0.2 | Agent 分析 |
| `ToolAgent.analyze()` | 各 agent 配置 | 0.2 | Agent + 工具循环 |
| `make_verdict_ai()` | verdict 模型 | 0.1 | 首席决策 |
| `run_debate()` Bull | debate 模型 | 0.3 | 牛方论证 |
| `run_debate()` Bear | debate 模型 | 0.3 | 熊方论证 |
| `judge_debate()` | debate 模型 | 0.1 | 法官裁决 |
| `debate_round()` | debate 模型 | 0.3 | 交叉辩论 |
| `langchain_agents.py` | supervisor 模型 | 0.2 | 备选 supervisor |
| `graph_supervisor.py` | supervisor 模型 | — | 备选 supervisor |
| `learning/reflect.py` | reflection 模型 | 0.3 | Agent 自我反思 |

**JSON 解析**：所有 LLM 输出经 `_extract_json()` 平衡括号提取，处理 markdown fence 和额外文本。

**deepseek-reasoner 兼容**：`extract_content()` 处理推理内容在 `additional_kwargs['reasoning_content']` 中的情况。

## 6. 数据层

### 6.1 统一 SQLite 存储

**路径**：`~/.cryptotrader/market_data.db`

**表结构**：
- `market_data (source TEXT, date TEXT, data TEXT, updated_at REAL) PK(source, date)` — JSON 格式存储所有时序数据
- `fetch_log (source TEXT PK, last_fetch REAL)` — 按源的速率限制记录

**61+ 数据源**，7 个类别：宏观、链上、衍生品、DeFi、情绪、ETF、稳定币

**速率限制**（秒）：

| 源 | TTL | 说明 |
|----|-----|------|
| `fred_DFF`, `fred_DTWEXBGS` | 3600 | 各自独立，不共享 |
| `sosovalue_etf_metrics` | 300 | 5 分钟 |
| `live_news_rss`, `live_social_buzz` | 600 | 10 分钟 |
| `binance_derivatives`, `defillama` | 3600 | 1 小时 |

### 6.2 OHLCV 缓存

**路径**：`~/.cryptotrader/ohlcv_cache.db`

**表**：`ohlcv (pair, timeframe, ts, o, h, l, c, v) PK(pair, timeframe, ts)`

### 6.3 历史数据（回测用，与 OHLCV 缓存同库）

| 表 | 源 |
|----|-----|
| `fear_greed (date PK, value, classification)` | alternative.me |
| `funding_rate (date PK, rate, count)` | Binance 8h 聚合 |
| `btc_dominance (date PK, dominance)` | CoinGecko 估算 |
| `fred_series (series_id, date PK, value)` | FRED API |
| `futures_volume (date PK, volume, quote_volume)` | Binance 期货 |

### 6.4 数据提供者

| Provider | API | 认证 | 数据 |
|----------|-----|------|------|
| Binance Futures | `fapi.binance.com` | 免费 | OI/多空比/资金费率/期货量 |
| CoinGlass | `open-api.coinglass.com` | API Key | OI/清算 |
| CryptoQuant | `api.cryptoquant.com` | Bearer Token | 交易所净流量 |
| DefiLlama | `api.llama.fi` | 免费 | TVL/稳定币供应 |
| Whale Alert | `api.whale-alert.io` | API Key | 大额转账 |
| SoSoValue | `api.sosovalue.xyz` | API Key | ETF 指标/历史/新闻 |
| FRED | `api.stlouisfed.org` | API Key | 联邦利率/DXY/VIX/S&P500 |
| CoinGecko | `api.coingecko.com` | 免费 | BTC 主导率/社交数据 |
| alternative.me | `api.alternative.me` | 免费 | 恐惧贪婪指数 |
| RSS | CoinDesk/CoinTelegraph/Decrypt | 免费 | 新闻标题 |

### 6.5 数据同步（arena sync）

`sync_all()` 批量拉取历史数据到 SQLite 存储：

| 函数 | 存储 key | 前向填充 |
|------|----------|----------|
| `sync_fred_series(DFF/DTWEXBGS/VIXCLS/SP500)` | `fred_*` | Yes |
| `sync_sosovalue_etf_history` | `sosovalue_etf` | No |
| `sync_fear_greed_history` | `fear_greed` | No |
| `sync_binance_oi` | `binance_oi_{symbol}` | No |
| `sync_binance_ls_ratio` | `binance_ls_ratio_{symbol}` | No |
| `sync_defillama_tvl` | `defillama_tvl` | No |
| `sync_stablecoin_supply` | `stablecoin_total_supply` | No |
| `sync_btc_hashrate` | `btc_hashrate` | No |
| `sync_news_headlines` | `news_headlines_{symbol}` | No |

## 7. 执行层

### 7.1 PaperExchange（模拟）

- 初始余额：`config.backtest.initial_capital`（默认 $10,000）
- 滑点模型：`slippage_base + amount × price × 1e-8`
- `asyncio.Lock` 保证线程安全
- 余额不足返回 `status: "failed"`，不抛异常

### 7.2 LiveExchange（实盘）

- 基于 ccxt async，`enableRateLimit=True`
- 精度处理：`amount_to_precision` / `price_to_precision`
- 最小订单量检查
- 重试机制：指数退避 3 次（1s, 2s, 4s）
- 订单超时：30 秒轮询，超时自动取消

### 7.3 订单状态机

```
PENDING → SUBMITTED → FILLED
                    → PARTIALLY_FILLED → FILLED
                                       → CANCELLED
                    → CANCELLED
                    → FAILED
```

## 8. 决策日志

**PostgreSQL 表 `decision_commits`**：

| 列 | 类型 | 说明 |
|----|------|------|
| hash | String(16) PK | 短哈希标识 |
| parent_hash | String(16) | 形成链式结构 |
| timestamp | DateTime, indexed | 决策时间 |
| pair | String(20), indexed | 交易对 |
| snapshot_summary | JSONB | 市场快照摘要 |
| analyses | JSONB | 4 Agent 分析结果 |
| debate_rounds | Integer | 辩论轮数 |
| verdict | JSONB | AI 决策 |
| risk_gate | JSONB | 风控结果 |
| order_data | JSONB | 订单详情 |
| pnl | Float | 已实现损益 |
| retrospective | Text | 事后复盘 |

**Fallback**：无数据库时使用内存列表（`_MAX_MEMORY = 10,000` 条上限）。

## 9. 入口和接口

### 9.1 CLI 命令（arena）

```bash
# 核心操作
arena run --pair BTC/USDT --mode paper          # 单次分析+执行
arena run --pair BTC/USDT ETH/USDT --graph full # 多对，完整图
arena backtest --pair BTC/USDT --start 2024-01-01 --end 2024-06-01 --interval 4h
arena sync                                       # 同步所有历史数据

# 日志查看
arena journal log --limit 10                     # 最近 10 条决策
arena journal show abc123                        # 单条详情

# 服务
arena serve --port 8003                          # FastAPI 服务器
arena dashboard                                  # Streamlit 仪表板
arena scheduler start                            # APScheduler 定时调度（需 scheduler.enabled=true）
arena scheduler status                           # 组合状态

# 维护
arena migrate                                    # 创建 PostgreSQL 表
arena risk reset-breaker                         # 重置熔断器
arena live-check                                 # 实盘就绪检查
```

### 9.2 FastAPI 端点

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| GET | `/health` | 公开 | 健康检查（API/Redis/DB 状态） |
| GET | `/metrics` | Auth | 指标（JSON 或 Prometheus 格式） |
| POST | `/analyze` | Auth | 触发一次完整分析 |
| GET | `/journal/log` | Auth | 最近决策列表 |
| GET | `/journal/{hash}` | Auth | 单条决策详情 |
| GET | `/portfolio` | Auth | 组合状态 |
| GET | `/risk/status` | Auth | 风控状态（交易计数、熔断） |

认证：`X-API-Key` header（仅当设置 `API_KEY` 环境变量时启用）
限流：60 次/分钟/IP（health 和 metrics 豁免）

### 9.3 高级回测脚本（`scripts/run_backtest.py`）

比 `arena backtest` 更多高级参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--pair` | BTC/USDT | 交易对 |
| `--model` | "" | 覆盖所有 agent 的 LLM 模型 |
| `--start/--end` | 2025-06-01/2025-12-31 | 回测区间 |
| `--stop-loss` | 0.08 | 灾难止损 (8%) |
| `--trailing-stop` | 0 | 追踪止损 (0=禁用) |
| `--reversal-days` | 3 | 连续反向信号天数触发退出 |
| `--drawdown-pause` | 0.10 | 账户回撤暂停交易 (10%) |
| `--min-reentry-days` | 3 | AI 关仓后重入冷却期 |
| `--atr-sizing` | False | ATR 自适应仓位 |
| `--debate` | False | 使用牛熊辩论图 |
| `--debate-rounds` | 2 | 辩论轮数 |

**run_backtest.py vs BacktestEngine 对比**：

| 特性 | run_backtest.py | BacktestEngine |
|------|-----------------|----------------|
| 间隔 | 1d（日线） | 4h（可配） |
| 灾难止损 | Yes (可配) | No |
| 追踪止损 | Yes | No |
| 反转退出 | Yes | No |
| 回撤暂停 | Yes | No |
| 重入冷却 | Yes | No |
| 最小持仓天数 | Yes (5天) | No |
| ADX 过滤 | Yes | No |
| 辩论图支持 | Yes | No (仅 lite) |
| Verdict 记忆 | Yes | No |
| position_scale 连续映射 | Yes | Yes (3 档) |

### 9.4 Streamlit 仪表板

| 页面 | 内容 |
|------|------|
| Overview | 总价值/日损益/回撤指标 + 权益曲线 + 持仓表 |
| Decisions | 最近 20 条决策展开卡片（风控/verdict/辩论/PnL） |
| Risk Status | Redis 状态（交易计数/熔断） + 重置按钮 + 风控参数表 |
| Backtest | 回测表单 → 运行 → 收益/Sharpe/MaxDD/胜率 + 权益曲线 + 交易表 |

## 10. 配置系统

### 配置结构

```toml
# config/default.toml

[app]
mode = "standalone"    # standalone | api
engine = "paper"       # paper | live

[llm]
api_key = ""           # 统一 API Key
base_url = ""          # 统一 API 端点

[models]               # 每个角色可独立配置模型
analysis = "deepseek-chat"
debate = "deepseek-chat"
verdict = "deepseek-reasoner"
tech_agent = "deepseek-reasoner"
chain_agent = "deepseek-chat"
news_agent = "deepseek-chat"
macro_agent = "deepseek-chat"
fallback = "deepseek-chat"

[debate]
max_rounds = 3
convergence_threshold = 0.1
divergence_hold_threshold = 0.7

[risk]
max_stop_loss_pct = 0.05

[risk.position]
max_single_pct = 0.10
max_total_exposure_pct = 0.50

[risk.loss]
max_daily_loss_pct = 0.03
max_drawdown_pct = 0.10
max_cvar_95 = 0.05

[risk.cooldown]
same_pair_minutes = 60
post_loss_minutes = 120

[backtest]
initial_capital = 10000
slippage_base = 0.0005

[backtest.position_sizing]
high_confidence_pct = 0.35    # position_scale 映射天花板
medium_confidence_pct = 0.12
low_confidence_pct = 0.06     # position_scale 映射地板

[scheduler]
enabled = false
pairs = ["BTC/USDT", "ETH/USDT"]
interval_minutes = 240
exchange_id = "binance"
daily_summary_hour = 0    # UTC hour for daily summary (0-23)

[providers]               # 各数据源 API Key + 开关
# ... 20+ 配置项
```

**加载逻辑**：
1. 加载 `config/default.toml`
2. 深度合并 `config/local.toml`（gitignored，存放 API keys）
3. 首次加载后全局缓存（`_cached_config`），运行中不可变
