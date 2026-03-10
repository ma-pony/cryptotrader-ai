# CryptoTrader AI

基于 LangGraph 多智能体辩论的 AI 加密货币交易系统。

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-288%20passed-brightgreen.svg)]()

## 概述

4 个专业 AI 智能体（技术面、链上数据、新闻情绪、宏观经济）独立分析市场数据，然后通过交叉挑战辩论轮次达成共识。硬编码的风控门（11 项规则检查，不依赖 LLM）强制执行仓位限制、损失限制和熔断机制。每个决策都记录在类 Git 的决策日志中，便于审计和基于经验的学习。

每个智能体在输出信号前都会运行**领域专属的预信号检查清单**（受 Devin 的先思后行模式启发），以减少过度自信和幻觉。

### 核心特性

- **多智能体辩论** — 4 个智能体独立分析后，经 2-3 轮交叉质疑达成共识
- **三种图模式** — 完整辩论流程、轻量回测、多空对抗+评委
- **11 项风控检查** — 纯规则，零 LLM：仓位限制、CVaR、相关性、熔断器
- **决策日志链** — 类 Git 不可变提交链，支持相似搜索和校准分析
- **语言强化** — 将历史决策经验注入智能体 prompt，实现经验学习
- **智能体自我反思** — 定期 LLM 驱动的策略备忘录，基于历史准确率分析
- **回测引擎** — 历史模拟，含真实成本建模和防前视偏差
- **实盘就绪** — 基于 ccxt 的交易所适配器，带重试、精度处理和超时控制
- **APScheduler 自动化** — 周期性交易循环 + 每日组合摘要
- **61+ 数据源** — 统一 SQLite 存储，覆盖 7 个类别，按源独立限速

## 系统架构

```
数据采集 → 语言强化注入 → 4 智能体（并行）
  → 交叉挑战辩论（2-3 轮）→ 收敛检查
  → 裁决 → 风控门（11 项检查）→ 执行 / 拒绝 → 日志记录
                                    ↓
                          持仓回写 → 快照保存
```

**三种图模式：**
- `build_trading_graph()` — 完整流程，含辩论循环和收敛检查
- `build_lite_graph()` — 跳过辩论，用于回测
- `build_debate_graph()` — 多空对抗辩论 + 评委（TradingAgents 风格）

### 智能体分工

| 智能体 | 类型 | 数据 | 职责 |
|--------|------|------|------|
| TechAgent | BaseAgent | OHLCV + pandas-ta 指标（RSI、MACD、SMA、BBands、ATR）| 技术形态识别 |
| ChainAgent | ToolAgent | OI、资金费率、交易所净流量、鲸鱼转账、DeFi TVL | 链上信号检测 |
| NewsAgent | ToolAgent | RSS 标题 + FinBERT 情绪 + CoinGecko 社交热度 | 新闻情绪分析 |
| MacroAgent | BaseAgent | 利率、美元指数、BTC 主导率、恐惧贪婪、ETF 流量、VIX | 宏观环境评估 |

- **BaseAgent**：单次 LLM 调用，结构化 JSON 输出
- **ToolAgent**：LangChain 代理，带工具调用循环，可实时查询数据（回测模式下降级为单次调用，避免前视偏差）

每个智能体的系统提示都包含 **5 点预信号检查清单**：矛盾检查、证据落地、信心合理性、基准概率意识、近因偏差防范。`data_sufficiency="low"` 时信心值上限为 0.3。

### 辩论流程

1. **第 1 轮**：4 个智能体独立分析
2. **第 2-3 轮**：每个智能体看到其他人的分析结果，必须用具体数据点支持自己保持或修改立场
3. **收敛检查**：每轮计算分歧度（`confidence × direction` 的总体标准差），相对变化 < 10% 或达到最大轮数时停止
4. **裁决**：单个 LLM（temperature=0.1）综合所有智能体输出、持仓上下文（空仓/多头/空头、入场价、浮动盈亏）、价格趋势和风控约束 → 输出 `{action, confidence, position_scale, reasoning, thesis, invalidation}`

### 学习系统

- **语言强化**：`search_similar()` 在日志中查找最多 3 条匹配当前资金费率、波动率和价格趋势方向的历史决策（50% 容差），作为"过往经验"注入 prompt
- **智能体反思**：每 N 个周期（默认 20），每个智能体撰写 3-5 点策略备忘录，分析哪些信号有效、哪些误导。备忘录持久化到 SQLite，在后续分析中注入系统提示
- **校准分析**：逐智能体准确率追踪 + 偏差检测（过度自信、方向偏好、中性默认）。校正信息注入裁决 prompt

## 快速开始

### 前置条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 包管理器
- 一个兼容 OpenAI 的 LLM API Key（OpenAI、Anthropic、DeepSeek 等）

### 安装

```bash
# 克隆并安装
git clone https://github.com/your-org/cryptotrader-ai.git
cd cryptotrader-ai
uv sync

# 配置 LLM 端点
cp config/default.toml config/local.toml
# 编辑 config/local.toml：设置 [llm] api_key 和 base_url

# 或使用环境变量
export OPENAI_API_KEY=your_key
```

### 首次运行

```bash
# 运行单次分析（模拟交易）
arena run --pair BTC/USDT --mode paper

# 多交易对分析（完整辩论）
arena run --pair BTC/USDT --pair ETH/USDT --graph full

# 查看决策日志
arena journal log --limit 10
arena journal show <hash>
```

### 回测

```bash
# AI 智能体回测
arena backtest --pair BTC/USDT --start 2024-01-01 --end 2024-06-01 --interval 4h

# 快速 SMA 交叉回测（无 LLM 调用）
arena backtest --pair BTC/USDT --start 2024-01-01 --end 2024-06-01 --no-llm

# 先同步历史数据，回测数据更丰富
arena sync
```

回测引擎特性：
- **防前视偏差**：bar[i] 生成信号，bar[i+1] 开盘执行
- **真实成本**：可配置滑点（5 bps）+ 手续费（10 bps）
- **动态仓位**：高信心 35%、中等 12%、低 6%
- **丰富数据**：ETF 流量、OI、多空比、DeFi TVL、VIX、S&P500、稳定币供应、哈希率
- **指标输出**：总收益、夏普比率（365 天年化）、最大回撤、胜率、权益曲线

### 调度器

```bash
# 启动周期性交易循环（需 config 中 scheduler.enabled=true）
arena scheduler start

# 查看组合状态
arena scheduler status
```

基于 APScheduler，`IntervalTrigger`（默认 4 小时）执行交易循环，`CronTrigger` 发送每日组合摘要。

### 仪表盘 & API

```bash
# Streamlit 仪表盘
arena dashboard

# FastAPI 服务
arena serve --port 8003
```

## CLI 命令参考

| 命令 | 说明 |
|------|------|
| `arena run --pair BTC/USDT --mode paper` | 单次分析 + 执行 |
| `arena run --pair BTC/USDT --graph full\|lite\|debate` | 选择图模式 |
| `arena backtest --pair BTC/USDT --start DATE --end DATE` | 历史回测 |
| `arena sync` | 同步所有历史数据到 SQLite |
| `arena serve --port 8003` | 启动 FastAPI 服务 |
| `arena dashboard` | 启动 Streamlit 仪表盘 |
| `arena scheduler start` | 启动周期调度器 |
| `arena scheduler status` | 查看组合和仓位 |
| `arena journal log --limit 10` | 最近决策列表 |
| `arena journal show <hash>` | 决策详情 |
| `arena migrate` | 创建 PostgreSQL 表 |
| `arena risk reset-breaker` | 重置熔断器 |
| `arena live-check --exchange binance` | 实盘就绪检查 |

## 数据源

### 行情与链上

5 个数据源，支持优雅降级（无 API Key 也能运行）：

| 数据源 | 数据 | 成本 | 需要 Key |
|--------|------|------|---------|
| Binance | 期货 OI、资金费率、清算、多空比 | 免费 | 否 |
| DefiLlama | DeFi TVL、7 日变化、稳定币供应 | 免费 | 否 |
| CoinGlass | 持仓量、清算数据 | 免费层（1000 次/月）| 是 |
| CryptoQuant | 交易所净流入流出 | 免费层（每日）| 是 |
| Whale Alert | 大额转账 | 免费层（10 次/分钟）| 是 |

### 新闻与情绪

| 数据源 | 数据 | 成本 |
|--------|------|------|
| CoinDesk, CoinTelegraph, Decrypt | RSS 标题抓取 | 免费 |
| CoinGecko 社区 API | 社交热度（Twitter 粉丝、Reddit 订阅、情绪投票）| 免费 |
| FinBERT（本地模型）| 新闻标题情绪评分 | 免费（需安装 `ml` 可选依赖）|

### 宏观

| 数据源 | 数据 | 成本 |
|--------|------|------|
| FRED | 美联储利率、美元指数、VIX、S&P 500 | 免费（需 Key）|
| CoinGecko | BTC 主导率 | 免费 |
| Alternative.me | 恐惧贪婪指数 | 免费 |
| SoSoValue | BTC/ETH ETF 日流量、净资产 | 免费（需 Key）|

### 统一数据存储

所有数据缓存在 `~/.cryptotrader/market_data.db`（SQLite，WAL 模式）：
- 61+ 数据源，覆盖 7 个类别（宏观、链上、衍生品、DeFi、情绪、ETF、稳定币）
- 按源独立限速（5 分钟到 1 小时 TTL）
- 交易日数据前向填充（FRED、ETF），处理周末和假期
- `arena sync` 批量拉取全部历史数据用于回测

## 配置

### 配置文件

```
config/
├── default.toml          # 主配置（模式、模型、风控、调度器、数据源）
├── local.toml            # 本地覆盖（API Key，已 gitignore）
└── exchanges.toml.example  # 交易所凭证模板
```

先加载 `default.toml`，再深度合并 `local.toml`。首次加载后全局缓存。

### 关键配置段

```toml
[llm]
api_key = ""           # 统一 LLM API Key
base_url = ""          # API 端点（如 "http://localhost:3000/v1"）

[models]               # 按角色选择模型
analysis = "deepseek-chat"
verdict = "deepseek-reasoner"
fallback = "deepseek-chat"
# 还有：debate, tech_agent, chain_agent, news_agent, macro_agent

[debate]
max_rounds = 3
convergence_threshold = 0.1

[risk]
max_stop_loss_pct = 0.05
[risk.position]
max_single_pct = 0.10
max_total_exposure_pct = 0.50
[risk.loss]
max_daily_loss_pct = 0.03
max_drawdown_pct = 0.10

[scheduler]
enabled = false
pairs = ["BTC/USDT", "ETH/USDT"]
interval_minutes = 240
exchange_id = "binance"
daily_summary_hour = 0    # UTC 小时（0-23）

[reflection]
enabled = true
every_n_cycles = 20
```

### 环境变量

```bash
# LLM 服务商
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# 链上数据源（可选 — 无 Key 时优雅降级）
COINGLASS_API_KEY=your_key
CRYPTOQUANT_API_KEY=your_key
WHALE_ALERT_API_KEY=your_key
FRED_API_KEY=your_key

# 基础设施（可选 — 无时使用内存降级）
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://localhost:6379
```

## 风控门

11 项规则检查（不依赖 LLM），全部可在 `config/default.toml` 的 `[risk]` 段配置：

| 检查项 | 功能 | 默认值 |
|--------|------|--------|
| 最大单仓位 | 限制单个仓位占组合比例 | 10% |
| 总敞口 | 限制总敞口 | 50% |
| 日损失限制 | 日损失达到阈值触发熔断 | 3% |
| 最大回撤 | 深度回撤期间拒绝交易 | 10% |
| CVaR (99%) | 基于近期收益的条件风险价值 | 5% |
| 相关性 | 阻止高度相关的仓位（14 个硬编码组）| — |
| 冷却时间 | 同一交易对的最小交易间隔 | 60 分钟 |
| 亏损后冷却 | 亏损交易后的额外冷却 | 120 分钟 |
| 波动率 | 极端波动或闪崩时拒绝 | — |
| 资金费率 | 资金费率异常时阻止（拥挤信号）| — |
| 频率限制 | 每小时/每天交易次数上限 | — |
| 交易所健康 | 执行前检查 API 延迟 | — |

`close` 动作（平仓）**豁免全部风控检查** — 减仓是降低风险，不应被阻断。

## 通知

6 种事件类型的 Webhook 通知（在 `config/default.toml` 配置）：

| 事件 | 触发条件 |
|------|---------|
| `trade` | 订单成功成交 |
| `rejection` | 风控门拒绝交易 |
| `circuit_breaker` | 日损失限制触发熔断 |
| `daily_summary` | 调度器每日发送组合摘要 |
| `reconcile_mismatch` | 仓位对账发现不一致 |
| `portfolio_stale` | 组合数据过期或不可用 |

## API 端点

| 方法 | 路径 | 认证 | 描述 |
|------|------|------|------|
| POST | `/analyze` | API Key | 运行完整分析周期 |
| GET | `/journal/log?limit=10` | API Key | 最近决策记录 |
| GET | `/journal/{hash}` | API Key | 单条记录详情 |
| GET | `/portfolio` | API Key | 当前组合状态 |
| GET | `/risk/status` | API Key | 风控状态（交易次数、熔断）|
| GET | `/health` | 公开 | 系统状态（API、Redis、DB）|
| GET | `/metrics` | API Key | 统计：总决策数、胜率、平均分歧度 |

认证方式：`X-API-Key` 请求头（设置 `API_KEY` 环境变量后启用）。限流 60 次/分钟/IP。

## 执行层

### 模拟交易

- 默认模式，不涉及真实资金
- 可配置初始余额（默认 $10,000）
- 滑点模型：`base + amount × price × 1e-8`
- 通过 `asyncio.Lock` 保证线程安全

### 实盘交易

生产级加固的 `LiveExchange`，封装 ccxt：

- **重试机制**：指数退避（3 次），致命错误不重试（认证、权限、余额不足）
- **余额预检**：每次下单前验证可用余额
- **精度处理**：应用交易所特定的 `amount_to_precision()` / `price_to_precision()`
- **最小下单量**：检查交易所市场限制
- **超时控制**：每 2 秒轮询，30 秒后自动撤单
- **飞行前检查**：`arena live-check` 验证凭证、API 延迟、Redis 和数据库

```bash
# 验证实盘就绪状态
arena live-check --exchange binance
```

## Docker 部署

```bash
# 启动全套服务（PostgreSQL 16 + Redis 7 + 应用 + 仪表盘 + 调度器）
docker compose up -d

# 服务清单：
#   app        — FastAPI :8003
#   dashboard  — Streamlit :8501
#   scheduler  — 周期性交易循环
#   postgres   — 决策日志 + 组合持久化
#   redis      — 风控状态 + 冷却 + 熔断器
```

Dockerfile 使用多阶段构建 + 非 root 用户。健康检查每 30 秒轮询 `/health`。

## 项目结构

```
src/cryptotrader/
├── models.py          # 所有数据模型（DataSnapshot, AgentAnalysis, TradeVerdict, Order 等）
├── config.py          # TOML 配置加载 + 数据类验证
├── graph.py           # LangGraph 编排（3 种图模式）
├── state.py           # ArenaState TypedDict + build_initial_state() 工厂
├── scheduler.py       # APScheduler 周期性交易循环 + 每日摘要
├── notifications.py   # Webhook 通知（6 种事件）
├── db.py              # 共享 async DB session 工厂
├── data/
│   ├── store.py       # 统一 SQLite 存储（61+ 源，按源限速）
│   ├── snapshot.py    # SnapshotAggregator（数据聚合入口）
│   ├── market.py      # ccxt OHLCV + ticker + 资金费率 + 波动率
│   ├── onchain.py     # 聚合 5 个数据源（并行获取）
│   ├── news.py        # RSS + FinBERT 情绪 + CoinGecko 社交热度
│   ├── macro.py       # FRED + CoinGecko + 恐惧贪婪 + SoSoValue ETF
│   ├── sync.py        # 批量历史同步（arena sync）
│   └── providers/     # Binance, DefiLlama, CoinGlass, CryptoQuant, WhaleAlert, SoSoValue
├── agents/
│   ├── base.py        # BaseAgent + ToolAgent + create_llm() 工厂
│   ├── tech.py        # TechAgent（pandas-ta 指标）
│   ├── chain.py       # ChainAgent（ToolAgent + 链上工具）
│   ├── news.py        # NewsAgent（ToolAgent + 新闻工具）
│   ├── macro.py       # MacroAgent（宏观环境分析）
│   └── data_tools.py  # LangChain @tool 定义（6 链上 + 3 新闻）
├── debate/
│   ├── challenge.py   # 交叉挑战 prompt 构建
│   ├── convergence.py # 分歧度计算 + 收敛检测
│   ├── verdict.py     # AI 裁决（LLM）+ 规则裁决（回测）
│   └── researchers.py # 多空对抗辩论 + 评委
├── nodes/             # LangGraph 节点函数
│   ├── agents.py      # 4 智能体并行
│   ├── data.py        # 数据采集 + PnL 更新 + 趋势上下文
│   ├── debate.py      # 辩论轮次 + 收敛路由
│   ├── verdict.py     # 裁决 + 风控检查
│   ├── execution.py   # 下单 + 止损 + 仓位更新
│   └── journal.py     # 日志记录
├── risk/
│   ├── gate.py        # RiskGate（11 项顺序检查）
│   └── state.py       # RedisStateManager（含内存降级）
├── execution/
│   ├── simulator.py   # PaperExchange（模拟交易）
│   ├── exchange.py    # LiveExchange（ccxt，生产级加固）
│   ├── order.py       # OrderManager（状态机）
│   └── reconcile.py   # 仓位对账
├── portfolio/
│   └── manager.py     # PortfolioManager（DB + 内存）
├── journal/
│   ├── store.py       # JournalStore（PostgreSQL + 内存降级）
│   ├── search.py      # 相似搜索（资金费率、波动率、趋势）
│   └── calibrate.py   # 逐智能体准确率追踪 + 偏差检测
├── learning/
│   ├── verbal.py      # 语言强化（历史经验注入）
│   └── reflect.py     # 智能体自我反思（周期性策略备忘录）
└── backtest/
    ├── engine.py      # BacktestEngine（LLM + SMA 模式）
    ├── cache.py       # OHLCV SQLite 缓存
    ├── historical_data.py  # FnG、资金费率、FRED、期货成交量
    └── result.py      # BacktestResult 指标
src/cli/main.py        # Typer CLI（arena 命令）
src/api/               # FastAPI 服务（认证、限流、中间件）
src/dashboard/app.py   # Streamlit 仪表盘（概览、决策、风控、回测）
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.12+ |
| 包管理 | uv + Hatchling |
| LLM 编排 | LangChain 1.2+ / LangGraph 1.0+ |
| LLM 提供商 | ChatOpenAI（兼容 OpenAI、DeepSeek、Anthropic）|
| 交易所连接 | ccxt（Binance、OKX 等）|
| 数据处理 | pandas + pandas-ta + numpy |
| 调度 | APScheduler 3.x |
| 数据库 | PostgreSQL 16 + SQLAlchemy 2.0 async |
| 缓存/状态 | Redis 7 |
| 本地存储 | SQLite（数据存储 + LLM 缓存 + 反思备忘录）|
| API 服务 | FastAPI + Uvicorn |
| 仪表盘 | Streamlit |
| CLI | Typer + Rich |
| NLP（可选）| FinBERT via transformers + torch |

## 开发

```bash
make install          # uv pip install -e ".[dev]"
make test             # pytest tests/ -v（288 个测试）
make lint             # ruff check src/ tests/
make format           # ruff format src/ tests/
make scheduler        # arena scheduler start
make pre-commit-run   # 运行所有 pre-commit 钩子

# 运行单个测试
uv run pytest tests/test_risk_gate.py -v
uv run pytest tests/test_risk_gate.py::test_max_position -v

# Docker 基础设施
docker compose up -d   # PostgreSQL 16 + Redis 7
arena migrate          # 创建数据库表
arena sync             # 同步历史数据
```

### 代码质量

- **零 lint 错误**：`ruff check src/ tests/` 必须零错误通过
- **禁止 `noqa` 注释**：遇到 C901 必须重构（阈值 = 10）
- **288 个测试**，1 个跳过，70% 覆盖率
- **异步测试**：`asyncio_mode = "auto"` — 无需 `@pytest.mark.asyncio`
- **必须用 `uv run pytest`**（Python 3.12 venv），不要用裸 `pytest`

## 许可证

MIT
