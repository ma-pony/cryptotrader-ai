# CryptoTrader AI

基于 LangGraph 多智能体辩论的 AI 加密货币交易系统。

## 概述

4 个专业 AI 智能体（技术面、链上数据、新闻情绪、宏观经济）独立分析市场数据，然后通过交叉挑战辩论轮次达成共识。硬编码的风控门（11 项规则检查，不依赖 LLM）强制执行仓位限制、损失限制和熔断机制。每个决策都记录在类 Git 的决策日志中，便于审计和基于经验的学习。

每个智能体在输出信号前都会运行**领域专属的预信号检查清单**（受 Devin 的先思后行模式启发），以减少过度自信和幻觉。

## 系统架构

```
数据采集 → 语言强化注入 → 4 智能体（并行）
→ 交叉挑战辩论（2-3 轮）→ 收敛检查
→ 裁决 → 风控门（11 项检查）→ 执行 / 拒绝 → 日志记录                                        ↓
                              持仓回写 → 快照保存
```

**三种图模式：**
- `build_trading_graph()` — 完整流程，含辩论循环和收敛检查
- `build_lite_graph()` — 跳过辩论，用于回测
- `build_debate_graph()` — 多空对抗辩论 + 评委（TradingAgents 风格）

## 快速开始

```bash
# 安装
uv pip install -e ".[dev]"

# 运行单次分析（模拟交易）
arena run --pair BTC/USDT --mode paper

# 多交易对分析
arena run --pair BTC/USDT --pair ETH/USDT --mode paper

# 回测
arena backtest --pair BTC/USDT --start 2024-01-01 --end 2024-06-01 --interval 4h

# 查看决策日志
arena journal log --limit 10
arena journal show <hash>

# 调度器
arena scheduler start       # 周期性交易循环
arena scheduler status      # 持仓和仓位状态

# 仪表盘 & API
arena dashboard             # Streamlit UI
arena serve --port 8003     # FastAPI 服务
```

## 数据源

### 行情与链上

5 个数据源，支持优雅降级（无 API Key 也能运行）：

| 数据源 | 数据 | 成本 | 需要 Key |
|--------|------|------|---------|
| Binance | 期货持仓量、资金费率、清算数据 | 免费 | 否 |
| DefiLlama | DeFi TVL、7日变化 | 免费 | 否 |
| CoinGlass | 持仓量、清算数据 | 免费层（1000次/月）| 是 |
| CryptoQuant | 交易所净流入流出 | 免费层（每日）| 是 |
| Whale Alert | 大额转账 | 免费层（10次/分钟）| 是 |

### 新闻与情绪

| 数据源 | 数据 | 成本 |
|--------|------|------|
| CoinDesk, CoinTelegraph, Decrypt | RSS 标题抓取 | 免费 |
| CoinGecko 社区 API | 社交热度（Twitter 粉丝、Reddit 订阅、情绪投票）| 免费 |

### 宏观

| 数据源 | 数据 | 成本 |
|--------|------|------|
| FRED | 美联储利率、美元指数 | 免费（需 Key）|
| CoinGecko | BTC 主导率 | 免费 |
| Alternative.me | 恐惧贪婪指数 | 免费 |

## 配置

### 配置文件

- `config/default.toml` — 模式、模型、辩论参数、数据源 API Key、调度器、通知
- `config/risk.toml` — 11 项风控参数（仓位限制、损失限制、冷却时间）
- `.env` — LLM API Key、DATABASE_URL、REDIS_URL

### API Keys（环境变量）

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

11 项规则检查（不依赖 LLM），全部可在 `config/risk.toml` 配置：

| 检查项 | 功能 |
|--------|------|
| 最大单仓位 | 限制单个仓位占组合比例 |
| 总敞口 | 限制总敞口 |
| 日损失限制 | 日损失达到阈值触发熔断 |
| 最大回撤 | 深度回撤期间拒绝交易 |
| CVaR (99%) | 基于近期收益的条件风险价值 |
| 相关性 | 阻止高度相关的仓位 |
| 冷却时间 | 同一交易对的最小交易间隔 |
| 亏损后冷却 | 亏损交易后的额外冷却 |
| 波动率 | 极端波动或闪崩时拒绝 |
| 资金费率 | 资金费率异常时阻止（拥挤信号）|
| 频率限制 | 每小时/每天交易次数上限 |
| 交易所健康 | 执行前检查 API 延迟 |

## 通知

5 种事件类型的 Webhook 通知（在 `config/default.toml` 配置）：

| 事件 | 触发条件 |
|------|---------|
| `trade` | 订单成功成交 |
| `rejection` | 风控门拒绝交易 |
| `circuit_breaker` | 日损失限制触发熔断 |
| `daily_summary` | 调度器每日发送组合摘要 |
| `reconcile_mismatch` | 仓位对账发现不一致 |

## API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/analyze` | 运行完整分析周期 |
| GET | `/journal/log?limit=10` | 最近决策记录 |
| GET | `/journal/{hash}` | 单条记录详情 |
| GET | `/portfolio` | 当前组合状态 |
| GET | `/risk/status` | 风控状态（交易次数、熔断）|
| GET | `/health` | 系统状态（API、Redis、DB）|
| GET | `/metrics` | 统计：总决策数、胜率、平均分歧度 |

## 项目结构

```
src/cryptotrader/
├── models.py          # 所有数据模型
├── config.py          # TOML 配置加载 + Pydantic 验证
├── graph.py           # LangGraph 编排（3 种图模式）
├── scheduler.py       # 周期性交易循环 + 每日摘要
├── notifications.py   # Webhook 通知（5 种事件）
├── data/
│   ├── market.py      # ccxt 行情数据
│   ├── onchain.py     # 聚合 5 个数据源
│   ├── news.py        # RSS + 关键词情绪 + CoinGecko 社交热度
│   ├── macro.py       # FRED + CoinGecko + 恐惧贪婪（并行获取）
│   └── providers/     # Binance, DefiLlama, CoinGlass, CryptoQuant, WhaleAlert
├── agents/            # 4 个智能体（含预信号检查清单）
├── debate/            # 交叉挑战、多空对抗、收敛检查、裁决
├── risk/              # 风控门 + 11 项规则检查 + Redis 状态
├── execution/         # 订单管理器、交易所适配器（实盘+模拟）、对账器
├── portfolio/         # 持仓追踪 + 权益快照（DB + 内存）
├── journal/           # 决策提交链 + 相似搜索 + 校准
└── learning/          # 基于历史决策的语言强化
src/cli/               # Typer CLI（arena 命令）
src/api/               # FastAPI 服务
src/dashboard/         # Streamlit 仪表盘
```

## 开发

```bash
make install          # uv pip install -e ".[dev]"
make test             # pytest tests/ -v（165 个测试）
make lint             # ruff check src/ tests/
```

## 许可证

MIT
