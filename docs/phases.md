# CryptoTrader AI — 分阶段设计需求文档

> 基于 ARCHITECTURE.md v5 + Phase 1/2 实际代码审计
> 日期：2026-02-20

---

## Phase 1: 最小闭环 ✅ (已完成, commit e2e555a)

### 目标
跑通一次完整的「数据 → 分析 → 辩论 → 风控 → 模拟盘执行」流程。

### 交付物清单

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 数据模型 | models.py | ✅ | MarketData/OnchainData/NewsSentiment/MacroData/DataSnapshot/AgentAnalysis/TradeVerdict/GateResult/Order/OrderStatus/DecisionCommit |
| 配置 | config.py, config/*.toml | ✅ | TOML 加载 + Pydantic 验证 |
| 行情采集 | data/market.py | ✅ | ccxt OHLCV + Ticker + 资金费率 + 订单簿不平衡度 + 波动率 |
| 链上数据 | data/onchain.py | ✅ | 占位，Phase 1 只用 ccxt 资金费率 |
| 新闻/宏观 | data/news.py, data/macro.py | ✅ | 占位，返回默认值 |
| 数据聚合 | data/snapshot.py | ✅ | DataSnapshot 统一输出 |
| TechAgent | agents/tech.py | ✅ | pandas-ta 指标（RSI/MACD/SMA/BBands/ATR）+ LLM 分析 |
| ChainAgent | agents/chain.py | ✅ | 资金费率 + OI + Exchange Flow 数据注入 LLM |
| NewsAgent | agents/news.py | ✅ | 占位 |
| MacroAgent | agents/macro.py | ✅ | 占位 |
| Agent 基类 | agents/base.py | ✅ | 标准 analyze() 接口 + prompt 模板 + litellm 调用 |
| 交叉质询 | debate/challenge.py | ✅ | Round 1 独立分析 / Round 2+ 看到他人结论后质疑 |
| 收敛判定 | debate/convergence.py | ✅ | 分歧度计算 + 稳定性检测（变化 <10% 或达到上限） |
| 共识生成 | debate/verdict.py | ✅ | Confidence-weighted 投票 + 分歧度仓位调节 |
| 风控门控 | risk/gate.py | ✅ | 11 项检查顺序执行，任一失败即拒绝 |
| 风控检查 | risk/checks/*.py | ✅ | 仓位/亏损/CVaR/相关性/冷却/波动率/资金费率/频率/交易所健康 |
| Redis 状态 | risk/state.py | ✅ | Redis 管理 + 不可用时保守拒绝 |
| 订单管理 | execution/order.py | ✅ | 状态机 + 合法转换校验 |
| 模拟盘 | execution/simulator.py | ✅ | PaperExchange + 滑点模型 + 余额追踪 |
| 实盘适配 | execution/exchange.py | ✅ | LiveExchange (ccxt) + ExchangeAdapter Protocol |
| 对账 | execution/reconcile.py | ✅ | 本地 vs 交易所状态比对 |
| Decision Journal | journal/commit.py | ✅ | commit 构建 + SHA256 hash |
| Journal 存储 | journal/store.py | ✅ | 内存存储（无 DB 时降级） |
| LangGraph 编排 | graph.py | ✅ | 完整 DAG：collect → experience → fan-out(4 agent) → debate loop → verdict → risk → execute/reject |
| CLI | cli/main.py | ✅ | `arena run/journal/serve` |
| API | api/ | ✅ | FastAPI 骨架 |
| 测试 | tests/ | ✅ | 11 tests passing |

### 设计决策
- LangGraph fan-out 实现并行分析（4 Agent 同时跑）
- 辩论用 conditional_edges 实现循环（check_convergence → converged/continue）
- 风控纯规则，零 LLM 调用
- JournalStore 内存降级，无需 PostgreSQL 也能跑

---

## Phase 2: 完整智能层 ✅ (已完成, commit ac3aaac)

### 目标
所有数据源接入真实 API，Agent 使用真实数据分析，API 服务可用。

### 交付物清单

| 模块 | 文件 | 状态 | 说明 |
|------|------|------|------|
| DefiLlama | data/providers/defillama.py | ✅ | TVL + 7 日变化率，免费无 Key |
| CoinGlass | data/providers/coinglass.py | ✅ | OI + 清算，需 API Key |
| CryptoQuant | data/providers/cryptoquant.py | ✅ | Exchange netflow，需 API Key |
| WhaleAlert | data/providers/whale_alert.py | ✅ | 巨鲸转账，需 API Key |
| 链上聚合 | data/onchain.py | ✅ | 4 provider 聚合 + 优雅降级 |
| 新闻采集 | data/news.py | ✅ | CoinDesk/CoinTelegraph RSS + 关键词情绪 |
| 宏观数据 | data/macro.py | ✅ | FRED(利率/DXY) + CoinGecko(BTC dominance) + Fear&Greed |
| NewsAgent | agents/news.py | ✅ | 使用真实新闻数据 |
| MacroAgent | agents/macro.py | ✅ | 使用真实宏观数据 |
| Verbal Reinforcement | learning/verbal.py | ✅ | 检索相似历史 → 格式化经验 → 注入 prompt |
| Journal 搜索 | journal/search.py | ✅ | 按 funding_rate/volatility 范围检索 |
| 权重校准 | journal/calibrate.py | ✅ | 按历史准确率计算 Agent 权重 |
| API /analyze | api/routes/analyze.py | ✅ | 完整 graph 调用 |
| API /health | api/routes/health.py | ✅ | 系统状态检查 |
| API /journal | api/routes/journal.py | ✅ | log + show |
| 多币种 CLI | cli/main.py | ✅ | `--pair BTC/USDT ETH/USDT` |
| Provider 配置 | config/default.toml | ✅ | API Key + enable/disable 开关 |
| 测试 | tests/ | ✅ | 51 tests passing（+40） |

### 设计决策
- 所有 provider 优雅降级：无 API Key → 返回默认值 + 日志警告，不崩溃
- 新闻情绪用关键词匹配（不依赖外部 NLP 服务），Phase 4 可升级到 FinBERT
- JournalStore 仍用内存，Phase 3 升级到 PostgreSQL
- 权重校准需要历史 PnL 数据，初期返回等权

### 当前已知限制（Phase 2 遗留）
1. JournalStore 纯内存，重启丢失
2. 无回测引擎，无法验证策略历史表现
3. 无持久化 portfolio 状态，每次运行独立
4. 无定时调度，需手动触发
5. 无 Dashboard / 可视化
6. LiveExchange 未经实盘验证

---

## Phase 3: 实盘就绪（预计 2-3 周）

### 目标
系统可以在真实交易所上运行，有持久化存储、回测验证、定时调度和基础监控。

### 3.1 持久化存储

**PostgreSQL Journal Store**

当前 JournalStore 纯内存，重启即丢。需要：

- SQLAlchemy 2.0 async ORM 模型：`decision_commits` 表
  - 字段映射 DecisionCommit dataclass 所有字段
  - `hash` 为主键，`parent_hash` 为外键（自引用链表）
  - `analyses` / `snapshot_summary` / `challenges` 存为 JSONB
  - `created_at` 索引，`pair` 索引
- Alembic 迁移管理
- 连接池配置（asyncpg, pool_size=5, max_overflow=10）
- 启动时自动建表（`create_all`），生产环境用 Alembic
- 保留内存降级：`DATABASE_URL` 未配置时用内存 store

**Portfolio 持久化**

当前每次运行 portfolio 是硬编码的 `{"total_value": 10000}`。需要：

- `portfolios` 表：account_id, pair, amount, avg_price, updated_at
- `portfolio_snapshots` 表：定期快照，用于回撤计算
- PortfolioManager 类：
  - `get_portfolio(account_id)` → 当前持仓
  - `update_position(account_id, pair, amount, price)` → 更新持仓
  - `get_daily_pnl(account_id)` → 当日盈亏
  - `get_drawdown(account_id)` → 当前回撤
  - `get_returns(account_id, days=60)` → 历史收益率序列（CVaR 用）

**Redis 持久化**

当前 RedisStateManager 只有接口。需要：
- 实际连接 Redis（aioredis）
- 冷却期状态：`cooldown:{pair}` → TTL 自动过期
- 日交易计数：`trades:daily:{date}` → INCR + TTL
- 小时交易计数：`trades:hourly:{hour}` → INCR + TTL
- 熔断状态：`circuit_breaker:active` → 需人工确认解除

### 3.2 回测引擎

**核心需求：验证策略在历史数据上的表现。**

```
arena backtest --pair BTC/USDT --start 2025-01-01 --end 2025-12-31 --interval 4h
```

- BacktestEngine 类：
  - 输入：pair, start_date, end_date, interval, initial_capital
  - 按时间步进，每步构造 DataSnapshot（历史数据）
  - 调用完整 graph（分析 → 辩论 → 风控 → 模拟执行）
  - 记录每步的 DecisionCommit
  - 输出：BacktestResult

- BacktestResult：
  - total_return, sharpe_ratio, max_drawdown, win_rate
  - 每笔交易明细（entry/exit/pnl）
  - 按月/周汇总
  - equity curve 数据（用于绘图）

- 历史数据获取：
  - ccxt `fetch_ohlcv` 分页拉取历史 K 线
  - 本地缓存到 SQLite（避免重复请求）
  - 链上/新闻/宏观数据：回测时用占位值（这些数据难以获取历史）

- 防前视偏差：
  - 严格按时间顺序，每步只能看到当前时间之前的数据
  - 滑点模型使用历史成交量估算
  - 手续费按交易所实际费率（默认 0.1%）

### 3.3 定时调度

**Scheduler：按配置的频率自动运行分析周期。**

```toml
# config/default.toml 新增
[scheduler]
enabled = false
pairs = ["BTC/USDT", "ETH/USDT"]
interval_minutes = 240  # 每 4 小时
timezone = "UTC"
```

- APScheduler 集成（或简单的 asyncio 定时器）
- 每个 interval 对所有配置的 pair 运行完整 graph
- 运行结果写入 Journal
- 异常不中断调度，记录错误日志
- CLI：`arena scheduler start` / `arena scheduler status`

### 3.4 实盘交易加固

**LiveExchange 生产级加固：**

- 订单超时处理：下单后 30s 未成交 → 自动撤单
- 部分成交处理：PARTIALLY_FILLED → 等待或撤销剩余
- 网络重试：ccxt 调用加 tenacity 重试（3 次，指数退避）
- API 限流：per-exchange rate limiter（ccxt 内置 + 额外保护）
- 余额预检：下单前检查可用余额是否足够
- 最小下单量：检查交易所 `markets[pair].limits.amount.min`
- 精度处理：`amount_to_precision()` / `price_to_precision()`

**对账增强：**

- 定时对账（每 5 分钟）
- 发现不一致 → 日志告警 + 可选 webhook 通知
- 孤儿订单检测：交易所有但本地没有的订单

### 3.5 通知系统

- Webhook 通知（POST JSON）：
  - 交易执行通知
  - 风控拒绝通知
  - 熔断触发通知
  - 对账不一致告警
  - 每日 PnL 摘要
- 配置：

```toml
[notifications]
webhook_url = ""
enabled = true
events = ["trade", "rejection", "circuit_breaker", "reconcile_mismatch", "daily_summary"]
```

### 3.6 基础 Dashboard

**最小可用的 Web 监控界面。**

- 技术选型：FastAPI + Jinja2 模板（或 Streamlit，更快）
- 页面：
  1. **Overview**：当前持仓、今日 PnL、总 PnL、活跃 pair
  2. **Decisions**：最近决策列表（方向/置信度/分歧度/风控结果）
  3. **Decision Detail**：单个决策的完整推理链（各 Agent 分析 + 辩论过程）
  4. **Risk Status**：风控状态（熔断是否激活、今日交易次数、冷却中的 pair）
  5. **Backtest**：回测结果展示（equity curve 图表）

### 3.7 测试要求

- 集成测试：完整 graph 端到端（mock LLM 响应）
- 回测引擎测试：已知数据 → 已知结果
- LiveExchange 测试：ccxt sandbox 模式
- 对账测试：模拟状态不一致场景
- API 测试：所有端点 + 错误场景
- 目标：≥80% 覆盖率（核心模块 ≥90%）

### 3.8 文档

- docs/getting-started.md：从零到运行的完整指南
- docs/configuration.md：所有配置项说明
- docs/risk-management.md：风控机制详解
- docs/backtest.md：回测使用指南
- docs/api.md：API 接口文档（OpenAPI 自动生成 + 补充说明）
- docs/deployment.md：Docker 部署指南

### Phase 3 交付标准
- [ ] `arena run --pair BTC/USDT --mode paper` 结果持久化到 PostgreSQL
- [ ] `arena backtest --pair BTC/USDT --start 2025-06-01 --end 2025-12-31` 输出完整回测报告
- [ ] `arena scheduler start` 自动定时运行
- [ ] `arena serve` 启动 API + Dashboard
- [ ] LiveExchange 在 Binance testnet 上验证通过
- [ ] Webhook 通知正常发送
- [ ] 测试覆盖率 ≥80%
- [ ] 文档完整

---

## Phase 4: 进化与优化（长期）

### 目标
从"能用"到"好用"——性能优化、高级功能、社区化。

### 4.1 LLM + RL 混合决策

**学术依据：** FinCon (NeurIPS 2024), Meta-RL-Crypto (arXiv 2025), LLM-guided RL Trading (2025)

**核心思路：** LLM 提供语义理解和策略框架，RL 做参数自适应优化。

- RL 层位于 Verdict 之后、Risk Gate 之前
- 输入：Verdict 的 direction/confidence/divergence + 市场特征向量
- 输出：调整后的 position_scale（RL 学习最优仓位比例）
- 训练：
  - 环境：回测引擎作为 gym 环境
  - Reward：risk-adjusted return（Sharpe ratio）
  - 算法：PPO（稳定，适合金融场景）
  - 框架：Stable-Baselines3 或 FinRL
- Regime Detection：
  - 市场状态分类（trending/ranging/volatile/crash）
  - 不同 regime 下使用不同 RL 策略权重
  - Meta-learning 快速适应新 regime（参考 Meta-RL-Crypto）

### 4.2 高级情绪分析

当前：关键词匹配（Phase 2）。升级路径：

- **Phase 4a**：FinBERT 本地推理（HuggingFace transformers）
  - 模型：ProsusAI/finbert（110M 参数，CPU 可跑）
  - 输入：新闻标题 + 摘要
  - 输出：positive/negative/neutral + 置信度
- **Phase 4b**：社交媒体实时情绪
  - Twitter/X API（crypto KOL 监控）
  - Reddit（r/cryptocurrency, r/bitcoin 热帖）
  - 情绪突变检测（短时间内情绪急剧变化 → 信号）
- **Phase 4c**：多模态
  - YouTube crypto 频道标题/缩略图分析
  - TradingView 图表截图 → 视觉模式识别

### 4.3 高级链上分析

当前：Exchange Flow + OI + 清算 + 巨鲸（Phase 2）。升级路径：

- **MEV 监控**：大额 DEX 交易前的 MEV 活动 → 预判大单方向
- **稳定币流动**：USDT/USDC mint/burn → 市场资金流入/流出
- **DeFi 协议健康**：借贷协议清算风险 → 系统性风险预警
- **NFT/GameFi 资金流**：资金从 NFT/GameFi 流向主流币 → 风险偏好变化
- **跨链桥流量**：资金跨链方向 → 生态热度变化

### 4.4 多交易所套利

- 同币种跨交易所价差监控
- 三角套利（BTC/USDT → ETH/BTC → ETH/USDT）
- 资金费率套利（现货 + 永续合约对冲）
- 需要：
  - 多交易所并行连接
  - 亚秒级延迟要求
  - 独立的套利风控（不走 AI 辩论，纯规则）

### 4.5 DEX 支持

- Uniswap/PancakeSwap 集成（web3.py）
- DEX 聚合器（1inch API）
- Gas 费优化
- MEV 保护（Flashbots）
- 滑点预估（基于池子深度）

### 4.6 Agent 插件系统

**让社区贡献自定义 Agent。**

```python
# 用户只需实现 BaseAgent 接口
class MyCustomAgent(BaseAgent):
    def _build_prompt(self, snapshot, experience):
        # 自定义分析逻辑
        ...

# 注册到 config
[agents.custom]
module = "my_agents.sentiment_v2"
class = "SentimentV2Agent"
enabled = true
model = "gpt-4o-mini"
```

- Agent 发现机制：扫描 `plugins/` 目录或 pip 安装的包
- Agent 沙箱：自定义 Agent 不能直接访问交易所 API
- Agent 市场：社区分享 Agent 配置和 prompt

### 4.7 性能优化

- **数据缓存层**：Redis 缓存行情数据（TTL 按 timeframe）
- **LLM 缓存**：相同 prompt hash → 缓存响应（短 TTL）
- **并发控制**：semaphore 限制同时运行的 LLM 调用数
- **流式输出**：Agent 分析结果流式返回到 Dashboard
- **数据库优化**：Journal 查询加索引、分区表（按月）

### 4.8 高级 Dashboard

- 实时 WebSocket 推送（决策进度、持仓变化）
- 交互式 equity curve（Plotly/ECharts）
- Agent 辩论过程可视化（谁说了什么、谁被说服了）
- 风控状态实时面板
- 回测对比（多策略并排）
- 移动端适配

### 4.9 开源社区建设

- GitHub Actions CI/CD
- 贡献指南（CONTRIBUTING.md）
- Issue/PR 模板
- Discord 社区
- 文档站（MkDocs Material）
- PyPI 发布（`pip install cryptotrader-ai`）
- Docker Hub 镜像

### Phase 4 优先级排序

| 优先级 | 功能 | 理由 |
|--------|------|------|
| P0 | Agent 插件系统 | 社区增长的基础 |
| P0 | 高级情绪分析 (FinBERT) | 低成本高收益，CPU 可跑 |
| P1 | LLM + RL 混合 | 学术前沿，差异化 |
| P1 | 高级 Dashboard | 用户体验 |
| P2 | 多交易所套利 | 独立策略，可并行开发 |
| P2 | DEX 支持 | 市场需求大但开发复杂 |
| P3 | 高级链上分析 | 数据源获取难度高 |
| P3 | 性能优化 | 用户量上来再做 |

---

## 总览：四阶段里程碑

```
Phase 1 ✅  最小闭环        52 files, 11 tests   → 能跑通一次完整流程
Phase 2 ✅  完整智能层      57 files, 51 tests   → 真实数据源 + API 服务
Phase 3     实盘就绪        ~80 files, ~120 tests → 持久化 + 回测 + 实盘 + Dashboard
Phase 4     进化与优化      ~100+ files           → RL + 插件 + 套利 + 社区
```

### 技术债务追踪

| 债务 | 引入阶段 | 计划解决 | 说明 |
|------|---------|---------|------|
| JournalStore 纯内存 | Phase 1 | Phase 3 | 重启丢失所有决策记录 |
| Portfolio 硬编码 | Phase 1 | Phase 3 | 每次运行独立，无持仓追踪 |
| Redis 未实际连接 | Phase 1 | Phase 3 | 冷却/频率限制不生效 |
| 新闻情绪关键词匹配 | Phase 2 | Phase 4 | 准确率有限，升级到 FinBERT |
| 无回测验证 | Phase 2 | Phase 3 | 无法量化策略表现 |
| LiveExchange 未验证 | Phase 1 | Phase 3 | 需 testnet 实测 |
| 无定时调度 | Phase 2 | Phase 3 | 需手动触发 |
| 无通知机制 | Phase 2 | Phase 3 | 交易/告警无法推送 |
| Agent 等权 | Phase 2 | Phase 3+ | 需积累历史数据后校准 |

---

*文档随项目演进持续更新。每个 Phase 完成后回顾并标记实际交付状态。*
