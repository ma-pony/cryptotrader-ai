# CryptoTrader AI — 系统架构设计文档

> 版本：v5 Professional Edition
> 日期：2026-02-20
> 基于：4 轮深度调研（12+ 开源项目源码分析、25 篇学术论文、6 个 API 文档）

---

## 1. 项目定位

### 1.1 市场空白

```
规则驱动交易系统                    AI 研究框架
Freqtrade (28k⭐)                 TradingAgents (30.2k⭐)
Hummingbot (8k⭐)                 ai-hedge-fund (45.8k⭐)
Jesse (6k⭐)                      NOFX (10.5k⭐)
│                                  │
│ ✅ 实盘执行                       │ ✅ AI 决策
│ ✅ 风控                          │ ✅ 多 Agent
│ ❌ 无 AI 决策                     │ ❌ 不执行交易
│ ❌ 无链上数据                     │ ❌ 无硬风控
│                                  │ ❌ 无链上数据（多数只做美股）
└──────────────┐    ┌──────────────┘
               ▼    ▼
          CryptoTrader AI
    ✅ AI 辩论决策（LangGraph）
    ✅ 实盘执行（ccxt）
    ✅ 硬风控（11 项规则检查）
    ✅ 链上数据（三件套零成本）
    ✅ 决策追溯（Git-like Journal）
    ✅ 可集成现有量化系统
```

### 1.2 目标用户

| 用户类型 | 需求 | 我们提供 |
|---------|------|---------|
| 加密交易者 | 自动化交易 + 不亏光 | 完整闭环 + 硬风控 |
| 量化开发者 | AI 层集成到现有系统 | HTTP API + 标准信号格式 |
| AI 研究者 | 多 Agent 交易实验 | LangGraph 编排 + Decision Journal |

### 1.3 竞品对比

| 维度 | ai-hedge-fund | TradingAgents | NOFX | OpenAlice | Freqtrade | **CryptoTrader AI** |
|------|--------------|---------------|------|-----------|-----------|---------------------|
| 语言 | Python | Python | Go+TS | TypeScript | Python | **Python** |
| 市场 | 美股 | 美股 | 加密 | 加密+美股 | 加密 | **加密** |
| AI 决策 | 并行投票 | 辩论(1轮) | 辩论 | 单Agent | 规则 | **辩论(2轮)+分歧度信号** |
| 链上数据 | ❌ | ❌ | 部分 | ❌ | ❌ | **✅ 三件套** |
| 风控 | LLM软判断 | LLM软判断 | 中等 | 无 | 规则止损 | **硬风控11项+一票否决** |
| 实盘 | ❌ | ❌ | ✅ | demo | ✅ | **✅** |
| 决策追溯 | ❌ | ❌ | 部分 | Git-like | ❌ | **Git-like+Verbal Reinforcement** |
| 经验学习 | ❌ | ❌ | ❌ | ❌ | ❌ | **✅ 从历史决策中学习** |
| 集成能力 | 独立 | 独立 | 独立 | 独立 | 独立 | **独立+HTTP API集成** |

---

## 2. 架构总览

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Interface Layer                          │
│              CLI  │  FastAPI (REST)  │  Dashboard                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    LangGraph Orchestrator                        │
│                   (StateGraph + 条件循环)                         │
│                                                                 │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │             │  │                  │  │                    │  │
│  │  Data Layer │  │  Intelligence    │  │  Execution Layer   │  │
│  │             │  │  Layer           │  │                    │  │
│  │ ┌─────────┐ │  │ ┌──────────────┐ │  │ ┌────────────────┐ │  │
│  │ │ Market  │ │  │ │  Analysis    │ │  │ │  Risk Gate     │ │  │
│  │ │ (ccxt)  │ │  │ │  Phase      │ │  │ │  (11项硬检查)   │ │  │
│  │ ├─────────┤ │  │ │             │ │  │ │  一票否决权     │ │  │
│  │ │ OnChain │ │  │ │ Tech Agent  │ │  │ ├────────────────┤ │  │
│  │ │ (3件套)  │ │  │ │ Chain Agent │ │  │ │  Order Manager │ │  │
│  │ ├─────────┤ │  │ │ News Agent  │ │  │ │  (状态机+滑点)  │ │  │
│  │ │ News    │ │  │ │ Macro Agent │ │  │ ├────────────────┤ │  │
│  │ │ (爬虫)   │ │  │ ├──────────────┤ │  │ │  Exchange      │ │  │
│  │ ├─────────┤ │  │ │  Debate     │ │  │ │  Adapter       │ │  │
│  │ │ Macro   │ │  │ │  Phase      │ │  │ │  (ccxt)        │ │  │
│  │ │ (FRED)  │ │  │ │             │ │  │ │  paper / live   │ │  │
│  │ └─────────┘ │  │ │ Cross-      │ │  │ └────────────────┘ │  │
│  │             │  │ │ Challenge   │ │  │                    │  │
│  │  DataSnapshot│  │ │ Convergence │ │  │                    │  │
│  │  (统一输出)  │  │ │ Verdict     │ │  │                    │  │
│  │             │  │ └──────────────┘ │  │                    │  │
│  └─────────────┘  └──────────────────┘  └────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Decision Journal (Git-like Commit Chain)        │ │
│  │              + Verbal Reinforcement (经验反哺)                │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │              PostgreSQL  │  Redis  │  File Store             │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 核心决策流程

```
                    ┌──────────────┐
                    │  Scheduler   │
                    │  (定时触发)   │
                    └──────┬───────┘
                           ▼
                ┌──────────────────┐
                │  Data Collection │  ← ccxt + DefiLlama + CoinGlass + CryptoQuant
                │  (DataSnapshot)  │
                └──────┬───────────┘
                       ▼
        ┌──────────────────────────────┐
        │     Verbal Reinforcement     │  ← 从 Decision Journal 检索
        │  (注入相似市场条件下的经验)     │    相似历史经验到 Agent prompt
        └──────────────┬───────────────┘
                       ▼
    ┌─────────┬────────┴────────┬─────────┐
    ▼         ▼                 ▼         ▼
┌───────┐ ┌───────┐       ┌───────┐ ┌───────┐
│ Tech  │ │ Chain │       │ News  │ │ Macro │   ← Phase A: 并行分析
│ Agent │ │ Agent │       │ Agent │ │ Agent │     (LangGraph fan-out)
└───┬───┘ └───┬───┘       └───┬───┘ └───┬───┘
    └─────────┴───────┬───────┴─────────┘
                      ▼
            ┌──────────────────┐
            │ Cross-Challenge  │  ← Phase B: 交叉质询
            │ (每个 Agent 质疑  │    Agent 能看到其他人的分析
            │  其他 Agent 结论) │    基于数据分歧辩论
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │   Convergence    │  ← 稳定性检测
            │   Check          │    direction+confidence 变化 < 阈值?
            └───┬──────────┬───┘
          stable│          │continue (≤3轮)
                ▼          └──→ 回到 Cross-Challenge
        ┌──────────────┐
        │   Verdict    │  ← 加权共识 + 分歧度计算
        │              │    分歧度本身作为仓位调节信号
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │  Risk Gate   │  ← 11 项硬检查，纯规则
        │  (一票否决)   │    任一失败 → 拒绝交易
        └───┬──────┬───┘
      pass  │      │ reject
            ▼      ▼
     ┌──────────┐ ┌──────────────┐
     │ Execute  │ │ Journal Only │
     │ Order    │ │ (记录拒绝原因)│
     └────┬─────┘ └──────┬───────┘
          └───────┬──────┘
                  ▼
        ┌──────────────────┐
        │ Decision Journal │  ← Git-like commit
        │ (记录完整推理链)  │    hash → parent_hash 链表
        └──────────────────┘
```

---

## 3. LangGraph 编排设计

### 3.1 State Schema

参考 ai-hedge-fund 的共享 state dict 模式（源码验证），扩展辩论和经验学习字段。

```python
from typing import Annotated, Sequence, Literal, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
import operator

def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}

class ArenaState(TypedDict):
    # 消息链（追加合并，保留所有 Agent 输出）
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 共享数据（Agent 通过写入不同 key 传递数据）
    data: Annotated[dict[str, Any], merge_dicts]
    # 配置（模型、参数等）
    metadata: Annotated[dict[str, Any], merge_dicts]
    # 辩论控制
    debate_round: int           # 当前轮次
    max_debate_rounds: int      # 上限（默认 3）
    # 分歧度追踪
    divergence_scores: list[float]  # 每轮的分歧度
```

### 3.2 Graph 构建

```python
from langgraph.graph import StateGraph, END, START

def build_trading_graph(config: dict) -> StateGraph:
    graph = StateGraph(ArenaState)

    # ── 数据采集 ──
    graph.add_node("collect_data", collect_snapshot)
    graph.add_node("inject_experience", verbal_reinforcement)

    # ── 并行分析（fan-out）──
    graph.add_node("tech_agent", tech_analyze)
    graph.add_node("chain_agent", chain_analyze)
    graph.add_node("news_agent", news_analyze)
    graph.add_node("macro_agent", macro_analyze)

    # ── 辩论 ──
    graph.add_node("cross_challenge", debate_round)
    graph.add_node("check_convergence", check_stability)

    # ── 决策 + 执行 ──
    graph.add_node("verdict", make_verdict)
    graph.add_node("risk_gate", risk_check)
    graph.add_node("execute", place_order)
    graph.add_node("record_rejection", journal_rejection)

    # ── 边连接 ──
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "inject_experience")

    # fan-out: 经验注入后并行分析
    graph.add_edge("inject_experience", "tech_agent")
    graph.add_edge("inject_experience", "chain_agent")
    graph.add_edge("inject_experience", "news_agent")
    graph.add_edge("inject_experience", "macro_agent")

    # fan-in: 所有分析完成后进入辩论
    graph.add_edge("tech_agent", "cross_challenge")
    graph.add_edge("chain_agent", "cross_challenge")
    graph.add_edge("news_agent", "cross_challenge")
    graph.add_edge("macro_agent", "cross_challenge")

    # 辩论循环
    graph.add_edge("cross_challenge", "check_convergence")
    graph.add_conditional_edges("check_convergence", convergence_router, {
        "converged": "verdict",
        "continue": "cross_challenge",
    })

    # 风控门控
    graph.add_edge("verdict", "risk_gate")
    graph.add_conditional_edges("risk_gate", risk_router, {
        "approved": "execute",
        "rejected": "record_rejection",
    })
    graph.add_edge("execute", END)
    graph.add_edge("record_rejection", END)

    return graph.compile()
```

### 3.3 Agent 间数据传递

沿用 ai-hedge-fund 模式——共享 state dict，不通过 messages 传递结构化数据：

```python
# 每个 Agent 写入 state["data"]["analyses"][agent_id]
# merge_dicts 确保并行写入不冲突（各 Agent 写不同 key）

def tech_analyze(state: ArenaState) -> dict:
    snapshot = state["data"]["snapshot"]
    experience = state["data"].get("experience", {})
    # ... 分析逻辑 ...
    return {
        "data": {
            "analyses": {
                "tech_agent": {
                    "direction": "bullish",
                    "confidence": 0.75,
                    "reasoning": "...",
                    "key_factors": ["MA20 上穿 MA60", "RSI 55 中性偏多"],
                    "risk_flags": ["成交量未放大"],
                }
            }
        }
    }
```

### 3.4 辩论收敛判定

```python
def check_stability(state: ArenaState) -> str:
    """自适应停止：观点稳定就停，最多 3 轮"""
    round_num = state["debate_round"]
    max_rounds = state["max_debate_rounds"]

    if round_num >= max_rounds:
        return "converged"

    # 计算本轮 vs 上轮的分歧变化
    analyses = state["data"]["analyses"]
    directions = [a["direction"] for a in analyses.values()]
    confidences = [a["confidence"] for a in analyses.values()]

    # 分歧度 = confidence-weighted direction 的标准差
    divergence = compute_divergence(directions, confidences)
    prev_divergence = state["divergence_scores"][-1] if state["divergence_scores"] else float("inf")

    # 分歧度变化 < 10% → 收敛
    if abs(divergence - prev_divergence) / max(prev_divergence, 0.01) < 0.1:
        return "converged"

    return "continue"
```

---

## 4. Intelligence Layer（智能决策层）

### 4.1 Agent 设计哲学

**数据驱动辩论，不是角色扮演。**

TradingAgents 用预设立场（Bull/Bear Researcher）驱动辩论。我们用数据分歧驱动——4 个 Agent 各自掌握不同维度的数据，天然产生信息不对称，这才是辩论的真正价值。

TechAgent 看到上升趋势，但 ChainAgent 发现巨鲸在抛——这种分歧来自真实数据，不是人为预设。

### 4.2 四个分析 Agent

| Agent | 数据源 | 分析维度 | 输出 |
|-------|--------|---------|------|
| TechAgent | K线、技术指标 | 趋势、形态、动量、波动率 | 趋势判断 + 关键价位 |
| ChainAgent | Exchange Flow、巨鲸、资金费率、OI、TVL | 链上资金动向、杠杆水平 | 资金流向 + 异常检测 |
| NewsAgent | crypto 新闻、社交情绪 | 事件影响、市场情绪 | 情绪评分 + 关键事件 |
| MacroAgent | 利率、DXY、BTC dominance | 宏观环境、风险偏好 | 宏观评级 + 趋势 |

### 4.3 Agent 输出标准格式

```python
@dataclass
class AgentAnalysis:
    agent_id: str
    pair: str                                          # e.g. "BTC/USDT"
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float                                  # 0.0 - 1.0
    reasoning: str                                     # 自然语言推理链
    key_factors: list[str]                             # 支撑判断的关键因素
    risk_flags: list[str]                              # 识别到的风险
    data_points: dict[str, Any]                        # 原始数据引用
    timestamp: datetime
```

### 4.4 交叉质询机制

辩论 prompt 分两层（参考 NOFX 源码）：

**Round 1 prompt（独立分析）：**
```
你是 {agent_role}，基于以下数据分析 {pair} 的交易机会。
{data_snapshot}
{historical_experience}  ← Verbal Reinforcement 注入

输出你的判断，包括方向、置信度、关键因素和风险。
```

**Round 2+ prompt（交叉质询）：**
```
你是 {agent_role}。以下是其他分析师的判断：
{other_analyses}

你必须：
1. 指出其他分析师判断中最薄弱的论据
2. 用你掌握的数据反驳或支持
3. 更新你自己的判断（可以改变方向和置信度）
4. 如果被说服，明确说明原因
```

### 4.5 Verdict（共识生成）

```python
def make_verdict(state: ArenaState) -> TradeVerdict:
    analyses = state["data"]["analyses"]

    # 1. Confidence-weighted 方向投票
    score = sum(
        a["confidence"] * (1 if a["direction"] == "bullish" else
                          -1 if a["direction"] == "bearish" else 0)
        for a in analyses.values()
    )

    # 2. 分歧度计算（来自 "Many Men, Many Minds" 论文）
    divergence = compute_divergence(analyses)

    # 3. 分歧度调节仓位
    #    高分歧 → 降低仓位或不交易
    if divergence > DIVERGENCE_THRESHOLD:
        return TradeVerdict(action="hold", reason="Agent 分歧过大")

    # 4. 生成交易信号
    position_scale = max(0, 1 - divergence)  # 分歧越大仓位越小
    return TradeVerdict(
        action="long" if score > 0 else "short" if score < 0 else "hold",
        confidence=abs(score) / len(analyses),
        position_scale=position_scale,
        divergence=divergence,
        reasoning=summarize_debate(analyses),
    )
```

### 4.6 Verbal Reinforcement（经验反哺，来自 FinCon NeurIPS 2024）

Decision Journal 不只是被动记录——每次决策前，检索相似市场条件下的历史经验，注入 Agent prompt。

```python
def verbal_reinforcement(state: ArenaState) -> dict:
    snapshot = state["data"]["snapshot"]

    # 从 Decision Journal 检索相似市场条件
    similar_commits = journal.search_similar(
        funding_rate=snapshot.market.funding_rate,
        exchange_flow=snapshot.onchain.exchange_netflow,
        volatility=snapshot.market.volatility,
        limit=3,
    )

    # 提取语言化经验
    experiences = []
    for commit in similar_commits:
        outcome = "盈利" if commit.pnl > 0 else "亏损"
        experiences.append(
            f"[{commit.timestamp.date()}] 类似市场条件下决策{commit.verdict.action}，"
            f"结果{outcome} {commit.pnl:.1%}。"
            f"复盘：{commit.retrospective}"
        )

    return {"data": {"experience": "\n".join(experiences)}}
```

### 4.7 模型分级策略

| 角色 | 模型 | 理由 | 单次成本 |
|------|------|------|---------|
| 4× 分析 Agent (Round 1) | GPT-4o-mini / Haiku | 结构化分析，便宜够用 | ~$0.02 |
| 4× 交叉质询 (Round 2) | GPT-4o-mini / Haiku | 结构化质疑 | ~$0.02 |
| Verdict 最终决策 | GPT-4o / Sonnet | 需要综合判断力 | ~$0.05 |
| **单次决策总计** | | | **~$0.10** |
| **月成本（日级3次×30天）** | | | **~$9** |

通过 litellm 统一接口，按 Agent 配置不同模型。

---

## 5. Data Layer（数据层）

### 5.1 数据源矩阵

| 数据源 | 内容 | API/库 | 成本 | 延迟 | 更新频率 |
|--------|------|--------|------|------|---------|
| 行情 | K线、Ticker、订单簿、资金费率 | ccxt | 免费 | 实时 | 按需 |
| DeFi TVL | 协议锁仓量、收益率 | DefiLlama | 免费无限，无需Key | 5-15min | 每5min |
| 衍生品 | OI、清算、资金费率历史 | CoinGlass (1000次/月) | 免费层 | 分钟级 | 每分钟 |
| 资金流 | Exchange In/Outflow | CryptoQuant | 免费层(日级) | 日级 | 每日 |
| 巨鲸 | 大额链上转账 | Whale Alert | 免费层(10次/min) | 分钟级 | 实时 |
| 新闻 | crypto 新闻聚合 | 自建爬虫(CoinDesk等) | 免费 | 小时级 | 每小时 |
| 社交情绪 | Twitter/Reddit 情绪 | snscrape + transformers | 免费 | 小时级 | 每小时 |
| 宏观 | 美联储利率、DXY | FRED API | 免费 | 日级 | 每日 |
| BTC Dominance | 市场份额 | CoinGecko | 免费 | 分钟级 | 每分钟 |

### 5.2 DataSnapshot 统一数据模型

```python
@dataclass
class MarketData:
    pair: str
    ohlcv: pd.DataFrame          # K线数据
    ticker: dict                  # 最新价格、成交量
    funding_rate: float           # 当前资金费率
    orderbook_imbalance: float    # 买卖盘不平衡度
    volatility: float             # 历史波动率

@dataclass
class OnchainData:
    exchange_netflow: float       # 交易所净流入（正=流入=卖压）
    whale_transfers: list[dict]   # 近期巨鲸转账
    open_interest: float          # 未平仓合约
    liquidations_24h: dict        # 24h 清算（long/short）
    defi_tvl: float               # DeFi TVL
    defi_tvl_change_7d: float     # 7日 TVL 变化率

@dataclass
class NewsSentiment:
    headlines: list[str]          # 近期标题
    sentiment_score: float        # -1.0 ~ 1.0
    key_events: list[str]         # 重大事件摘要
    social_buzz: float            # 社交热度

@dataclass
class MacroData:
    fed_rate: float               # 联邦基金利率
    dxy: float                    # 美元指数
    btc_dominance: float          # BTC 市场占比
    fear_greed_index: int         # 恐惧贪婪指数 0-100

@dataclass
class DataSnapshot:
    timestamp: datetime
    pair: str
    market: MarketData
    onchain: OnchainData
    news: NewsSentiment
    macro: MacroData
```

### 5.3 链上 Alpha 使用策略

基于学术研究（IEEE Access 2025, Herremans; SSRN 2025, Many Men Many Minds）：

**组合信号矩阵（单一信号 alpha 已衰减，必须组合使用）：**

| 信号组合 | 含义 | 交易倾向 |
|---------|------|---------|
| Exchange Inflow↑ + Funding Rate极高 + OI↑ | 多头过热，抛压+杠杆集中 | 强看空 |
| Exchange Outflow↑ + Funding Rate极低 + 清算偏空 | 空头过度，囤币+空头挤压 | 强看多 |
| TVL↓ + Exchange Inflow↑ + 巨鲸转入交易所 | 资金外逃，大户出货 | 看空 |
| TVL↑ + Exchange Outflow↑ + 巨鲸从交易所转出 | 生态恢复，大户囤币 | 看多 |
| OI急升 + 价格横盘 + Funding Rate中性 | 即将大波动，方向不确定 | 观望/减仓 |

**关键原则：**
- 极端市场条件下信号最有效，常规市场噪音大
- 短时间窗口（1-2h）比日级更有 edge（ResearchGate 2025 论文验证）
- 分歧度本身是信号——Agent 对链上数据解读分歧大时，市场不确定性高

---

## 6. Risk Gate（风控层）

### 6.1 设计原则

**硬约束，不是建议。纯规则，不用 LLM。确定性 100%。**

这是和 TradingAgents/ai-hedge-fund 最本质的区别——它们的风控是 LLM 软判断（"Risk Judge" 用 GPT 评估风险），概率性的，可能被 prompt 绕过。我们的风控是代码逻辑，不可绕过。

参考：Freqtrade trailing stop + Hummingbot kill switch + crypto_trade_service elastic gating。

### 6.2 11 项检查清单

```python
class RiskGate:
    """所有检查必须通过，任一失败则拒绝交易，记录拒绝原因到 Journal"""

    def __init__(self, config: RiskConfig, redis: Redis):
        self.checks = [
            MaxPositionSize(config),       # 1. 单币种最大仓位（占总资金比例）
            MaxTotalExposure(config),      # 2. 总敞口限制（所有持仓之和）
            DailyLossLimit(config),        # 3. 日亏损上限 → 触发 circuit breaker
            DrawdownLimit(config),         # 4. 最大回撤 → 强制平仓所有持仓
            CVaRCheck(config),             # 5. CVaR 风险评估（60天回看）
            CorrelationCheck(config),      # 6. 持仓相关性（避免同向集中）
            CooldownCheck(config, redis),  # 7. 同币种交易冷却期
            VolatilityGate(config),        # 8. 闪崩检测（5min 跌幅>5% → 暂停）
            FundingRateGate(config),       # 9. 资金费率异常（>0.1%/8h → 暂停）
            RateLimitCheck(config, redis), # 10. 交易频率限制
            ExchangeHealthCheck(config),   # 11. 交易所 API 健康（延迟>2s → 暂停）
        ]

    async def check(self, verdict: TradeVerdict, portfolio: Portfolio) -> GateResult:
        for check in self.checks:
            result = await check.evaluate(verdict, portfolio)
            if not result.passed:
                return GateResult(
                    passed=False,
                    rejected_by=check.name,
                    reason=result.reason,
                )
        return GateResult(passed=True)
```

### 6.3 关键风控机制详解

**Circuit Breaker（熔断器）：**
```python
class DailyLossLimit:
    """日亏损超限 → 熔断，需人工确认恢复"""
    async def evaluate(self, verdict, portfolio):
        daily_pnl = await self.get_daily_pnl(portfolio)
        if daily_pnl < -self.config.max_daily_loss_pct:
            await self.trigger_circuit_breaker()
            return CheckResult(passed=False,
                reason=f"日亏损 {daily_pnl:.1%} 超限 {self.config.max_daily_loss_pct:.1%}")
        return CheckResult(passed=True)
```

**CVaR（替代 VaR，加密市场厚尾分布）：**
```python
class CVaRCheck:
    """Conditional VaR，60天回看，每日重算"""
    async def evaluate(self, verdict, portfolio):
        returns = await self.get_returns(days=60)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        # 如果新仓位会使组合 CVaR 超限 → 拒绝
        projected_cvar = self.project_cvar(verdict, portfolio, cvar_95)
        if projected_cvar < -self.config.max_cvar:
            return CheckResult(passed=False,
                reason=f"CVaR {projected_cvar:.1%} 超限")
        return CheckResult(passed=True)
```

**闪崩检测：**
```python
class VolatilityGate:
    """5分钟窗口滑动监控，跌幅>5% → 暂停所有交易"""
    async def evaluate(self, verdict, portfolio):
        recent = await self.get_recent_prices(minutes=5)
        drop = (recent[-1] - max(recent)) / max(recent)
        if drop < -0.05:
            return CheckResult(passed=False,
                reason=f"闪崩检测：5min 跌幅 {drop:.1%}")
        return CheckResult(passed=True)
```

### 6.4 Redis 降级策略

```python
# 所有依赖 Redis 的检查（Cooldown、RateLimit）：
# Redis 不可用 → 默认拒绝交易（保守策略）
# 参考 Hummingbot kill switch 模式
try:
    cooldown_remaining = await redis.get(f"cooldown:{pair}")
except RedisError:
    logger.warning("Redis unavailable, rejecting trade (conservative)")
    return CheckResult(passed=False, reason="Redis 不可用，保守拒绝")
```

### 6.5 风控参数（可热更新）

```toml
# config/risk.toml
[position]
max_single_pct = 0.10          # 单币种最大仓位 10%
max_total_exposure_pct = 0.50  # 总敞口最大 50%

[loss]
max_daily_loss_pct = 0.03      # 日亏损上限 3%
max_drawdown_pct = 0.10        # 最大回撤 10% → 强制平仓
max_cvar_95 = 0.05             # CVaR(95) 上限 5%

[cooldown]
same_pair_minutes = 60         # 同币种冷却 60 分钟
post_loss_minutes = 120        # 亏损后冷却 120 分钟

[volatility]
flash_crash_threshold = 0.05   # 5分钟跌幅 5% 触发
funding_rate_threshold = 0.001 # 资金费率 0.1%/8h 触发

[exchange]
max_api_latency_ms = 2000      # API 延迟 2s 暂停
health_check_interval_s = 30   # 健康检查间隔 30s

[rate_limit]
max_trades_per_hour = 6        # 每小时最多 6 笔
max_trades_per_day = 20        # 每天最多 20 笔
```

---

## 7. Execution Layer（执行层）

### 7.1 订单状态机

```
                 place_order()
                      │
                      ▼
                 ┌─────────┐
                 │ PENDING  │
                 └────┬─────┘
                      │ submit_to_exchange()
                      ▼
                 ┌──────────┐
            ┌────│ SUBMITTED│────┐
            │    └──────────┘    │
            │         │          │
            ▼         ▼          ▼
     ┌──────────┐ ┌────────┐ ┌──────────┐
     │ CANCELLED│ │ FILLED │ │  FAILED  │
     └──────────┘ └────────┘ └──────────┘
                      │
                      ▼
              ┌───────────────┐
              │ PARTIALLY_    │
              │ FILLED        │
              └───────────────┘
```

```python
class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

VALID_TRANSITIONS = {
    OrderStatus.PENDING: {OrderStatus.SUBMITTED, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.SUBMITTED: {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.FILLED, OrderStatus.CANCELLED},
}
```

### 7.2 Exchange Adapter（双模式）

```python
class ExchangeAdapter(Protocol):
    async def place_order(self, order: Order) -> ExchangeResponse: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def get_order(self, order_id: str) -> ExchangeOrder: ...
    async def get_balance(self) -> dict: ...

class LiveExchange(ExchangeAdapter):
    """实盘：ccxt 封装"""
    def __init__(self, exchange_id: str, api_key: str, secret: str):
        self.exchange = getattr(ccxt, exchange_id)({
            "apiKey": api_key, "secret": secret,
            "enableRateLimit": True,
        })

class PaperExchange(ExchangeAdapter):
    """模拟盘：含滑点模型"""
    async def place_order(self, order: Order) -> ExchangeResponse:
        slippage = self.estimate_slippage(order)
        fill_price = order.price * (1 + slippage)
        # ...
```

### 7.3 滑点模型

```python
def estimate_slippage(self, order: Order) -> float:
    """简单滑点模型：成交量越大、流动性越差，滑点越大"""
    daily_volume = self.get_daily_volume(order.pair)
    order_ratio = order.amount * order.price / daily_volume
    base_slippage = 0.0005  # 0.05% 基础滑点
    impact = order_ratio * 0.1  # 市场冲击
    return base_slippage + impact
```

### 7.4 对账机制

```python
class Reconciler:
    """定期对账：本地状态 vs 交易所状态"""
    async def reconcile(self):
        local_orders = await self.db.get_open_orders()
        for order in local_orders:
            exchange_order = await self.exchange.get_order(order.exchange_id)
            if exchange_order.status != order.status:
                await self.sync_status(order, exchange_order)
                logger.warning(f"状态不一致: {order.id} local={order.status} exchange={exchange_order.status}")
```

---

## 8. Decision Journal（Git-like 决策追溯）

### 8.1 设计理念

参考 OpenAlice Wallet 的 commit chain（源码验证）+ FinCon 的 Verbal Reinforcement（NeurIPS 2024）。

Decision Journal 有三重角色：
1. **审计追溯** — 每笔决策的完整推理链，可回溯
2. **经验反哺** — Verbal Reinforcement 的数据源
3. **自我校准** — 回溯"哪些辩论结论是对的"，调整 Agent 权重

### 8.2 Commit 数据模型

```python
@dataclass
class DecisionCommit:
    # Git-like 元数据
    hash: str                          # SHA256[:8]
    parent_hash: str | None            # 链表，指向上一个 commit
    timestamp: datetime

    # 输入：当时的市场快照
    pair: str
    snapshot_summary: dict             # DataSnapshot 的关键指标摘要

    # 辩论过程
    analyses: dict[str, AgentAnalysis] # 各 Agent 的分析
    debate_rounds: int                 # 实际辩论轮数
    challenges: list[dict]             # 交叉质询记录
    divergence: float                  # 最终分歧度

    # 决策
    verdict: TradeVerdict              # 最终决策
    risk_gate: GateResult              # 风控结果（通过/拒绝+原因）

    # 执行
    order: Order | None                # 实际下单（风控拒绝则 None）
    fill_price: float | None           # 实际成交价
    slippage: float | None             # 实际滑点

    # 执行后状态
    portfolio_after: dict              # 组合快照

    # 事后复盘（异步填充）
    pnl: float | None                  # 该笔交易的盈亏
    retrospective: str | None          # 语言化复盘
```

### 8.3 Journal 操作

```python
class DecisionJournal:
    async def commit(self, data: dict) -> DecisionCommit:
        """创建新 commit，自动链接 parent"""

    async def log(self, limit=10, pair=None) -> list[DecisionCommit]:
        """类似 git log，按时间倒序"""

    async def show(self, hash: str) -> DecisionCommit:
        """类似 git show，查看单个 commit 详情"""

    async def diff(self, hash_a: str, hash_b: str) -> dict:
        """对比两次决策的差异"""

    async def search_similar(self, **market_conditions) -> list[DecisionCommit]:
        """检索相似市场条件下的历史决策（Verbal Reinforcement 用）"""

    async def update_pnl(self, hash: str, pnl: float, retrospective: str):
        """异步更新盈亏和复盘（交易结束后）"""

    async def accuracy_report(self, days=30) -> dict:
        """统计各 Agent 的预测准确率，用于权重校准"""
```

### 8.4 Agent 权重校准

```python
async def calibrate_weights(journal: DecisionJournal, days=30):
    """基于历史准确率调整 Agent 在 Verdict 中的权重"""
    report = await journal.accuracy_report(days)
    # report: {"tech_agent": 0.62, "chain_agent": 0.71, ...}
    # 准确率高的 Agent 在 verdict 中权重更大
    total = sum(report.values())
    return {agent: acc / total for agent, acc in report.items()}
```

---

## 9. 集成设计（双模式）

### 9.1 运行模式

```toml
# config/default.toml
[mode]
# standalone: 完整闭环（数据→辩论→风控→执行）
# api: 只暴露 HTTP API，不自动执行
# external: 输出决策给外部系统，风控和执行由外部处理
mode = "standalone"

[execution]
# paper: 模拟盘（含滑点模型）
# live: 实盘（ccxt）
engine = "paper"
```

### 9.2 独立使用

```bash
# CLI 模式
arena run --pair BTC/USDT --mode paper
arena run --pair BTC/USDT ETH/USDT --mode live --exchange binance
arena journal log --limit 10
arena journal show abc12345
arena backtest --pair BTC/USDT --start 2025-01-01 --end 2025-12-31

# API 模式
arena serve --port 8003
```

### 9.3 集成到 crypto_trade_service

```
crypto_trade_service/
├── services/
│   ├── backend/          # 现有，不动
│   ├── market-data/      # 现有，不动
│   └── ai-arena/         # 新增，独立服务
│       └── app/main.py   # FastAPI，端口 8003
```

**API 接口：**

```python
# POST /analyze
# 输入：pair + 可选的外部数据
# 输出：标准化决策信号

@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """供外部系统调用的分析接口"""
    return AnalyzeResponse(
        pair=request.pair,
        direction="bullish",        # bullish / bearish / neutral
        confidence=0.72,
        position_scale=0.85,        # 分歧度调节后的仓位比例
        divergence=0.15,            # Agent 分歧度
        reasoning="TechAgent 看多（MA 金叉+RSI 55），ChainAgent 确认（巨鲸净流出）...",
        risk_flags=["资金费率偏高 0.08%"],
        debate_rounds=2,
        timestamp="2026-02-20T10:30:00Z",
    )

# GET /journal/log
# GET /journal/{hash}
# GET /health
# GET /metrics
```

### 9.4 集成模式建议：AI 验证（起步）

```python
# crypto_trade_service 的 decision engine 中加一步：
async def make_decision(self, signal: Signal) -> Decision:
    # 1. 现有 Freqtrade 策略产出信号
    strategy_signal = await self.strategy.evaluate(signal)

    # 2. AI 二次验证（新增）
    ai_result = await httpx.post("http://localhost:8003/analyze", json={
        "pair": signal.pair,
        "timeframe": signal.timeframe,
    })

    # 3. 综合判断
    if strategy_signal.direction == ai_result.direction:
        confidence_boost = 1.2  # 方向一致，加强信心
    else:
        confidence_boost = 0.5  # 方向冲突，降低信心

    return Decision(
        direction=strategy_signal.direction,
        confidence=strategy_signal.confidence * confidence_boost,
        ai_validation=ai_result,
    )
```

---

## 10. 项目结构

```
cryptotrader-ai/
├── README.md
├── LICENSE (MIT)
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── Makefile
│
├── config/
│   ├── default.toml           # 默认配置
│   ├── exchanges.toml         # 交易所 API 配置
│   └── risk.toml              # 风控参数（可热更新）
│
├── src/
│   └── cryptotrader/
│       ├── __init__.py
│       │
│       ├── data/              # Layer 1: 数据层
│       │   ├── __init__.py
│       │   ├── market.py      # ccxt 行情采集
│       │   ├── onchain.py     # DefiLlama + CoinGlass + CryptoQuant
│       │   ├── news.py        # 新闻爬虫 + 情绪分析
│       │   ├── macro.py       # FRED + CoinGecko 宏观数据
│       │   ├── snapshot.py    # DataSnapshot 聚合
│       │   └── providers/     # 数据源适配器
│       │       ├── defillama.py
│       │       ├── coinglass.py
│       │       ├── cryptoquant.py
│       │       └── whale_alert.py
│       │
│       ├── agents/            # Layer 2: 分析 Agent
│       │   ├── __init__.py
│       │   ├── base.py        # Agent 基类 + 标准输出格式
│       │   ├── tech.py        # 技术面 Agent
│       │   ├── chain.py       # 链上数据 Agent
│       │   ├── news.py        # 新闻情绪 Agent
│       │   └── macro.py       # 宏观 Agent
│       │
│       ├── debate/            # Layer 3: 辩论层
│       │   ├── __init__.py
│       │   ├── challenge.py   # 交叉质询 prompt 构建
│       │   ├── convergence.py # 稳定性检测 + 分歧度计算
│       │   └── verdict.py     # 共识生成 + 仓位调节
│       │
│       ├── risk/              # Layer 4: 风控层
│       │   ├── __init__.py
│       │   ├── gate.py        # RiskGate 主入口
│       │   ├── checks/        # 11 项检查
│       │   │   ├── __init__.py
│       │   │   ├── position.py      # MaxPositionSize + MaxTotalExposure
│       │   │   ├── loss.py          # DailyLossLimit + DrawdownLimit
│       │   │   ├── cvar.py          # CVaR 检查
│       │   │   ├── correlation.py   # 持仓相关性
│       │   │   ├── cooldown.py      # 交易冷却期
│       │   │   ├── volatility.py    # 闪崩检测 + 资金费率异常
│       │   │   ├── rate_limit.py    # 交易频率限制
│       │   │   └── exchange.py      # 交易所健康检查
│       │   └── state.py       # Redis 风控状态管理
│       │
│       ├── execution/         # Layer 5: 执行层
│       │   ├── __init__.py
│       │   ├── order.py       # OrderManager + 状态机
│       │   ├── exchange.py    # LiveExchange (ccxt 封装)
│       │   ├── simulator.py   # PaperExchange (含滑点模型)
│       │   └── reconcile.py   # 对账
│       │
│       ├── journal/           # Decision Journal
│       │   ├── __init__.py
│       │   ├── commit.py      # DecisionCommit 模型
│       │   ├── store.py       # 存储（PostgreSQL）
│       │   ├── search.py      # 相似条件检索
│       │   └── calibrate.py   # Agent 权重校准
│       │
│       ├── learning/          # 经验学习
│       │   ├── __init__.py
│       │   └── verbal.py      # Verbal Reinforcement
│       │
│       ├── graph.py           # LangGraph 主编排
│       ├── models.py          # 全局数据模型
│       └── config.py          # 配置加载
│
├── src/api/                   # FastAPI 服务
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│       ├── analyze.py         # POST /analyze
│       ├── journal.py         # GET /journal/*
│       └── health.py          # GET /health + /metrics
│
├── src/cli/                   # CLI
│   ├── __init__.py
│   └── main.py               # arena run / journal / backtest
│
├── tests/
│   ├── unit/
│   │   ├── test_agents/
│   │   ├── test_debate/
│   │   ├── test_risk/
│   │   └── test_execution/
│   ├── integration/
│   │   ├── test_graph.py      # 完整流程集成测试
│   │   └── test_api.py
│   └── fixtures/
│       ├── snapshots/         # 测试用 DataSnapshot
│       └── exchanges/         # Mock 交易所响应
│
└── docs/
    ├── architecture.md        # 本文档
    ├── getting-started.md
    ├── configuration.md
    ├── risk-management.md
    └── contributing.md
```

---

## 11. 技术选型

| 组件 | 选型 | 版本 | 理由 |
|------|------|------|------|
| Agent 编排 | LangGraph | ≥0.2 | 状态机+并行+条件循环，TradingAgents/ai-hedge-fund 验证 |
| LLM 统一接口 | litellm | latest | 100+ provider，按 Agent 配模型 |
| 交易 | ccxt | ≥4.0 | 100+ 交易所统一接口 |
| API 框架 | FastAPI | ≥0.115 | 异步，自动文档，Pydantic v2 |
| ORM | SQLAlchemy 2.0 | async | Decision Journal 存储 |
| 数据库 | PostgreSQL | ≥15 | 决策记录、回测数据 |
| 缓存 | Redis | ≥7 | 风控状态、冷却计时 |
| 技术指标 | pandas-ta | latest | 纯 Python，无 C 依赖，安装简单 |
| HTTP 客户端 | httpx | latest | 异步，链上 API 调用 |
| 配置 | tomli + pydantic | — | TOML 配置 + 类型验证 |
| CLI | typer | latest | 类型安全的 CLI 框架 |
| 测试 | pytest + pytest-asyncio | — | 异步测试 |
| 包管理 | uv | latest | 快速依赖解析 |

---

## 12. MVP 路线图

### Phase 1: 最小闭环（2-3 周）

**目标：跑通一次完整的"数据→分析→辩论→风控→模拟盘执行"流程。**

```
Week 1: 骨架 + 数据层
├── 项目初始化（pyproject.toml, 目录结构, CI）
├── LangGraph 编排骨架（graph.py + ArenaState）
├── 数据层：ccxt 行情（K线 + Ticker + 资金费率）
└── DataSnapshot 模型

Week 2: Agent + 辩论 + 风控
├── TechAgent（技术面分析）
├── ChainAgent（链上数据，先只接 ccxt 资金费率）
├── 2 轮交叉质询 + 稳定性检测
├── Verdict（加权共识 + 分歧度）
├── RiskGate：MaxPositionSize + DailyLossLimit
└── PaperExchange（模拟盘 + 滑点模型）

Week 3: 闭环 + CLI
├── OrderManager + 状态机
├── Decision Journal（基础 commit chain）
├── CLI：arena run --pair BTC/USDT --mode paper
├── 基础测试
└── README + 文档
```

**Phase 1 交付物：**
- `arena run --pair BTC/USDT --mode paper` 能跑通
- 2 个 Agent 辩论，2 项风控检查
- 模拟盘执行 + Decision Journal 记录
- 单元测试覆盖核心逻辑

**Phase 1 不包含：** 新闻、宏观、链上三件套、UI、实盘、API 服务、Verbal Reinforcement。

### Phase 2: 完整智能层（2-3 周）

- [ ] NewsAgent + MacroAgent
- [ ] 链上数据三件套（DefiLlama + CoinGlass + CryptoQuant）
- [ ] 完整 11 项风控
- [ ] Verbal Reinforcement（经验反哺）
- [ ] FastAPI 服务（/analyze + /journal）
- [ ] Agent 权重校准
- [ ] 多币种并行

### Phase 3: 实盘 + 优化（2-3 周）

- [ ] 实盘交易（Binance/OKX via ccxt）
- [ ] 对账系统
- [ ] 回测引擎
- [ ] 基础 Dashboard
- [ ] 性能优化（数据缓存、并发控制）
- [ ] 完善文档，开源发布

### Phase 4: 进化（长期）

- [ ] LLM + RL 结合（参考 Meta-RL-Crypto, FinCon）
- [ ] 多交易所套利
- [ ] DEX 支持
- [ ] 社区贡献的 Agent 插件系统

---

## 13. 关键设计决策（带依据）

| # | 决策 | 依据 | 替代方案及否决理由 |
|---|------|------|-------------------|
| 1 | 数据驱动辩论，不用预设 Bull/Bear | 4 Agent 各持不同数据源，天然信息不对称 | TradingAgents 的 Bull/Bear Researcher：预设立场是人为的，数据分歧是真实的 |
| 2 | 辩论 2 轮，最多 3 轮 | TradingAgents 默认 1 轮；NeurIPS 2025 自适应稳定性论文 | 固定 5 轮：边际收益递减，成本线性增长 |
| 3 | 分歧度作为仓位调节信号 | "Many Men, Many Minds" (SSRN 2025)：Agent 分歧度可预测资产定价 | 只用 confidence 加权：忽略了不确定性信息 |
| 4 | 硬风控，纯规则，不用 LLM | 真金白银场景，确定性 > 灵活性 | TradingAgents 的 LLM Risk Judge：概率性的，可能被绕过 |
| 5 | CVaR 替代 VaR | 加密市场厚尾分布，VaR 低估极端风险 | VaR：在正态分布假设下有效，加密市场不适用 |
| 6 | Git-like Decision Journal | OpenAlice Wallet 源码验证；commit chain 可追溯可 diff | 简单日志：无法结构化检索和对比 |
| 7 | Verbal Reinforcement | FinCon (NeurIPS 2024, 221引用)：语言化经验反哺提升决策 | 无经验学习：每次决策独立，不从历史中学习 |
| 8 | 链上信号组合使用 | IEEE Access 2025 (Herremans)：单一信号 alpha 衰减 | 单一信号策略：已被市场 price in |
| 9 | 混合模型分级 | 成本模型估算：纯 4o ~$90/月 vs 混合 ~$9/月 | 全用强模型：10x 成本，分析层不需要 |
| 10 | 双模式（独立+集成） | 开源用户直接用 + 公司项目无缝集成 | 只做独立：失去集成市场 |

---

## 14. 学术参考文献

### 核心参考（直接影响架构设计）

1. **FinCon** — Yu et al., NeurIPS 2024 (221引用). Verbal Reinforcement 机制 → 我们的经验反哺设计
2. **TradingAgents** — Xiao et al., arXiv 2024 (134引用). 多 Agent 交易框架 → 我们的 Agent 角色分工参考
3. **FinDebate** — Cai et al., ACL 2025. 三 Agent debate 协议 → 我们的辩论机制参考
4. **Many Men, Many Minds** — Zhang et al., SSRN 2025. Agent 分歧度量化 → 我们的分歧度信号
5. **BTC Whale+CryptoQuant** — Herremans, IEEE Access 2025 (16引用). 链上数据特征工程 → 我们的 ChainAgent 设计
6. **HedgeAgents** — Li et al., WWW 2025 (26引用). Balance-aware 仓位管理 → 我们的风控参考
7. **Agent Market Arena** — Qian et al., arXiv 2025. 实盘 benchmark → 我们的评估方法论
8. **debate-or-vote** — NeurIPS 2025 Spotlight. 辩论 vs 投票对比 → 确认辩论优于投票

### 进化方向参考

9. **Meta-RL-Crypto** — Wang et al., arXiv 2025. Meta-learning + RL → Phase 4 方向
10. **LLM-guided RL Trading** — Darmanin, 2025. LLM 为 RL 提供 reward shaping → Phase 4 方向
11. **FinRL Contests** — Wang et al., arXiv 2025. RL benchmark 含 crypto → 评估参考

### 开源项目参考

12. **ai-hedge-fund** (45.8k⭐) — LangGraph fan-out/fan-in 编排模式
13. **TradingAgents** (30.2k⭐) — 辩论 + 风险辩论双层架构
14. **NOFX** (10.5k⭐) — 5 角色辩论 prompt 设计
15. **OpenAlice** (159⭐) — Git-like Wallet commit chain
16. **Freqtrade** (28k⭐) — Trailing stop + 止损体系
17. **Hummingbot** (8k⭐) — Kill switch 熔断机制

---

*文档结束。所有设计决策均有调研依据，可追溯到具体源码或论文。*
