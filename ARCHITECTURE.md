# CryptoTrader AI — 系统架构设计文档

> 版本：v7（2026-05-21；spec 022 stamp 后）
> 与 `docs/ARCHITECTURE.md` 分工：本文档侧重设计 rationale / 学术参考 / 市场定位 / MVP 路线；
> `docs/ARCHITECTURE.md` 侧重技术参考（数据模型 / 状态机 / FastAPI 端点 / 配置 schema）。
> 设计原则：agent prompt 只含 `skill (role + thinking)` + `snapshot (current data)`。
> 所有把 LLM 自生成内容反馈回下一次 prompt 的闭环都已移除（自指循环 → 噪声放大）。
> 反馈通道仅限：人工编辑 `agent_skills/_internal/<id>/SKILL.md`、trilogy 进化系统离线产出
> （spec 016–020c：Memory Evolution / Skill Evolution / Pareto / Git Lineage），
> + 实时 `live_steering`（一次性，cycle 结束失效）。

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
            │   Debate Gate    │  ← 渐进式过滤
            │ (共识/迷茫检测)    │    强共识或共同迷茫时跳过辩论
            └───┬──────────┬───┘
          skip  │          │debate
                ▼          ▼
        ┌──────────┐  ┌──────────────────┐
        │ Enrich   │  │ Cross-Challenge  │  ← 轮内并行
        │ Context  │  │ (2 轮辩论)        │
        └────┬─────┘  └────────┬─────────┘
             │                 ▼
             │           ┌──────────┐
             │           │ Enrich   │
             │           │ Context  │
             └─────┬─────┘
                   ▼
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
    graph.add_node("tag_regime", tag_regime_node)

    # ── 并行分析（fan-out）──
    graph.add_node("tech_agent", tech_analyze)
    graph.add_node("chain_agent", chain_analyze)
    graph.add_node("news_agent", news_analyze)
    graph.add_node("macro_agent", macro_analyze)

    # ── 辩论 ──
    graph.add_node("debate_gate", debate_gate)      # 渐进式过滤：强共识/共同迷茫时跳过
    graph.add_node("debate_round_1", debate_round)
    graph.add_node("debate_round_2", debate_round)
    graph.add_node("enrich_context", enrich_context)

    # ── 决策 + 执行 ──
    graph.add_node("verdict", make_verdict)
    graph.add_node("risk_gate", risk_check)
    graph.add_node("execute", place_order)
    graph.add_node("record_rejection", journal_rejection)

    # ── 边连接 ──
    graph.add_edge(START, "collect_data")
    graph.add_edge("collect_data", "tag_regime")

    # fan-out: regime 标签后并行分析
    graph.add_edge("tag_regime", "tech_agent")
    graph.add_edge("tag_regime", "chain_agent")
    graph.add_edge("tag_regime", "news_agent")
    graph.add_edge("tag_regime", "macro_agent")

    # fan-in: 所有分析完成后进入 debate_gate（渐进式过滤）
    graph.add_edge("tech_agent", "debate_gate")
    graph.add_edge("chain_agent", "debate_gate")
    graph.add_edge("news_agent", "debate_gate")
    graph.add_edge("macro_agent", "debate_gate")

    # debate_gate 条件路由：跳过辩论 or 进入两轮辩论
    graph.add_conditional_edges("debate_gate", debate_gate_router, {
        "debate": "debate_round_1",
        "skip": "enrich_context",
    })

    # 两轮辩论后汇入 enrich_context，enrich_context → verdict
    graph.add_edge("debate_round_1", "debate_round_2")
    graph.add_edge("debate_round_2", "enrich_context")
    graph.add_edge("enrich_context", "verdict")

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

**Round 1 prompt（独立分析，由 PromptBuilder 拼装）：**
```
你是 {agent_role}（system_prompt + available_skills，来自 config/agents/<id>.md 和 EvolvingSkillProvider）。
基于以下数据分析 {pair} 的交易机会：
{snapshot}    # 当前 cycle 数据
{portfolio}   # 当前持仓
[用户实时引导]  # 仅在 Redis 队列非空时出现，cycle 结束失效

输出你的判断，包括方向、置信度、关键因素和风险。
```

Prompt **只含静态 skill + 当前 snapshot + 可选实时 steering**——不含任何
LLM 自生成的历史 dump 或 prior，保留 round-3 minimal-skill 反锚定属性。

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

### 4.6 Prompt 与技能系统

Agent prompt 的内容只由两类来源拼装：

1. **静态 Skill 文件**（`agent_skills/_internal/<id>/SKILL.md`，spec 022 重组）——
   人工维护、git tracked。每个 SKILL.md 由 frontmatter（name / scope / regime_tags /
   triggers_keywords / importance / predictive_value 等数字字段）+ Markdown body
   （角色 + 思路 + checklist）组成。**不含历史 case dump、不含方向预测、不含具体
   数字阈值**。`access_count` / `last_accessed_at` 写入 gitignored sidecar，保持
   主文件 git-friendly。对外协议层（`_external/<id>/SKILL.md`）走相同规范但不
   注入内部 prompt，只供 `GET /skill/<name>` 暴露给外部 agent。

2. **当前 Cycle Snapshot**（`snapshot_summary` + `trend_context`）—— OHLCV、
   indicator、funding、news、macro，纯数据。

EvolvingSkillProvider 在每个 agent 调用前做两层检索：

```python
# learning/evolution/skill_provider.py — get_available_skills 简化版
def get_available_skills(agent_id: str, snapshot: dict, k: int = 5) -> list[Skill]:
    # 第一层：scope 过滤（agent:<id> 或 shared）+ regime_tags 交集
    candidates = scope_filter(load_all_skills(), agent_id, snapshot.regime_tags)
    # 第二层：idf + importance × predictive_value + recency
    for s in candidates:
        s._score = (
            idf_score(s, query_keywords)
            + s.importance * s.predictive_value
            + recency_bonus(s.last_accessed_at)
        )
    return sorted(candidates, key=lambda s: -s._score)[:k]
```

被选中的 skill 写回 frontmatter `access_count++` / `last_accessed_at=now`——
这是唯一的自动反馈，**只影响下次检索排序，不改 prompt 内容**。

`tag_regime()` 把 snapshot 分类为 `high_funding / high_vol / trending_up / ...`
等离散标签，给 SkillProvider 做第一层过滤。

```python
# nodes/data.py — regime 标签节点（无历史 case 检索）
async def tag_regime_node(state: ArenaState) -> dict:
    config = load_config()
    summary = state["data"].get("snapshot_summary", {})
    regime_tags = tag_regime(summary, config.experience.regime_thresholds)
    return {"data": {"regime_tags": regime_tags}}
```

**Live Steering**：用户从前端 chat 实时给 agent 加引导，通过 Redis 队列
（`steering:<session>:<agent_id>`）传递。`_collect_steering()` 在 agent 调用前
drain 队列，PromptBuilder 在非空时把内容作为 `live_steering` section 注入
HumanMessage。一次性消费，cycle 结束自动失效——不形成 LLM-自动写回闭环。

### 4.7 模型分级策略

| 角色 | 模型 | 理由 | 单次成本 |
|------|------|------|---------|
| 4× 分析 Agent (Round 1) | GPT-4o-mini / Haiku | 结构化分析，便宜够用 | ~$0.02 |
| 4× 交叉质询 (Round 2) | GPT-4o-mini / Haiku | 结构化质疑 | ~$0.02 |
| Verdict 最终决策 | GPT-4o / Sonnet | 需要综合判断力 | ~$0.05 |
| **单次决策总计** | | | **~$0.10** |
| **月成本（日级3次×30天）** | | | **~$9** |

通过 LangChain ChatOpenAI 统一接口，按 Agent 配置不同模型。`create_llm()` 工厂自动处理 fallback 和缓存。

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
| 社交情绪 | CoinGecko 社交热度（Twitter 粉丝 / Reddit 订阅 / 情绪投票）| CoinGecko Community API | 免费 | 小时级 | 每小时 |
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

### 6.5 风控参数

> 全部位于 `config/default.toml` 的 `[risk.*]` 段；`config/local.toml` 可覆盖（gitignored）。
> 不存在独立的 `risk.toml`。

```toml
# 默认（多 pair 分散）
[risk.position]
max_single_pct = 0.10          # 单币种最大仓位 10%
max_total_exposure_pct = 0.50  # 总敞口最大 50%
max_margin_used_pct = 0.40

[risk.loss]
max_daily_loss_pct = 0.03      # 日亏损上限 3%
max_drawdown_pct = 0.10        # 最大回撤 10% → 强制平仓
max_cvar_95 = 0.05             # CVaR(95) 上限 5%

[risk.cooldown]
same_pair_minutes = 60         # 同币种冷却 60 分钟
post_loss_minutes = 120        # 亏损后冷却 120 分钟

[risk.volatility]
flash_crash_threshold = 0.05   # 5分钟跌幅 5% 触发
funding_rate_threshold = 0.005 # 资金费率 0.5% 触发

[risk.exchange]
max_api_latency_ms = 5000      # API 延迟 5s 才视为降级（OKX VPN 现实）

[risk.rate_limit]
max_trades_per_hour = 6
max_trades_per_day = 20

# ── BTC-only 集中模式生产覆盖（config/local.toml）──
# [risk.position]
# max_single_pct = 0.80
# max_total_exposure_pct = 4.00     # 5x 杠杆 × 80% single → 总 notional 上限
# max_margin_used_pct = 0.90
# max_same_direction_positions = 1  # BTC 是唯一 pair
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
        """检索相似市场条件下的历史决策（供 dashboard / 离线分析使用）"""

    async def update_pnl(self, hash: str, pnl: float, retrospective: str):
        """异步更新盈亏和复盘（交易结束后）"""
```

### 8.4 反馈通道

系统**不**自动把任何 LLM 生成的内容写回 prompt。反馈通道仅限：

- **Skill 检索热度**：每次 EvolvingSkillProvider 选中的 skill 累加
  `access_count` 并刷新 `last_accessed_at`，影响下次检索打分（不改 prompt 内容）
- **人工编辑 SKILL.md**：用户读 `decision_commits` 分析后手动改 git-tracked
  Markdown，提交后下个 cycle 即可生效
- **Live steering**：用户在前端 chat 给 agent 实时指令，本 cycle 即用即丢

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
# CLI 模式（生产推荐 perp 线性合约）
arena run --pair BTC/USDT:USDT --mode paper
arena run --pair BTC/USDT:USDT --mode live   # 默认走 config 的 exchange_id=okx
arena journal log --limit 10
arena journal show abc12345
arena backtest --pair BTC/USDT:USDT --start 2025-01-01 --end 2025-12-31

# API 模式
arena serve --port 8003   # FastAPI + 嵌入式 scheduler + watchdog
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

> **以下结构反映 spec 022 stamp 时（2026-05-21）的实际目录拓扑。
> 风控参数全部在 `config/default.toml` 的 `[risk.*]` 段（不存在 `risk.toml`）。**

```
cryptotrader-ai/
├── README.md / README_EN.md
├── LICENSE (MIT)
├── pyproject.toml
├── docker-compose.yml       # 6 service: postgres / redis / api / web / scheduler / caddy
├── Makefile
│
├── config/
│   ├── default.toml         # 全部默认配置（模式 / 模型 / 风控 / 调度器 / 数据源）
│   ├── local.toml           # 本地覆盖（API key / 生产参数；gitignored）
│   └── agents/<name>.md     # tech / chain / news / macro 的 system_prompt
│
├── agent_skills/            # spec 022 重组：内外协议分目录
│   ├── _internal/{tech,chain,news,macro,trading-knowledge}/SKILL.md
│   │                        # 注入 agent prompt 的内部能力（EvolvingSkillProvider 读）
│   └── _external/{cryptotrader,verdict-feed,market-intel,
│                  evolution-insights,execution-replay}/SKILL.md
│                            # 对外 Anthropic SKILL.md 协议（GET /skill/<name>）
│
├── agent_memory/            # trilogy 进化系统数据（spec 018-021）
│   └── {tech,chain,news,macro}/patterns/*.md
│
├── src/cryptotrader/
│   ├── data/                # 数据层：market / onchain / news / macro + providers/
│   ├── agents/              # base + tech/chain/news/macro + _indicators.py（纯 pandas/numpy）
│   ├── debate/              # challenge / convergence / verdict / researchers（bull/bear）
│   ├── nodes/               # LangGraph 节点函数（agents/data/debate/verdict/execution/journal）
│   ├── risk/                # gate.py（11 项顺序检查）+ state.py（Redis + 内存降级）
│   ├── execution/           # simulator / exchange（LiveExchange ccxt）/ order / reconcile
│   ├── portfolio/           # PortfolioManager（DB + 内存）
│   ├── journal/             # store / commit / events.py（spec 022 record_evolution_event）
│   ├── learning/            # regime + evolution/（skill_provider + idf）+ memory.py（spec 021/022）
│   ├── observability/       # cache_metrics / heartbeat_metrics（spec 022 3 gauge）
│   ├── ops/daemon.py        # evolution daemon（spec 020b；尚未在 compose 启用）
│   ├── backtest/            # engine / session / cache / historical_data
│   ├── graph.py             # 3 graph variants（full / lite / debate）
│   ├── state.py             # ArenaState TypedDict + build_initial_state()
│   ├── scheduler.py         # APScheduler + watchdog（spec post-022 fix）
│   └── config.py            # TOML 配置加载 + dataclass 校验
│
├── src/api/                 # FastAPI 服务
│   └── routes/
│       ├── decisions.py / portfolio.py / risk.py / scheduler.py
│       ├── chat.py / hitl.py / market.py / backtest.py
│       ├── memory.py        # skills + patterns（spec 022 closes 021 T021）
│       ├── events.py        # GET /api/events/heartbeat（spec 022）
│       ├── skills.py        # GET /skill/<name>（spec 022 外部 SKILL.md）
│       └── metrics.py       # Prometheus exporter（含 spec 022 新 3 gauge）
│
├── src/cli/main.py          # Typer CLI（arena run / serve / scheduler / backtest / live-check）
│
├── web/                     # React 19 + Vite 8 + TS 5.9 前端（仪表盘 / 决策 / 回测 / 风控）
│
├── tests/                   # 2279 tests（spec 022 stamp 基线）
│
├── specs/                   # spec-kit 历史归档（001–025；spec 016-022 = trilogy + agent-native）
│
└── docs/
    ├── ARCHITECTURE.md      # 技术参考（与本设计文档分工）
    ├── DEPLOYMENT.md        # 部署指南
    ├── PRD.md               # 产品需求
    ├── phases.md            # 阶段交付物清单
    ├── frontend-architecture.md
    ├── TRACING.md / logging-conventions.md / EDGE_CASES.md
```

---

## 11. 技术选型

| 组件 | 选型 | 版本 | 理由 |
|------|------|------|------|
| Agent 编排 | LangGraph | ≥1.0 | 状态机+并行+条件循环；3 graph variants（full / lite / debate）|
| LLM 统一接口 | LangChain ChatOpenAI | ≥1.2 | `create_llm()` 工厂，`with_fallbacks([fallback])` + `SQLiteCache`，按 agent 配模型 |
| 交易 | ccxt | ≥4.x | OKX perp 主战场（spec 013 起统一 `BTC/USDT:USDT` canonical 符号）|
| API 框架 | FastAPI | ≥0.135 | 异步，自动 OpenAPI，Pydantic v2 |
| ORM | SQLAlchemy 2.0 | async | Decision Journal 存储 |
| 数据库 | PostgreSQL | 16 | 决策记录、portfolio_snapshots、回测会话 |
| 缓存 | Redis | 7 | 风控状态、冷却计时、live_steering 队列、限流 fixed-window |
| 技术指标 | `agents/_indicators.py` | — | 纯 pandas/numpy 实现（替代 pandas-ta，零原生依赖）|
| 调度 | APScheduler | ≥3.10 | IntervalTrigger 4h + 自研 watchdog（spec post-022 fix）|
| HTTP 客户端 | httpx | latest | 异步链上 API 调用 |
| 配置 | tomli + dataclass | — | TOML 配置 + 类型验证 |
| CLI | typer + rich | latest | 类型安全的 CLI 框架 |
| 测试 | pytest + pytest-asyncio | — | `asyncio_mode = "auto"`，2279 tests collected |
| 包管理 | uv | latest | 快速依赖解析；litellm 已删除 |
| 前端 | React 19 + Vite 8 + TS 5.9 | — | strict 模式 + Zustand + React Query + Radix UI |
| 可观测性 | OpenTelemetry + Prometheus | — | OTel structlog 上下文 + Prometheus exporter（含 spec 022 3 gauge）|

---

## 12. 演进路线图（含已落地里程碑）

> 本节是历史 + 未来的合并视图；按阶段标 ✅ 已完成 / 🟡 进行中 / ⬜ 未开始。每阶段详细交付物见 `docs/phases.md`。

### Phase 1 — 最小闭环 ✅
`arena run --pair BTC/USDT:USDT --mode paper` 跑通；2 个 agent 辩论 + 2 项风控；PaperExchange + Decision Journal。

### Phase 2 — 完整智能层 ✅
4 agent（tech/chain/news/macro）+ 链上数据三件套 + 完整 11 项风控 + FastAPI 服务（含 SSE 流式 chat）。

### Phase 3 — 实盘 + 回测 + Dashboard ✅
LiveExchange（ccxt OKX perp）+ server-side OCO 保护；回测引擎 + SQLite OHLCV 缓存；React 19 + Vite 8 仪表盘；APScheduler 周期循环。

### Phase 4 — Trilogy 进化系统 + Agent-Native（spec 016 → 022）✅

| Spec | 主题 | 主要交付物 |
|------|------|-----------|
| 016 | 项目研究 + daemon 决策 | 8 项研究综述 + D-ENG-01 daemon + D-ENG-02 lineage |
| 017a/b | PromptBuilder 基建 + 4 agent 集成 | 删 ROLE 硬编码 + middleware 模块 |
| 018 | Memory Evolution | 5-signal Maturity FSM + Pareto + IVE failure classification |
| 019 | Skill Evolution | EvolvingSkillProvider + D-RT-01 retrieval + LLM 自动 metadata 推断 |
| 020a | Trilogy Ops | cache observability + rollback runbook + staging_validate |
| 020b | Evolution Daemon | 独立进程（src/cryptotrader/ops/daemon.py）/ daily Pareto+Regime+Skill proposal / soft degrade |
| 020c | Git Lineage | GitLineageHook + evolution branch orphan + transitions batch commit |
| 021 | Pattern Cold-Start | `agent_memory/{agent}/patterns/*.md` + daemon `pattern_extraction` action |
| 022 | Agent-Native Skill Protocol | 5 `_external/SKILL.md` + `GET /skill/<name>` + `/api/memory/patterns` + `/api/events/heartbeat` + 3 Prometheus gauge |

累计 ~260 新测试 / 11 Prometheus Gauge / OTel 全覆盖。

### Phase 5 — 鲁棒性收尾（spec post-022 fix）✅

- `commit 07e105b`：OCO `sz` 用 total position size（修 OKX 51000 + 部分覆盖；16 场景实盘验证）
- `commit 57eb884`：scheduler watchdog（IntervalTrigger silent-miss 自愈，生产 3 次实战）
- `commit fc9211d`：execution close fallback `position_context`（OKX cooldown 时不再 silent drop）

### Phase 6 — 待规划 ⬜

- A/B Experiment 框架（spec 023 候选）
- OpenAPI 静态化 + demo external client（spec 022 deferred FR-022-7/8/9 + 23/24）
- LLM + RL 结合（参考 Meta-RL-Crypto / FinCon）
- 多交易所套利 / DEX 支持
- 社区 agent 插件系统

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
| 11 | 4h 周期（不是 1h） | 与 OKX perp 4h K 线 + funding 8h 对齐；1h 周期下 IntervalTrigger silent-miss 每小时 1 次；4h 让 timer 有更多余量 + LLM 成本降 75% | 1h 周期：触发更敏感但 silent miss + 噪音过载 |
| 12 | BTC-only 集中模式 | 多 pair 时 macro_concentration 频繁拦 BTC；alts SL 触发率高于 BTC（ATR/price ≈ 2-3% vs BTC 1.5%）；5x 杠杆 + 80% single cap 给 BTC 单 pair 充分仓位 | 多 pair 分散：风控更难配 + 实盘亏多胜少 |
| 13 | Scheduler Watchdog | APScheduler `IntervalTrigger.next_fire_time` 偶发卡在过去时间戳；`wakeup()` 不重新锚定。watchdog 每 5min 检查 `last_successful_cycle_at` 超过 `1.5×interval` → `modify_job` 强制 reschedule | 重启进程：人工介入 + 服务中断 |
| 14 | Close 风控豁免 + cooldown fallback | 减仓不应被风控阻断（"risk reduction must not be blocked"）；OKX cooldown 时 close 用 `position_context`（DB）fallback，避免 silent drop | 让 close 也走完整 risk_gate：仓位无法及时止损；fetch fail 时直接 return：journal 假象 commit |
| 15 | Trilogy 内/外协议分目录 | `_internal/` 注入 agent prompt；`_external/` 暴露给外部 AI agent（Codex / Cursor / Claude Code）。同一套 SKILL.md 规范，两个用途 | 单目录混用：外部 agent 看到内部 prompt + 容易拉错 skill |
| 16 | trilogy soft fail | 进化产出不阻塞 cycle（commit 失败 / LLM 失败 / lock timeout 全 soft fail） | 进化阻塞：单次进化失败 → 整个交易循环停止 |

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
