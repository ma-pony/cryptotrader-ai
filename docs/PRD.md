# CryptoTrader AI — 产品需求文档 (PRD)

> 多 Agent 加密货币交易系统，基于 LangGraph 编排，AI 驱动决策

---

## 1. 项目概述

一个 AI 驱动的加密货币交易系统。4 个专业 AI Agent 并行分析市场（技术面、链上、新闻、宏观），经过交叉辩论达成共识，由 AI 首席决策者做出交易判断，通过 11 项规则制风控门后执行交易。系统支持模拟交易、实盘交易和历史回测三种模式。

## 2. 核心问题

- **目标用户**：加密货币交易者（个人/机构）
- **用户场景**：7×24小时自动化监控市场，每4小时分析一次，给出 long/short/hold/close 决策
- **核心痛点**：
  - 人工无法同时处理技术面 + 链上数据 + 新闻情绪 + 宏观经济四个维度
  - 人工交易受情绪影响，容易在恐慌时卖出、贪婪时追涨
  - 需要 24 小时不间断监控，人力无法覆盖

## 3. 设计理念

**AI 为核心决策者**，不是执行器。系统给 AI 足够的上下文和自主权：
- Agent 分析用引导式 prompt 而非命令式规则
- Verdict AI 综合判断而非机械执行阈值
- 风控门是系统层安全网（灾难止损、熔断），不干预 AI 正常决策

## 4. 需求范围

**In-Scope**:
- 4-Agent 并行分析（Tech/Chain/News/Macro）
- 交叉辩论（2轮固定） + 收敛检测
- AI Verdict（首席决策者）
- 11 项规则制风控门（无 LLM 参与）
- 模拟交易（PaperExchange）
- 实盘交易（ccxt LiveExchange）
- 历史回测（LLM 驱动 + SMA 交叉 fallback）
- 决策日志（PostgreSQL + 内存 fallback）
- 言语强化学习（历史决策经验注入 prompt）
- CLI / FastAPI / Streamlit 三入口

**Out-of-Scope**:
- 多用户 / 权限管理
- 自动化参数优化 / 超参搜索
- 高频交易（最小间隔 1h）

---

## 5. 系统角色

| 角色 | 描述 | 实现 |
|------|------|------|
| TechAgent | 技术面分析师 — RSI/MACD/SMA/BB/ATR | `BaseAgent`，预计算指标后单次 LLM 调用 |
| ChainAgent | 链上/衍生品分析师 — OI/资金费率/鲸鱼转账/DEX TVL | `ToolAgent`，可调用实时 API 工具 |
| NewsAgent | 新闻情绪分析师 — RSS/关键词情绪/社交热度 | `ToolAgent`，可搜索新闻和社交数据 |
| MacroAgent | 宏观分析师 — Fed/DXY/恐惧贪婪/ETF/稳定币 | `BaseAgent`，数据预注入 |
| Verdict AI | 首席决策者 — 评估论证质量而非投票 | 独立 LLM 调用，temperature=0.1 |
| Risk Gate | 风控守门员 — 11 项硬规则检查 | 纯代码逻辑，无 LLM |

---

## 6. 核心业务流程

### 6.1 完整交易流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                      build_trading_graph()                       │
│                                                                  │
│  START                                                           │
│    │                                                             │
│    ▼                                                             │
│  collect_data ──► update_pnl ──► stop_loss_check                │
│                                      │                           │
│                           ┌──────────┴──────────┐                │
│                      loss < 5%             loss >= 5%            │
│                           │                     │                │
│                           ▼                     │                │
│                  verbal_reinforcement           │                │
│                           │                     │                │
│                           ▼                     │                │
│              ┌────────────┼────────────┐        │                │
│              ▼            ▼            ▼        │                │
│          TechAgent   ChainAgent   NewsAgent     │                │
│              │        MacroAgent       │        │                │
│              └────────────┼────────────┘        │                │
│                           ▼                     │                │
│                    debate_round_1                │                │
│                           │                     │                │
│                           ▼                     │                │
│                    debate_round_2                │                │
│                           │                     │                │
│                           ▼                     │                │
│                  enrich_verdict_context ◄────────┘                │
│                           │                                      │
│                           ▼                                      │
│                       verdict (AI)                               │
│                           │                                      │
│                           ▼                                      │
│                      risk_gate (11 checks)                       │
│                           │                                      │
│                  ┌────────┴────────┐                              │
│              approved          rejected                          │
│                  │                 │                              │
│                  ▼                 ▼                              │
│              execute        record_rejection                     │
│                  │                 │                              │
│                  ▼                 │                              │
│           record_trade             │                              │
│                  │                 │                              │
│                  └────────┬────────┘                              │
│                           ▼                                      │
│                          END                                     │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 三种图变体

| 变体 | 函数 | 用途 | 差异 |
|------|------|------|------|
| 完整图 | `build_trading_graph()` | 实盘/模拟 | 含辩论、风控、执行、日志 |
| 精简图 | `build_lite_graph()` | 回测 | 无辩论、无风控、无执行 |
| 辩论图 | `build_debate_graph()` | Bull/Bear 对抗 | 牛熊辩论 + 法官裁决 |

### 6.3 Agent 分析流程

每个 Agent 产出统一 JSON：
```json
{
  "direction": "bullish|bearish|neutral",
  "confidence": 0.0-1.0,
  "data_sufficiency": "high|medium|low",
  "reasoning": "2-3句引用具体数据",
  "key_factors": ["因素1", "..."],
  "risk_flags": ["风险1", "..."],
  "data_points": {"indicator": value}
}
```

**关键规则**：
- `data_sufficiency == "low"` → confidence 强制 ≤ 0.3
- `is_mock == True`（LLM 调用失败）→ verdict 前过滤掉
- 全部 mock → 强制 `action="hold"`

### 6.4 辩论机制

**交叉挑战辩论**（主流程使用）：
1. 每个 Agent 看到其他 3 个 Agent 的完整分析
2. 必须指出其他 Agent 最弱的论点
3. 必须捍卫自己的立场或说明被什么具体数据改变了
4. **反趋同规则**：不因"别人都这么想"而改变立场，必须有新证据
5. 固定 2 轮（避免人为趋同）

**牛熊对抗辩论**（`build_debate_graph()` 使用）：
1. Bull Researcher 构建最强买入论点
2. Bear Researcher 构建最强卖出论点
3. 交替反驳 2 轮
4. 法官 (Judge) 评估论证质量并裁决

### 6.5 Verdict 决策

Verdict AI 收到的完整上下文：
- **持仓状态**：FLAT / LONG(entry=$X, unrealized +Y%) / SHORT
- **价格趋势**：7d/14d/30d 变化、30d 价格区间、当前位置
- **风控约束**：最大仓位、剩余敞口、每日亏损预算、波动率、资金费率
- **偏差校正**：从历史日志检测到的认知偏差（如过度做多、时间段偏差）
- **4 个 Agent 的完整 JSON 报告**
- **上次决策上下文**（回测模式：防止关仓-开仓循环）

输出：
```json
{
  "action": "long|short|hold|close",
  "confidence": 0.0-1.0,
  "position_scale": 0.0-1.0,
  "reasoning": "2-3句",
  "thesis": "一句话交易论点",
  "invalidation": "论点失效的具体条件"
}
```

`position_scale` 直接控制仓位大小：`size = max(floor, scale × ceiling)`

### 6.6 风控门（11 项检查）

所有检查按顺序执行，第一个失败即拒绝。`close` 动作豁免所有检查。

| # | 检查 | 配置 | 说明 |
|---|------|------|------|
| 1 | MaxPositionSize | `max_single_pct=10%` | 单一仓位占比 |
| 2 | MaxTotalExposure | `max_total_exposure_pct=50%` | 总敞口 |
| 3 | DailyLossLimit | `max_daily_loss_pct=3%` | 日亏损限制 + 触发熔断 |
| 4 | DrawdownLimit | `max_drawdown_pct=10%` | 最大回撤 |
| 5 | CVaRCheck | `max_cvar_95=5%` | 95% 条件风险价值 |
| 6 | CorrelationCheck | 硬编码 14 组 | 同一关联组最多 2 个仓位 |
| 7 | CooldownCheck | `same_pair=60min`, `post_loss=120min` | 冷却期 |
| 8 | VolatilityGate | `flash_crash=5%` | 闪崩检测 |
| 9 | FundingRateGate | `threshold=0.5%` | 极端资金费率 |
| 10 | RateLimitCheck | `6/hour`, `20/day` | 交易频率 |
| 11 | ExchangeHealthCheck | `latency<2000ms` | API 健康 |

**Redis 失败行为**：若 Redis 配置了但不可用，Redis 依赖的检查（冷却、频率、日亏损）跳过而非阻断全部交易。

---

## 7. 言语强化学习

```
JournalStore
    │
    ▼
search_similar(fr, vol, limit=3)    # 搜索相似市场条件的历史决策
    │
    ▼
format_experience()                  # "Historical similar conditions:
    │                                #   - BTC @ 2024-11-15: verdict=long, pnl=+$320
    │                                #     Lesson: strong trend continuation was correct"
    ▼
注入每个 Agent 的 prompt           # agent 获得"前车之鉴"
    │
    ▼
detect_biases(store, days=30)       # 检测过去 30 天的认知偏差
    │
    ▼
generate_verdict_calibration()      # "你过去 30 天有 70% 做多倾向，注意确认偏差"
    │
    ▼
注入 Verdict prompt                 # verdict AI 获得"偏差校正"
```
