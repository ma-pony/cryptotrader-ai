// Realistic mock data for CryptoTrader AI
// Mixes scenarios across pages: bull / choppy / circuit-breaker

// ── Portfolio & equity ─────────────────────────────────────────
const MOCK = {
  now: "2026-04-24 14:32:08 UTC",
  portfolio: {
    equity:          128_450.33,
    equity_delta_24h: 1.87,       // %
    equity_delta_7d: 6.42,
    cash:             42_180.00,
    unrealized_pnl:   3_291.44,
    realized_pnl_30d: 14_620.18,
    sharpe_90d:       2.14,
    max_drawdown:    -8.3,
    win_rate:         0.627,
    total_trades:     142,
  },

  // 180 days of equity curve (rough upward with volatility)
  equity_curve: (() => {
    const pts = []; let v = 100_000; let t = Date.now() - 180 * 86_400_000;
    for (let i = 0; i < 180; i++) {
      const trend = 0.0025;
      const noise = (Math.sin(i * 0.37) + Math.cos(i * 0.91) + (Math.random() - 0.5) * 2) * 0.018;
      v = v * (1 + trend + noise);
      pts.push({ t: t + i * 86_400_000, v: Math.round(v * 100) / 100 });
    }
    return pts;
  })(),

  positions: [
    { pair: "BTC/USDT", side: "long",  size: 0.42,   entry: 89_120, current: 93_480, notional: 39_261, pnl:  1_831.2, pnl_pct: 4.89, opened: "2d 4h", thesis: "4h MACD 金叉 + ETF 净流入第 5 日" },
    { pair: "ETH/USDT", side: "long",  size: 8.50,   entry:  3_120, current:  3_294, notional: 27_999, pnl:  1_479.0, pnl_pct: 5.58, opened: "18h", thesis: "DEX TVL 回升 + 资金费率回正" },
    { pair: "SOL/USDT", side: "short", size: 55.0,   entry:    182, current:    174, notional:  9_570, pnl:    440.0, pnl_pct: 4.40, opened: "9h", thesis: "生态基本面恶化 + 技术反弹到阻力" },
    { pair: "LINK/USDT", side: "long", size: 420,    entry:   18.40, current:   18.86, notional: 7_921, pnl:    193.2, pnl_pct: 2.50, opened: "1d 2h", thesis: "CCIP 叙事 + 巨鲸增持" },
  ],

  scheduler: {
    mode: "paper",          // "paper" | "live" | "backtest"
    status: "running",
    next_analysis: "02:47:13",  // countdown
    last_analysis: "2026-04-24 10:32:00",
    interval_hours: 4,
    triggered_runs_24h: 7,
    pair: "BTC/USDT",
  },

  // ── Decisions list ─────────────────────────────────────────
  decisions: [
    { id: "c5a8f2e",  t: "2026-04-24 10:32", pair: "BTC/USDT", action: "long",  conf: 0.78, scale: 0.60, status: "executed",  debate: "skipped", price: 92_810, pnl: +1_831.2 },
    { id: "b3d7c91",  t: "2026-04-24 06:32", pair: "BTC/USDT", action: "hold",  conf: 0.42, scale: 0.00, status: "approved",  debate: "skipped", price: 92_110, pnl: null },
    { id: "a1e9f30",  t: "2026-04-24 02:32", pair: "ETH/USDT", action: "long",  conf: 0.71, scale: 0.45, status: "executed",  debate: "2-round", price:  3_175, pnl: +1_479.0 },
    { id: "9f2c8e1",  t: "2026-04-23 22:32", pair: "BTC/USDT", action: "short", conf: 0.66, scale: 0.30, status: "rejected",  debate: "2-round", price: 91_940, pnl: null, reject: "CooldownCheck · 同对冷却中 22min" },
    { id: "8d4b7a6",  t: "2026-04-23 18:32", pair: "SOL/USDT", action: "short", conf: 0.73, scale: 0.35, status: "executed",  debate: "2-round", price: 180.4, pnl: +440.0 },
    { id: "7e1a2b3",  t: "2026-04-23 14:32", pair: "BTC/USDT", action: "hold",  conf: 0.38, scale: 0.00, status: "approved",  debate: "skipped (共同困惑)", price: 89_900, pnl: null },
    { id: "6b9c3d4",  t: "2026-04-23 10:32", pair: "LINK/USDT", action:"long",  conf: 0.69, scale: 0.40, status: "executed",  debate: "1-round", price:  18.30, pnl: +193.2 },
    { id: "5a8f7e2",  t: "2026-04-23 06:32", pair: "BTC/USDT", action: "close", conf: 0.82, scale: 0.00, status: "executed",  debate: "skipped", price: 88_420, pnl: -320.0 },
    { id: "4c7d9e0",  t: "2026-04-23 02:32", pair: "ETH/USDT", action: "long",  conf: 0.58, scale: 0.25, status: "rejected",  debate: "2-round", price:  3_092, pnl: null, reject: "DailyLossLimit · 今日已亏 2.9%" },
    { id: "3b6e1f4",  t: "2026-04-22 22:32", pair: "BTC/USDT", action: "short", conf: 0.45, scale: 0.00, status: "approved",  debate: "skipped", price: 88_100, pnl: null },
  ],

  // ── The star decision (c5a8f2e) — 8 section detail ─────────
  decision_detail: {
    id: "c5a8f2e · 03824",
    commit: "c5a8f2e39b1...",
    t: "2026-04-24 10:32:08 UTC",
    pair: "BTC/USDT",
    price: 92_810.22,
    position_before: { side: "flat", entry: null, size: 0 },

    verdict: {
      action: "long",
      confidence: 0.78,
      position_scale: 0.60,
      thesis: "ETF 净流入第 5 日 + 4h MACD 金叉 + 资金费率未过热,做多 60% 仓位跟随突破",
      reasoning: "四个分析师罕见地全部看多(方向一致),技术面和链上数据互相印证(ETF 累计净流入 $2.1B + OI 温和上升无爆仓风险)。宏观虽然 DXY 略走强但 FnG 从 38 升至 54 显示情绪修复。辩论门控跳过 —— 强共识无需辩论。风险:资金费率已到 0.018%,若突破 0.03% 需减仓。",
      invalidation: "4h 收盘跌破 $91,800 (上升趋势线) · 或资金费率突破 0.03%",
    },

    agents: [
      { kind: "tech", direction: "bullish", confidence: 0.82, sufficiency: "high",
        reasoning: "4h MACD 12h 前金叉后柱状图持续扩张;RSI 58 未超买;价格站稳 20MA($91,450);BB 中轨向上扩张。",
        factors: ["MACD 金叉确认", "RSI 58 健康", "20MA 支撑有效"],
        risks:   ["日线 RSI 65 接近超买", "周线阻力 $95k"],
        data: { RSI_4h: 58.3, MACD_hist: 124.6, SMA_20: 91_450, BB_width: 2.14, ATR_4h: 1_820 }
      },
      { kind: "chain", direction: "bullish", confidence: 0.76, sufficiency: "high",
        reasoning: "现货 ETF 连续 5 日净流入共 $2.1B;OI 温和上升 4.2% 无爆仓;鲸鱼地址过去 48h 净增持 3,200 BTC。",
        factors: ["ETF 净流入 $2.1B", "OI +4.2% 健康", "鲸鱼增持 3.2k BTC"],
        risks:   ["资金费率 0.018% 接近偏高"],
        data: { ETF_flow_5d: 2_120_000_000, OI_change: 0.042, whale_net: 3_200, funding_rate: 0.00018 }
      },
      { kind: "news", direction: "bullish", confidence: 0.64, sufficiency: "medium",
        reasoning: "主流媒体情绪从中性转正;贝莱德 CIO 看涨言论发酵;社交热度 Twitter 提及量 +18%。",
        factors: ["BlackRock CIO 看涨", "Twitter 热度 +18%", "监管无负面"],
        risks:   ["社交情绪 FOMO 初现", "VC 观点分化"],
        data: { sentiment_score: 0.34, mentions_24h: 48_200, fear_greed: 54 }
      },
      { kind: "macro", direction: "bullish", confidence: 0.58, sufficiency: "medium",
        reasoning: "FnG 从 38(恐惧)升至 54(中性偏贪);DXY 104.2 略升但在区间内;比特币主导率稳定 54%;无重要 Fed 事件。",
        factors: ["FnG 38→54 修复", "DXY 在区间", "无 Fed 冲击"],
        risks:   ["下周 CPI 数据", "稳定币流入放缓"],
        data: { fear_greed: 54, DXY: 104.2, BTC_dominance: 0.541, stablecoin_mcap_chg: 0.008 }
      },
    ],

    // Debate
    debate: {
      gate: { decision: "skipped", reason: "强共识:4/4 方向一致 + 均值置信度 0.70" },
      rounds: [],
    },

    // Risk gate audit — all 11
    risk_audit: [
      { n: 1,  name: "MaxPositionSize",      status: "pass", detail: "仓位 6.0% ≤ 10%" },
      { n: 2,  name: "MaxTotalExposure",     status: "pass", detail: "总敞口 42% ≤ 50%" },
      { n: 3,  name: "DailyLossLimit",       status: "pass", detail: "今日盈利 +$1,204" },
      { n: 4,  name: "DrawdownLimit",        status: "pass", detail: "当前回撤 2.1%" },
      { n: 5,  name: "CVaRCheck",            status: "pass", detail: "95% CVaR = 3.4%" },
      { n: 6,  name: "CorrelationCheck",     status: "pass", detail: "BTC 组 1/2 仓位" },
      { n: 7,  name: "CooldownCheck",        status: "pass", detail: "上次同对 4h 前" },
      { n: 8,  name: "VolatilityGate",       status: "pass", detail: "4h 波动 1.8% < 5%" },
      { n: 9,  name: "FundingRateGate",      status: "warn", detail: "资金费率 0.018% 接近阈值" },
      { n: 10, name: "RateLimitCheck",       status: "pass", detail: "今日 2/20 交易" },
      { n: 11, name: "ExchangeHealthCheck",  status: "pass", detail: "Binance 延迟 142ms" },
    ],

    // Historical experience injected
    experience: [
      { date: "2026-02-18", cond: "ETF 净流入 + MACD 金叉", verdict: "long · 0.72", pnl: "+$2,840", lesson: "类似条件下趋势延续是对的" },
      { date: "2026-01-04", cond: "FnG 修复 + OI 温和", verdict: "long · 0.68", pnl: "+$1,210", lesson: "资金费率未过热,安全加仓" },
      { date: "2025-11-22", cond: "ETF 流入 + RSI 偏高", verdict: "long · 0.75", pnl: "-$480",  lesson: "RSI 超 70 时应减半仓位" },
    ],

    bias: { detected: "过去 30 天 68% 做多倾向 · 注意确认偏差", severity: "low" },

    // Execution
    execution: {
      order_id: "ord_7f2a91c4",
      side: "buy", qty: 0.42, filled_price: 92_814.30, fees: 3.87,
      slippage_bps: 0.4, latency_ms: 182, exchange: "Binance",
    },

    latency: { data: 820, agents: 4_210, debate: 0, verdict: 1_340, risk: 95, execute: 182, total: 6_647 },
    tokens:  { input: 12_840, output: 3_210, cost_usd: 0.168 },
  },

  // ── Debate scenario (for decision 9f2c8e1, full 2-round) ────
  debate_scenario: {
    id: "9f2c8e1",
    pair: "BTC/USDT", price: 91_940,
    gate: { decision: "debate", reason: "低均值置信度 0.51 + 高分歧度 0.73 = 真分歧" },
    initial: [
      { kind: "tech",  dir: "bearish", conf: 0.68 },
      { kind: "chain", dir: "bullish", conf: 0.58 },
      { kind: "news",  dir: "bearish", conf: 0.42 },
      { kind: "macro", dir: "bullish", conf: 0.36 },
    ],
    // Round 1 + Round 2 — each a speech-bubble chain
    rounds: [
      {
        n: 1,
        turns: [
          { from: "tech", to: "chain",  critique: "你的 OI 上升论点忽略了 CVD 在 4h 的显著下滑 —— 资金进场但买盘薄弱。",
            dir: "bearish", conf: 0.71, move: "强化" },
          { from: "chain", to: "tech",   critique: "你看的 MACD 背离是日线噪音;4h 鲸鱼净增持才是领先指标,你缺失链上视角。",
            dir: "bullish", conf: 0.61, move: "强化" },
          { from: "news",  to: "macro",  critique: "FnG 修复只是短期弹簧,CPI 超预期的尾部风险被你低估了。",
            dir: "bearish", conf: 0.48, move: "强化" },
          { from: "macro", to: "news",   critique: "情绪指标已经在我的 FnG 模型里,你在重复计算。DXY 和稳定币我看到的信号更宏观。",
            dir: "neutral", conf: 0.42, move: "让步(由看多转中性)" },
        ],
      },
      {
        n: 2,
        turns: [
          { from: "tech",  to: "chain",  critique: "你的鲸鱼数据来自 48h 窗口,但过去 12h 同一批地址已在高位减持 800 BTC。",
            dir: "bearish", conf: 0.76, move: "强化" },
          { from: "chain", to: "tech",   critique: "承认短期减持,但 ETF 净流入仍在支撑。我把方向改为中性 —— 承认你的 4h 风险。",
            dir: "neutral", conf: 0.44, move: "让步(由看多转中性)" },
          { from: "news",  to: null,     critique: "我坚持看空;没有新证据能改变我的论点。",
            dir: "bearish", conf: 0.50, move: "保持" },
          { from: "macro", to: null,     critique: "我维持中性 —— 分歧没有收敛,但 CPI 不确定性压倒性。",
            dir: "neutral", conf: 0.40, move: "保持" },
        ],
      },
    ],
    convergence: { before: 0.73, after: 0.41, target: 0.5 },
    final_verdict: { action: "short", confidence: 0.62, scale: 0.30, thesis: "辩论后 2/4 看空 + 2/4 中性,无看多;做空 30% 仓位,止损 $92,500 (上方关键阻力)。" },
  },

  // ── Backtest ───────────────────────────────────────────────
  backtest: {
    sessions: [
      { id: "bt_2026-04-23", name: "BTC 180d LLM", mode: "llm", pair: "BTC/USDT", days: 180, status: "done", return_pct: 24.6, sharpe: 1.82, mdd: -11.2 },
      { id: "bt_2026-04-22", name: "ETH 90d LLM",  mode: "llm", pair: "ETH/USDT", days: 90,  status: "done", return_pct:  8.4, sharpe: 0.91, mdd: -15.8 },
      { id: "bt_2026-04-21", name: "BTC 180d SMA fallback",  mode: "sma", pair: "BTC/USDT", days: 180, status: "done", return_pct: 12.3, sharpe: 0.68, mdd: -18.2 },
    ],
    running: {
      id: "bt_2026-04-24_live", name: "BTC 365d 多周期对比", mode: "llm", pair: "BTC/USDT", days: 365,
      progress: 0.428, current_day: "2025-09-17", processed: 156, total: 365,
      decisions_so_far: 623, eta_seconds: 1_840,
      running_return: +17.2, running_sharpe: 1.54,
    },
  },

  // ── Risk state ─────────────────────────────────────────────
  risk: {
    circuit_breaker: { tripped: false, trigger: null, reset_available: false },
    daily_loss_pct: -0.8,    // negative = loss
    daily_loss_budget: 3.0,
    drawdown_pct: 2.1,
    drawdown_limit: 10.0,
    total_exposure_pct: 42.0,
    exposure_limit: 50.0,
    cvar_95: 3.4,
    cvar_limit: 5.0,
    correlation_groups: [
      { name: "BTC-correlated", open: 1, max: 2, pairs: ["BTC/USDT"] },
      { name: "ETH-correlated", open: 1, max: 2, pairs: ["ETH/USDT"] },
      { name: "L1-alt",         open: 1, max: 2, pairs: ["SOL/USDT"] },
      { name: "DeFi",           open: 1, max: 2, pairs: ["LINK/USDT"] },
    ],
    rate_limits: { hour: { used: 1, max: 6 }, day: { used: 2, max: 20 } },
    cooldowns: [
      { pair: "BTC/USDT", until: "28m",  kind: "same-pair" },
      { pair: "ETH/USDT", until: "—",    kind: "free" },
    ],
    redis_status: "healthy",
    recent_blocks: [
      { t: "04:32", id: "9f2c8e1", rule: "CooldownCheck",  detail: "BTC/USDT 同对冷却中 22min" },
      { t: "02:32", id: "4c7d9e0", rule: "DailyLossLimit", detail: "今日已亏 2.9% · 接近阈值 3.0%" },
      { t: "昨 22:32", id: "2a4b5c6", rule: "CorrelationCheck", detail: "BTC 组已有 2 仓位" },
    ],
  },

  // ── Metrics ───────────────────────────────────────────────
  metrics: {
    llm_calls_24h: 612,
    llm_cost_24h: 8.42,
    agent_success_rate: 0.984,
    avg_latency_ms: 6_840,
    p95_latency_ms: 11_200,
    decisions_per_day: 6,
    cache_hit_rate: 0.72,
    // latency histogram buckets
    latency_buckets: [ 4, 12, 28, 61, 144, 218, 186, 104, 48, 20, 8, 3 ],
    latency_labels:  ["<1s","1-2","2-3","3-5","5-7","7-10","10-13","13-16","16-20","20-25","25-30",">30"],
    // cost over 14d
    cost_14d: [3.2, 5.8, 6.1, 4.9, 7.2, 8.6, 8.1, 9.4, 7.8, 6.3, 8.9, 10.2, 8.8, 8.42],
  },

  // ── Chat session ──────────────────────────────────────────
  chat: {
    session_id: "sess_f4a8c2",
    messages: [
      { role: "user", content: "当前 BTC 资金费率在 0.018%,要不要减仓?" },
      { role: "system", content: "路由到 4 Agent 并行分析..." },
      { role: "agent", kind: "tech",  content: "资金费率本身不是技术信号,但 4h 图形成近端高点 — 短期回调风险存在。建议轻仓减持 20-30%,保留核心头寸。", conf: 0.62 },
      { role: "agent", kind: "chain", content: "0.018% 还在历史中性区间(阈值 0.03%)。OI 未见爆仓堆积,现货 ETF 昨日仍净流入 $180M。我倾向维持。", conf: 0.71 },
      { role: "agent", kind: "news",  content: "过去 2h 无重大负面新闻。Twitter 情绪指数平稳在 +0.31。", conf: 0.58 },
      { role: "agent", kind: "macro", content: "DXY 今日横盘,Fed 本周无讲话。宏观面无需动作。", conf: 0.54 },
      { role: "verdict", content: "综合四方意见,资金费率尚未触发减仓条件(0.03% 阈值)。建议维持当前 60% 仓位,设置 0.028% 为自动减仓触发器。若 4h 收盘跌破 $91,800 立即减半。", conf: 0.74, action: "hold + trigger-set" },
    ],
  },

  // ── Market view ───────────────────────────────────────────
  market: {
    pair: "BTC/USDT",
    price: 93_480.22,
    delta_24h: 2.18,
    volume_24h: 38.2, // $B
    funding_rate: 0.00018,
    oi_usd: 18.4, // $B
    long_short_ratio: 1.42,
    liquidations_24h: { long: 28.4, short: 62.1 }, // $M
    top_traders_long_short: 0.58,
    // mini sparkline values
    price_24h: (() => {
      const a = []; let v = 91_500;
      for (let i = 0; i < 96; i++) {
        v += (Math.random() - 0.42) * 240;
        a.push(v);
      }
      return a;
    })(),
  },
};

window.MOCK = MOCK;
