// Metrics, Chat, Market pages

// ── Metrics ────────────────────────────────────────────────────
function MetricsPage() {
  const m = MOCK.metrics;
  const maxBucket = Math.max(...m.latency_buckets);
  const maxCost = Math.max(...m.cost_14d);
  return (
    <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        <Card pad={16}><KPI label="LLM 调用 (24h)" value={m.llm_calls_24h.toLocaleString()}/></Card>
        <Card pad={16}><KPI label="成本 (24h)" value={`$${m.llm_cost_24h}`} sub={`均 $${(m.llm_cost_24h / m.llm_calls_24h * 1000).toFixed(2)}/千调用`}/></Card>
        <Card pad={16}><KPI label="Agent 成功率" value={`${(m.agent_success_rate * 100).toFixed(1)}%`} sub={`缓存命中 ${(m.cache_hit_rate * 100).toFixed(0)}%`}/></Card>
        <Card pad={16}><KPI label="P95 延迟" value={`${(m.p95_latency_ms / 1000).toFixed(1)}s`} sub={`均值 ${(m.avg_latency_ms / 1000).toFixed(1)}s`}/></Card>
      </div>

      <Card title="延迟直方图" subtitle="决策流水线总耗时分布">
        <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 180, padding: "8px 0" }}>
          {m.latency_buckets.map((c, i) => (
            <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
              <div className="mono" style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>{c}</div>
              <div style={{
                width: "100%", height: `${(c / maxBucket) * 140}px`,
                background: i < 6 ? "linear-gradient(180deg, var(--cyan-500), var(--cyan-600))" :
                                    i < 9 ? "linear-gradient(180deg, var(--warning), var(--amber-600))" :
                                             "linear-gradient(180deg, var(--short), color-mix(in oklch, var(--short) 50%, #000))",
                borderRadius: "3px 3px 0 0", minHeight: 2,
              }}/>
              <div style={{ fontSize: 9, color: "var(--fg-tertiary)", whiteSpace: "nowrap" }}>{m.latency_labels[i]}s</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="LLM 成本趋势 · 14 天">
        <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 120 }}>
          {m.cost_14d.map((c, i) => (
            <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div style={{
                width: "100%", height: `${(c / maxCost) * 100}px`,
                background: "var(--amber-500)", opacity: 0.3 + (i / 14) * 0.7,
                borderRadius: "2px 2px 0 0",
              }}/>
              <span className="mono" style={{ fontSize: 9, color: "var(--fg-tertiary)" }}>${c.toFixed(1)}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ── Chat ───────────────────────────────────────────────────────
function ChatPage() {
  const [input, setInput] = useState("");
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ padding: "14px 20px", borderBottom: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", gap: 12 }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 600 }}>多 Agent 会话</div>
          <div style={{ fontSize: 11, color: "var(--fg-tertiary)" }} className="mono">{MOCK.chat.session_id} · SSE 流</div>
        </div>
        <div style={{ flex: 1 }}/>
        <StatusPill tone="cyan" live>已连接</StatusPill>
      </div>

      <div style={{ flex: 1, overflow: "auto", padding: "20px 20px 80px", display: "flex", flexDirection: "column", gap: 14, maxWidth: 860, margin: "0 auto", width: "100%" }}>
        {MOCK.chat.messages.map((msg, i) => {
          if (msg.role === "user") {
            return (
              <div key={i} style={{ display: "flex", justifyContent: "flex-end" }}>
                <div style={{
                  maxWidth: "75%", padding: "10px 14px", borderRadius: "14px 14px 2px 14px",
                  background: "var(--amber-500)", color: "var(--fg-inverse)", fontSize: 13, fontWeight: 500,
                }}>{msg.content}</div>
              </div>
            );
          }
          if (msg.role === "system") {
            return (
              <div key={i} style={{ display: "flex", justifyContent: "center" }}>
                <div style={{ fontSize: 11, color: "var(--fg-tertiary)", fontStyle: "italic", display: "flex", alignItems: "center", gap: 6 }}>
                  <span className="ct-pulse" style={{ width: 4, height: 4, borderRadius: "50%", background: "var(--fg-tertiary)" }}/>
                  {msg.content}
                </div>
              </div>
            );
          }
          if (msg.role === "agent") {
            const a = AGENTS[msg.kind];
            return (
              <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                <AgentBadge kind={msg.kind} size={28}/>
                <div style={{ flex: 1, maxWidth: 640 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: a.color }}>{a.zh}</span>
                    <span style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>{a.role}</span>
                    <div style={{ flex: 1 }}/>
                    <span className="mono" style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>置信 {(msg.conf * 100).toFixed(0)}%</span>
                  </div>
                  <div style={{
                    padding: "10px 14px", borderRadius: "2px 14px 14px 14px",
                    background: "var(--bg-card)", border: `1px solid color-mix(in oklch, ${a.color} 20%, var(--border-subtle))`,
                    fontSize: 13, lineHeight: 1.55, textWrap: "pretty",
                  }}>{msg.content}</div>
                </div>
              </div>
            );
          }
          if (msg.role === "verdict") {
            return (
              <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", marginTop: 8 }}>
                <AgentBadge kind="verdict" size={28}/>
                <div style={{ flex: 1, maxWidth: 720 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: "var(--amber-500)" }}>AI 首席决策者</span>
                    <StatusPill tone="amber">{msg.action}</StatusPill>
                    <div style={{ flex: 1 }}/>
                    <span className="mono" style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>置信 {(msg.conf * 100).toFixed(0)}%</span>
                  </div>
                  <div style={{
                    padding: 14, borderRadius: "2px 14px 14px 14px",
                    background: "linear-gradient(135deg, color-mix(in oklch, var(--amber-500) 12%, transparent), var(--bg-card))",
                    border: "1px solid color-mix(in oklch, var(--amber-500) 35%, transparent)",
                    boxShadow: "var(--shadow-glow-amber)",
                    fontSize: 13, lineHeight: 1.6, fontWeight: 500, textWrap: "pretty",
                  }}>{msg.content}</div>
                </div>
              </div>
            );
          }
        })}
      </div>

      <div style={{ padding: "12px 20px", borderTop: "1px solid var(--border-subtle)", background: "var(--bg-raised)" }}>
        <div style={{ maxWidth: 860, margin: "0 auto", display: "flex", gap: 8, alignItems: "center",
                      background: "var(--bg-card)", border: "1px solid var(--border-default)", borderRadius: 10, padding: "6px 6px 6px 14px" }}>
          <input value={input} onChange={e => setInput(e.target.value)} placeholder="向 4 个 Agent 提问,或 @tech @chain @news @macro 指定..."
                 style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "var(--fg-primary)", fontSize: 13, padding: "8px 0" }}/>
          <button style={{ padding: "7px 12px", background: "var(--amber-500)", color: "var(--fg-inverse)", borderRadius: 6, fontSize: 12, fontWeight: 600, display: "flex", alignItems: "center", gap: 6 }}>
            <Icon d={icons.telegram} size={13} stroke={2}/>发送
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Market ─────────────────────────────────────────────────────
function MarketPage() {
  const m = MOCK.market;
  // Build candle-ish svg from price_24h
  const w = 720, h = 240;
  const vals = m.price_24h;
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = max - min || 1;
  const stepX = w / (vals.length - 1);
  const pts = vals.map((v, i) => [i * stepX, h - ((v - min) / range) * (h - 30) - 15]);
  const path = "M " + pts.map(p => p.map(n => n.toFixed(1)).join(",")).join(" L ");

  return (
    <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
        <Card pad={0}>
          <div style={{ padding: 18, borderBottom: "1px solid var(--border-subtle)", display: "flex", alignItems: "center", gap: 14 }}>
            <div style={{ width: 40, height: 40, borderRadius: 10, background: "linear-gradient(135deg, #f7931a, #e57c00)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700, fontSize: 18 }}>₿</div>
            <div>
              <div style={{ fontSize: 16, fontWeight: 600 }}>BTC/USDT</div>
              <div className="mono" style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>Binance · Perpetual</div>
            </div>
            <div style={{ flex: 1 }}/>
            <div style={{ textAlign: "right" }}>
              <div className="mono" style={{ fontSize: 24, fontWeight: 600, fontFamily: "var(--font-display)" }}>${m.price.toLocaleString()}</div>
              <div className="mono" style={{ fontSize: 12, color: m.delta_24h >= 0 ? "var(--long)" : "var(--short)", fontWeight: 500 }}>
                {m.delta_24h >= 0 ? "+" : ""}{m.delta_24h}% · 24h
              </div>
            </div>
          </div>
          <div style={{ padding: "8px 12px" }}>
            <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: "100%", height: 240 }}>
              <defs>
                <linearGradient id="mkt-grad" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="var(--cyan-500)" stopOpacity="0.35"/>
                  <stop offset="100%" stopColor="var(--cyan-500)" stopOpacity="0"/>
                </linearGradient>
              </defs>
              {[0.25, 0.5, 0.75].map((p, i) => (
                <line key={i} x1="0" x2={w} y1={h * p} y2={h * p} stroke="var(--border-subtle)" strokeDasharray="2 4" strokeWidth="0.5"/>
              ))}
              <path d={path + ` L ${w},${h} L 0,${h} Z`} fill="url(#mkt-grad)"/>
              <path d={path} fill="none" stroke="var(--cyan-500)" strokeWidth="1.5" strokeLinejoin="round"/>
            </svg>
          </div>
        </Card>

        <Card title="衍生品快照" subtitle="Binance Perpetual">
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div>
              <div style={{ fontSize: 10, color: "var(--fg-tertiary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>资金费率 · 8h</div>
              <div className="mono" style={{ fontSize: 18, color: "var(--warning)", fontFamily: "var(--font-display)" }}>+{(m.funding_rate * 100).toFixed(4)}%</div>
              <div style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>阈值 0.0300% · 接近中性</div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: "var(--fg-tertiary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>持仓量 (OI)</div>
              <div className="mono" style={{ fontSize: 18, fontFamily: "var(--font-display)" }}>${m.oi_usd}B</div>
              <div style={{ fontSize: 10, color: "var(--long)" }}>+4.2% (24h)</div>
            </div>
            <div>
              <div style={{ fontSize: 10, color: "var(--fg-tertiary)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>多空比</div>
              <div className="mono" style={{ fontSize: 18, fontFamily: "var(--font-display)" }}>{m.long_short_ratio}</div>
              <div style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>顶级交易者 {(m.top_traders_long_short * 100).toFixed(0)}% 做多</div>
            </div>
          </div>
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
        <Card title="24h 清算" subtitle="强平资金来源">
          <div style={{ display: "flex", gap: 12, height: 80, alignItems: "flex-end" }}>
            <div style={{ flex: 1, textAlign: "center" }}>
              <div style={{ height: `${(m.liquidations_24h.long / (m.liquidations_24h.long + m.liquidations_24h.short)) * 100}%`, background: "var(--long-soft)", border: "1px solid var(--long)", borderRadius: 4, marginBottom: 6 }}/>
              <div className="mono" style={{ fontSize: 12, color: "var(--long)" }}>${m.liquidations_24h.long}M</div>
              <div style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>多头爆仓</div>
            </div>
            <div style={{ flex: 1, textAlign: "center" }}>
              <div style={{ height: `${(m.liquidations_24h.short / (m.liquidations_24h.long + m.liquidations_24h.short)) * 100}%`, background: "var(--short-soft)", border: "1px solid var(--short)", borderRadius: 4, marginBottom: 6 }}/>
              <div className="mono" style={{ fontSize: 12, color: "var(--short)" }}>${m.liquidations_24h.short}M</div>
              <div style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>空头爆仓</div>
            </div>
          </div>
        </Card>
        <Card title="量" subtitle="24h 现货+合约">
          <div className="mono" style={{ fontSize: 22, fontFamily: "var(--font-display)", fontWeight: 600 }}>${m.volume_24h}B</div>
          <div style={{ fontSize: 11, color: "var(--fg-tertiary)", marginTop: 4 }}>较 7d 均值 +12%</div>
        </Card>
        <Card title="链上快照" subtitle="过去 48h">
          <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between" }}><span style={{ color: "var(--fg-tertiary)" }}>ETF 净流入</span><span className="mono" style={{ color: "var(--long)" }}>+$180M</span></div>
            <div style={{ display: "flex", justifyContent: "space-between" }}><span style={{ color: "var(--fg-tertiary)" }}>鲸鱼增持</span><span className="mono" style={{ color: "var(--long)" }}>+3,200 BTC</span></div>
            <div style={{ display: "flex", justifyContent: "space-between" }}><span style={{ color: "var(--fg-tertiary)" }}>稳定币市值</span><span className="mono">+0.8%</span></div>
            <div style={{ display: "flex", justifyContent: "space-between" }}><span style={{ color: "var(--fg-tertiary)" }}>恐惧贪婪</span><span className="mono" style={{ color: "var(--warning)" }}>54 · 中性</span></div>
          </div>
        </Card>
      </div>
    </div>
  );
}

window.MetricsPage = MetricsPage;
window.ChatPage = ChatPage;
window.MarketPage = MarketPage;
