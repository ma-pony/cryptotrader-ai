// Dashboard page — portfolio overview, equity curve, positions, scheduler

function EquityCurve({ data, height = 220, accent = "var(--amber-500)" }) {
  const width = 800;
  if (!data?.length) return null;
  const vals = data.map(d => d.v);
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = max - min || 1;
  const stepX = width / (data.length - 1);
  const pts = data.map((d, i) => [i * stepX, height - ((d.v - min) / range) * (height - 20) - 10]);
  const path = "M " + pts.map(p => p.map(n => n.toFixed(1)).join(",")).join(" L ");
  const area = path + ` L ${width},${height} L 0,${height} Z`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ width: "100%", height, display: "block" }}>
      <defs>
        <linearGradient id="eq-grad" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={accent} stopOpacity="0.35"/>
          <stop offset="100%" stopColor={accent} stopOpacity="0"/>
        </linearGradient>
      </defs>
      {/* horizontal grid */}
      {[0.25, 0.5, 0.75].map((p, i) => (
        <line key={i} x1="0" x2={width} y1={height * p} y2={height * p}
              stroke="var(--border-subtle)" strokeDasharray="2 4" strokeWidth="0.5"/>
      ))}
      <path d={area} fill="url(#eq-grad)"/>
      <path d={path} stroke={accent} strokeWidth="1.5" fill="none" strokeLinejoin="round"/>
      <circle cx={pts[pts.length-1][0]} cy={pts[pts.length-1][1]} r="3" fill={accent}/>
      <circle cx={pts[pts.length-1][0]} cy={pts[pts.length-1][1]} r="6" fill={accent} opacity="0.2"/>
    </svg>
  );
}

function Dashboard() {
  const p = MOCK.portfolio;
  const [range, setRange] = useState("180D");
  const ranges = ["7D", "30D", "90D", "180D", "ALL"];

  return (
    <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 20, minHeight: "100%" }}>
      {/* Top row: 4 KPIs + scheduler status */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr 280px", gap: 12 }}>
        <Card>
          <KPI label="组合净值" value={`$${p.equity.toLocaleString()}`} delta={p.equity_delta_24h}
               sub={`7日 +${p.equity_delta_7d}%`} large
               trend={MOCK.equity_curve.slice(-30).map(d => d.v)} trendColor="var(--amber-500)"/>
        </Card>
        <Card>
          <KPI label="未实现盈亏" value={`+$${p.unrealized_pnl.toLocaleString()}`}
               sub={`已实现 (30D) +$${p.realized_pnl_30d.toLocaleString()}`}/>
        </Card>
        <Card>
          <KPI label="Sharpe (90D)" value={p.sharpe_90d.toFixed(2)}
               sub={`最大回撤 ${p.max_drawdown}%`}/>
        </Card>
        <Card>
          <KPI label="胜率" value={`${(p.win_rate * 100).toFixed(1)}%`}
               sub={`${p.total_trades} 次交易`}/>
        </Card>

        {/* Scheduler card — distinct "live" visual */}
        <Card pad={0} style={{ overflow: "hidden" }}>
          <div style={{
            padding: 14,
            background: "linear-gradient(135deg, color-mix(in oklch, var(--amber-500) 8%, transparent) 0%, transparent 100%)",
            borderBottom: "1px solid var(--border-subtle)",
            display: "flex", flexDirection: "column", gap: 10,
          }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="ct-pulse" style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--amber-500)", boxShadow: "0 0 8px var(--amber-glow)" }}/>
                <span style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 500 }}>下次分析</span>
              </div>
              <StatusPill tone="amber">模拟</StatusPill>
            </div>
            <div className="mono" style={{ fontSize: 28, fontFamily: "var(--font-display)", fontWeight: 600, letterSpacing: "-0.02em", color: "var(--amber-500)" }}>
              {MOCK.scheduler.next_analysis}
            </div>
            <div style={{ fontSize: 11, color: "var(--fg-tertiary)", display: "flex", justifyContent: "space-between" }}>
              <span>{MOCK.scheduler.pair} · {MOCK.scheduler.interval_hours}h 周期</span>
              <span>今日 {MOCK.scheduler.triggered_runs_24h}/24</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Equity curve */}
      <Card
        title="权益曲线"
        subtitle={`初始 $100,000 → 当前 $${p.equity.toLocaleString()} · 累计 +${((p.equity / 100_000 - 1) * 100).toFixed(1)}%`}
        right={
          <div style={{ display: "flex", gap: 2, padding: 2, background: "var(--bg-inset)", borderRadius: 6 }}>
            {ranges.map(r => (
              <button key={r} onClick={() => setRange(r)} style={{
                padding: "4px 10px", fontSize: 11, fontWeight: 500, borderRadius: 4,
                background: range === r ? "var(--bg-hover)" : "transparent",
                color: range === r ? "var(--fg-primary)" : "var(--fg-tertiary)",
              }}>{r}</button>
            ))}
          </div>
        }
      >
        <div style={{ padding: "8px 0" }}>
          <EquityCurve data={MOCK.equity_curve}/>
        </div>
      </Card>

      {/* Positions + Recent decisions */}
      <div style={{ display: "grid", gridTemplateColumns: "1.35fr 1fr", gap: 20 }}>
        <Card title="当前仓位" subtitle={`${MOCK.positions.length} 个 · 总敞口 ${MOCK.risk.total_exposure_pct}%`}
              right={<StatusPill tone="cyan">实时</StatusPill>}>
          <div style={{ display: "flex", flexDirection: "column", gap: 1, margin: "-4px -4px -4px -4px" }}>
            <div style={{
              display: "grid", gridTemplateColumns: "100px 60px 1fr 80px 80px 32px",
              gap: 12, padding: "6px 12px", fontSize: 10, textTransform: "uppercase",
              letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 500,
            }}>
              <div>交易对</div><div>方向</div><div>论点 · 持仓</div>
              <div style={{ textAlign: "right" }}>净值</div>
              <div style={{ textAlign: "right" }}>盈亏</div>
              <div/>
            </div>
            {MOCK.positions.map((pos, i) => (
              <div key={i} style={{
                display: "grid", gridTemplateColumns: "100px 60px 1fr 80px 80px 32px",
                gap: 12, padding: "10px 12px", alignItems: "center",
                borderTop: "1px solid var(--border-subtle)",
              }}>
                <div className="mono" style={{ fontWeight: 500 }}>{pos.pair}</div>
                <DirChip dir={pos.side}/>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: 12, color: "var(--fg-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{pos.thesis}</div>
                  <div className="mono" style={{ fontSize: 10, color: "var(--fg-tertiary)", marginTop: 2 }}>
                    {pos.size} @ ${pos.entry.toLocaleString()} · {pos.opened}
                  </div>
                </div>
                <div className="mono" style={{ textAlign: "right", fontSize: 12 }}>${pos.notional.toLocaleString()}</div>
                <div className="mono" style={{ textAlign: "right", fontSize: 12, color: pos.pnl >= 0 ? "var(--long)" : "var(--short)", fontWeight: 500 }}>
                  {pos.pnl >= 0 ? "+" : ""}${Math.abs(pos.pnl).toLocaleString()}
                  <div style={{ fontSize: 10, fontWeight: 400 }}>{pos.pnl >= 0 ? "+" : ""}{pos.pnl_pct}%</div>
                </div>
                <button style={{ color: "var(--fg-tertiary)", padding: 4 }}>
                  <Icon d={icons.chevRight} size={14}/>
                </button>
              </div>
            ))}
          </div>
        </Card>

        <Card title="最近决策" subtitle="最新 5 条"
              right={<button style={{ fontSize: 11, color: "var(--amber-500)" }}>查看全部 →</button>}>
          <div style={{ display: "flex", flexDirection: "column", gap: 1, margin: "-4px -4px -4px -4px" }}>
            {MOCK.decisions.slice(0, 5).map((d, i) => (
              <div key={d.id} style={{
                display: "flex", alignItems: "center", gap: 10, padding: "10px 12px",
                borderTop: i > 0 ? "1px solid var(--border-subtle)" : "none",
              }}>
                <div className="mono" style={{ fontSize: 10, color: "var(--fg-tertiary)", width: 36 }}>{d.t.slice(5, 10)}</div>
                <DirChip dir={d.action} confidence={d.conf}/>
                <div className="mono" style={{ fontSize: 11, color: "var(--fg-secondary)", flex: 1 }}>{d.pair}</div>
                {d.status === "rejected" && <StatusPill tone="danger">风控拒</StatusPill>}
                {d.status === "executed" && d.pnl != null && (
                  <span className="mono" style={{ fontSize: 11, color: d.pnl >= 0 ? "var(--long)" : "var(--short)", fontWeight: 500 }}>
                    {d.pnl >= 0 ? "+" : ""}${Math.abs(d.pnl).toFixed(0)}
                  </span>
                )}
                {d.status === "executed" && d.pnl == null && <StatusPill tone="default">持仓中</StatusPill>}
                {d.status === "approved" && d.action === "hold" && <StatusPill tone="default">观望</StatusPill>}
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

window.Dashboard = Dashboard;
window.EquityCurve = EquityCurve;
