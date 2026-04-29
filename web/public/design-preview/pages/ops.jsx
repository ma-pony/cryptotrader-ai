// Remaining pages: Backtest, Risk, Metrics, Chat, Market

// ── Backtest ───────────────────────────────────────────────────
function BacktestPage() {
  const [mode, setMode] = useState("llm");
  const [pair, setPair] = useState("BTC/USDT");
  const [days, setDays] = useState(180);
  const r = MOCK.backtest.running;

  return (
    <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Runner */}
      <Card title="新建回测" subtitle="LLM 驱动 + SMA 交叉 fallback">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 120px", gap: 12, alignItems: "flex-end" }}>
          <div>
            <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 500, marginBottom: 6 }}>模式</div>
            <div style={{ display: "flex", gap: 4, padding: 3, background: "var(--bg-inset)", borderRadius: 6 }}>
              {[{ id: "llm", l: "LLM 驱动" }, { id: "sma", l: "SMA 对照组" }].map(m => (
                <button key={m.id} onClick={() => setMode(m.id)} style={{
                  flex: 1, padding: "6px 10px", borderRadius: 4, fontSize: 12, fontWeight: 500,
                  background: mode === m.id ? "var(--bg-card)" : "transparent",
                  color: mode === m.id ? "var(--fg-primary)" : "var(--fg-tertiary)",
                  boxShadow: mode === m.id ? "var(--shadow-sm)" : "none",
                }}>{m.l}</button>
              ))}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 500, marginBottom: 6 }}>交易对</div>
            <select value={pair} onChange={e => setPair(e.target.value)} style={{
              width: "100%", padding: "7px 10px", background: "var(--bg-inset)",
              border: "1px solid var(--border-subtle)", borderRadius: 6, color: "var(--fg-primary)",
              fontFamily: "var(--font-mono)", fontSize: 12,
            }}>
              <option>BTC/USDT</option><option>ETH/USDT</option><option>SOL/USDT</option>
            </select>
          </div>
          <div>
            <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 500, marginBottom: 6 }}>时长</div>
            <div style={{ display: "flex", gap: 2 }}>
              {[30, 90, 180, 365].map(d => (
                <button key={d} onClick={() => setDays(d)} style={{
                  flex: 1, padding: "7px 0", fontSize: 11, fontWeight: 500, borderRadius: 4,
                  background: days === d ? "var(--bg-hover)" : "var(--bg-inset)",
                  color: days === d ? "var(--fg-primary)" : "var(--fg-tertiary)",
                }}>{d}d</button>
              ))}
            </div>
          </div>
          <button style={{
            padding: "9px 16px", background: "linear-gradient(135deg, var(--amber-500), var(--amber-600))",
            color: "var(--fg-inverse)", fontSize: 12, fontWeight: 600, borderRadius: 6,
            boxShadow: "var(--shadow-glow-amber)", display: "flex", alignItems: "center", gap: 6, justifyContent: "center",
          }}>
            <Icon d={icons.play} size={12} stroke={2.5}/> 启动回测
          </button>
        </div>
      </Card>

      {/* Running session */}
      <Card pad={0} style={{
        background: "linear-gradient(135deg, color-mix(in oklch, var(--amber-500) 5%, transparent), var(--bg-card) 60%)",
        border: "1px solid color-mix(in oklch, var(--amber-500) 25%, transparent)",
      }}>
        <div style={{ padding: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
            <span className="ct-pulse" style={{ width: 10, height: 10, borderRadius: "50%", background: "var(--amber-500)", boxShadow: "0 0 10px var(--amber-glow)" }}/>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600 }}>{r.name}</div>
              <div style={{ fontSize: 11, color: "var(--fg-tertiary)" }} className="mono">{r.id}</div>
            </div>
            <StatusPill tone="amber" live>运行中</StatusPill>
            <button style={{ padding: "6px 10px", fontSize: 12, color: "var(--fg-tertiary)", border: "1px solid var(--border-default)", borderRadius: 4 }}>
              <Icon d={icons.pause} size={11} stroke={2}/>
            </button>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 120px 120px 100px", gap: 16, marginBottom: 14, alignItems: "center" }}>
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--fg-tertiary)", marginBottom: 4 }}>
                <span>进度 · 当前 <span className="mono" style={{ color: "var(--fg-secondary)" }}>{r.current_day}</span></span>
                <span className="mono">{r.processed}/{r.total} 天 · {(r.progress * 100).toFixed(1)}%</span>
              </div>
              <Bar value={r.progress} color="var(--amber-500)" height={8} radius={4}/>
            </div>
            <KPI label="已处理决策" value={r.decisions_so_far}/>
            <KPI label="当前收益率" value={`+${r.running_return}%`}/>
            <KPI label="ETA" value={`${Math.floor(r.eta_seconds / 60)}m ${r.eta_seconds % 60}s`}/>
          </div>
        </div>
      </Card>

      {/* Historical sessions */}
      <Card title="历史会话">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 100px 120px 80px 80px 80px 80px", gap: 12, padding: "0 4px 8px", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 500 }}>
          <div>名称</div><div>模式</div><div>交易对 · 时长</div>
          <div style={{ textAlign: "right" }}>收益</div>
          <div style={{ textAlign: "right" }}>Sharpe</div>
          <div style={{ textAlign: "right" }}>MDD</div>
          <div/>
        </div>
        {MOCK.backtest.sessions.map(s => (
          <div key={s.id} style={{
            display: "grid", gridTemplateColumns: "1fr 100px 120px 80px 80px 80px 80px", gap: 12,
            padding: "10px 4px", alignItems: "center", borderTop: "1px solid var(--border-subtle)",
          }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>{s.name}</div>
              <div className="mono" style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>{s.id}</div>
            </div>
            <StatusPill tone={s.mode === "llm" ? "amber" : "default"}>{s.mode === "llm" ? "LLM" : "SMA"}</StatusPill>
            <div className="mono" style={{ fontSize: 12 }}>{s.pair} · {s.days}d</div>
            <div className="mono" style={{ textAlign: "right", fontSize: 12, color: s.return_pct >= 0 ? "var(--long)" : "var(--short)", fontWeight: 500 }}>
              +{s.return_pct}%
            </div>
            <div className="mono" style={{ textAlign: "right", fontSize: 12 }}>{s.sharpe}</div>
            <div className="mono" style={{ textAlign: "right", fontSize: 12, color: "var(--short)" }}>{s.mdd}%</div>
            <button style={{ fontSize: 11, color: "var(--amber-500)", textAlign: "right" }}>查看 →</button>
          </div>
        ))}
      </Card>
    </div>
  );
}

// ── Risk ───────────────────────────────────────────────────────
function RiskMeter({ label, value, limit, unit = "%", inverted = false }) {
  const pct = Math.abs(value) / limit;
  const tone = pct > 0.8 ? "danger" : pct > 0.5 ? "warning" : "success";
  const color = tone === "danger" ? "var(--short)" : tone === "warning" ? "var(--warning)" : "var(--long)";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 11, color: "var(--fg-tertiary)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 500 }}>{label}</span>
        <span className="mono" style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>上限 {limit}{unit}</span>
      </div>
      <div className="mono" style={{ fontSize: 22, fontWeight: 600, color, letterSpacing: "-0.02em", fontFamily: "var(--font-display)" }}>
        {value > 0 && !inverted ? "+" : ""}{value}{unit}
      </div>
      <div style={{ position: "relative", height: 6, background: "var(--bg-inset)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${Math.min(1, pct) * 100}%`, height: "100%", background: color, borderRadius: 3, transition: "width 0.3s" }}/>
      </div>
    </div>
  );
}

function RiskPage() {
  const r = MOCK.risk;
  return (
    <div style={{ padding: 20, display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Circuit breaker hero */}
      <Card pad={0} style={{
        background: r.circuit_breaker.tripped ?
          "linear-gradient(135deg, color-mix(in oklch, var(--short) 15%, transparent), var(--bg-card))" :
          "linear-gradient(135deg, color-mix(in oklch, var(--long) 6%, transparent), var(--bg-card))",
        border: `1px solid color-mix(in oklch, ${r.circuit_breaker.tripped ? "var(--short)" : "var(--long)"} 35%, transparent)`,
      }}>
        <div style={{ padding: 18, display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{
            width: 56, height: 56, borderRadius: 14,
            background: `color-mix(in oklch, ${r.circuit_breaker.tripped ? "var(--short)" : "var(--long)"} 18%, transparent)`,
            border: `1px solid color-mix(in oklch, ${r.circuit_breaker.tripped ? "var(--short)" : "var(--long)"} 40%, transparent)`,
            color: r.circuit_breaker.tripped ? "var(--short)" : "var(--long)",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Icon d={icons.shield} size={28} stroke={1.8}/>
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 500, marginBottom: 4 }}>熔断器状态</div>
            <div style={{ fontSize: 20, fontWeight: 600, fontFamily: "var(--font-display)" }}>
              {r.circuit_breaker.tripped ? "已触发 — 交易暂停" : "正常 · 所有闸门开放"}
            </div>
            <div style={{ fontSize: 12, color: "var(--fg-tertiary)", marginTop: 4 }}>
              Redis 状态: <span style={{ color: "var(--long)" }}>{r.redis_status}</span> · 11 项检查在线
            </div>
          </div>
          <button disabled={!r.circuit_breaker.tripped} style={{
            padding: "10px 16px", fontSize: 12, fontWeight: 500, borderRadius: 6,
            background: r.circuit_breaker.tripped ? "var(--short)" : "var(--bg-elev)",
            color: r.circuit_breaker.tripped ? "#fff" : "var(--fg-muted)",
            border: "1px solid var(--border-default)",
            cursor: r.circuit_breaker.tripped ? "pointer" : "not-allowed",
          }}>重置熔断器</button>
        </div>
      </Card>

      {/* 4 key meters */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12 }}>
        <Card pad={16}><RiskMeter label="当日盈亏" value={r.daily_loss_pct} limit={r.daily_loss_budget}/></Card>
        <Card pad={16}><RiskMeter label="当前回撤" value={r.drawdown_pct} limit={r.drawdown_limit} inverted/></Card>
        <Card pad={16}><RiskMeter label="总敞口" value={r.total_exposure_pct} limit={r.exposure_limit}/></Card>
        <Card pad={16}><RiskMeter label="95% CVaR" value={r.cvar_95} limit={r.cvar_limit}/></Card>
      </div>

      {/* Correlation + cooldowns + recent blocks */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Card title="相关性分组" subtitle="同组最多 2 仓位">
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {r.correlation_groups.map(g => (
              <div key={g.name} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <div style={{ width: 110, fontSize: 12, fontWeight: 500 }}>{g.name}</div>
                <div style={{ flex: 1, display: "flex", gap: 2 }}>
                  {Array.from({ length: g.max }).map((_, i) => (
                    <div key={i} style={{
                      flex: 1, height: 20, borderRadius: 3,
                      background: i < g.open ? "var(--amber-500)" : "var(--bg-inset)",
                      border: "1px solid var(--border-subtle)",
                    }}/>
                  ))}
                </div>
                <div className="mono" style={{ fontSize: 11, color: "var(--fg-tertiary)", width: 40, textAlign: "right" }}>{g.open}/{g.max}</div>
                <div style={{ fontSize: 10, color: "var(--fg-tertiary)", width: 80 }}>{g.pairs[0] || "—"}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="冷却 · 频率限制">
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div>
              <div style={{ fontSize: 11, color: "var(--fg-tertiary)", marginBottom: 6 }}>交易频率</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div style={{ padding: 10, background: "var(--bg-inset)", borderRadius: 6 }}>
                  <div style={{ fontSize: 10, color: "var(--fg-tertiary)", marginBottom: 2 }}>本小时</div>
                  <div className="mono" style={{ fontSize: 16 }}>{r.rate_limits.hour.used}<span style={{ color: "var(--fg-tertiary)", fontSize: 11 }}>/{r.rate_limits.hour.max}</span></div>
                </div>
                <div style={{ padding: 10, background: "var(--bg-inset)", borderRadius: 6 }}>
                  <div style={{ fontSize: 10, color: "var(--fg-tertiary)", marginBottom: 2 }}>今日</div>
                  <div className="mono" style={{ fontSize: 16 }}>{r.rate_limits.day.used}<span style={{ color: "var(--fg-tertiary)", fontSize: 11 }}>/{r.rate_limits.day.max}</span></div>
                </div>
              </div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "var(--fg-tertiary)", marginBottom: 6 }}>交易对冷却</div>
              {r.cooldowns.map((c, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 0", borderTop: i > 0 ? "1px solid var(--border-subtle)" : "none" }}>
                  <span className="mono" style={{ fontSize: 12, width: 90 }}>{c.pair}</span>
                  {c.kind === "free" ?
                    <StatusPill tone="success">可交易</StatusPill> :
                    <><StatusPill tone="warning">冷却中</StatusPill><span className="mono" style={{ fontSize: 11, color: "var(--warning)" }}>{c.until}</span></>
                  }
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>

      {/* Recent risk blocks */}
      <Card title="最近风控拦截" subtitle="风控门拒绝的决策">
        <div style={{ display: "flex", flexDirection: "column" }}>
          {r.recent_blocks.map((b, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "10px 0", borderTop: i > 0 ? "1px solid var(--border-subtle)" : "none" }}>
              <span className="mono" style={{ fontSize: 11, color: "var(--fg-tertiary)", width: 64 }}>{b.t}</span>
              <span className="mono" style={{ fontSize: 11, color: "var(--fg-tertiary)", width: 80 }}>{b.id}</span>
              <StatusPill tone="danger">{b.rule}</StatusPill>
              <span style={{ fontSize: 12, color: "var(--fg-secondary)", flex: 1 }}>{b.detail}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

window.BacktestPage = BacktestPage;
window.RiskPage = RiskPage;
