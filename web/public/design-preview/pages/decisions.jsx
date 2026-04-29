// Decisions page — list + 8-section detail panel

function DecisionsList({ onSelect, selectedId }) {
  const [filter, setFilter] = useState("all");
  const filters = [
    { id: "all", label: "全部", count: MOCK.decisions.length },
    { id: "executed", label: "已执行", count: MOCK.decisions.filter(d => d.status === "executed").length },
    { id: "rejected", label: "被拒", count: MOCK.decisions.filter(d => d.status === "rejected").length },
    { id: "hold", label: "观望", count: MOCK.decisions.filter(d => d.action === "hold").length },
  ];
  const list = filter === "all" ? MOCK.decisions :
               filter === "hold" ? MOCK.decisions.filter(d => d.action === "hold") :
               MOCK.decisions.filter(d => d.status === filter);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
      {/* Filter bar */}
      <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--border-subtle)", display: "flex", gap: 16, alignItems: "center" }}>
        <div style={{ display: "flex", gap: 4 }}>
          {filters.map(f => (
            <button key={f.id} onClick={() => setFilter(f.id)} style={{
              padding: "4px 10px", fontSize: 12, fontWeight: 500, borderRadius: 4,
              background: filter === f.id ? "var(--bg-hover)" : "transparent",
              color: filter === f.id ? "var(--fg-primary)" : "var(--fg-tertiary)",
            }}>
              {f.label} <span style={{ opacity: 0.6, marginLeft: 4 }}>{f.count}</span>
            </button>
          ))}
        </div>
        <div style={{ flex: 1 }}/>
        <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
          <Icon d={icons.search} size={12} stroke={1.8}/>
          <input placeholder="搜索 commit hash / 交易对..." style={{
            marginLeft: 8, background: "transparent", border: "none", outline: "none",
            color: "var(--fg-primary)", fontSize: 12, width: 220, fontFamily: "var(--font-mono)",
          }}/>
        </div>
      </div>

      {/* Table */}
      <div style={{ flex: 1, overflow: "auto" }}>
        <div style={{
          display: "grid", gridTemplateColumns: "110px 100px 90px 72px 90px 1fr 90px 32px",
          gap: 12, padding: "8px 16px", fontSize: 10, textTransform: "uppercase",
          letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 500,
          position: "sticky", top: 0, background: "var(--bg-base)",
          borderBottom: "1px solid var(--border-subtle)", zIndex: 1,
        }}>
          <div>时间</div><div>Commit</div><div>交易对</div><div>动作</div>
          <div>置信度</div><div>辩论 · 论点</div>
          <div style={{ textAlign: "right" }}>状态</div><div/>
        </div>
        {list.map(d => (
          <button key={d.id} onClick={() => onSelect && onSelect(d.id)} style={{
            display: "grid", gridTemplateColumns: "110px 100px 90px 72px 90px 1fr 90px 32px",
            gap: 12, padding: "12px 16px", alignItems: "center", width: "100%", textAlign: "left",
            borderBottom: "1px solid var(--border-subtle)",
            background: selectedId === d.id ? "var(--bg-hover)" : "transparent",
            borderLeft: selectedId === d.id ? "2px solid var(--amber-500)" : "2px solid transparent",
            paddingLeft: selectedId === d.id ? 14 : 16,
          }}>
            <div className="mono" style={{ fontSize: 11, color: "var(--fg-secondary)" }}>{d.t.slice(5)}</div>
            <div className="mono" style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>{d.id}</div>
            <div className="mono" style={{ fontSize: 12, fontWeight: 500 }}>{d.pair}</div>
            <DirChip dir={d.action}/>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 40, height: 4, background: "var(--bg-inset)", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ width: `${d.conf * 100}%`, height: "100%", background: d.conf > 0.6 ? "var(--amber-500)" : "var(--fg-tertiary)" }}/>
              </div>
              <span className="mono" style={{ fontSize: 11, color: "var(--fg-secondary)" }}>{(d.conf * 100).toFixed(0)}%</span>
            </div>
            <div style={{ fontSize: 12, color: "var(--fg-secondary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              <span style={{ color: d.debate.includes("skipped") ? "var(--fg-tertiary)" : "var(--violet-500)", fontSize: 11, marginRight: 8 }}>
                {d.debate}
              </span>
              {d.reject && <span style={{ color: "var(--short)" }}>{d.reject}</span>}
              {!d.reject && d.status === "executed" && `执行 @ $${d.price.toLocaleString()}`}
              {!d.reject && d.status === "approved" && d.action === "hold" && "无持仓变更"}
            </div>
            <div style={{ textAlign: "right" }}>
              {d.status === "executed" && d.pnl != null && (
                <span className="mono" style={{ fontSize: 12, color: d.pnl >= 0 ? "var(--long)" : "var(--short)", fontWeight: 500 }}>
                  {d.pnl >= 0 ? "+" : ""}${Math.abs(d.pnl).toFixed(0)}
                </span>
              )}
              {d.status === "executed" && d.pnl == null && <StatusPill tone="default">持仓</StatusPill>}
              {d.status === "rejected" && <StatusPill tone="danger">拒</StatusPill>}
              {d.status === "approved" && d.pnl == null && <StatusPill tone="default">—</StatusPill>}
            </div>
            <Icon d={icons.chevRight} size={14}/>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── 8-section Decision Detail ─────────────────────────────────
function DecisionDetail({ onBack, onOpenDebate }) {
  const d = MOCK.decision_detail;
  const sections = [
    { id: "summary", label: "1 · 摘要" },
    { id: "context", label: "2 · 市场上下文" },
    { id: "agents",  label: "3 · 四方分析" },
    { id: "debate",  label: "4 · 辩论" },
    { id: "verdict", label: "5 · AI 裁决" },
    { id: "risk",    label: "6 · 风控审计" },
    { id: "exec",    label: "7 · 执行" },
    { id: "meta",    label: "8 · 元数据" },
  ];
  const [active, setActive] = useState("summary");

  return (
    <div style={{ display: "flex", height: "100%", minHeight: 0 }}>
      {/* Left anchor nav */}
      <div style={{ width: 180, flexShrink: 0, borderRight: "1px solid var(--border-subtle)", padding: 16, display: "flex", flexDirection: "column", gap: 2 }}>
        <button onClick={onBack} style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--fg-tertiary)", fontSize: 11, marginBottom: 12 }}>
          <Icon d={icons.chevLeft} size={12}/> 返回列表
        </button>
        <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 500, marginBottom: 4 }}>
          决策详情
        </div>
        {sections.map(s => (
          <button key={s.id} onClick={() => setActive(s.id)} style={{
            textAlign: "left", padding: "6px 10px", borderRadius: 4, fontSize: 12,
            background: active === s.id ? "var(--bg-hover)" : "transparent",
            color: active === s.id ? "var(--fg-primary)" : "var(--fg-secondary)",
            fontWeight: active === s.id ? 500 : 400,
          }}>{s.label}</button>
        ))}
      </div>

      {/* Scrollable detail pane */}
      <div style={{ flex: 1, overflow: "auto", padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>
        {/* Header */}
        <div style={{ display: "flex", alignItems: "flex-start", gap: 16, paddingBottom: 16, borderBottom: "1px solid var(--border-subtle)" }}>
          <div style={{
            width: 48, height: 48, borderRadius: 12,
            background: "linear-gradient(135deg, var(--amber-500), var(--amber-600))",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "var(--shadow-glow-amber)", color: "var(--fg-inverse)",
          }}>
            <Icon d={icons.bolt} size={24} stroke={2}/>
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <DirChip dir={d.verdict.action} confidence={d.verdict.confidence}/>
              <span className="mono" style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>{d.commit}</span>
              <span style={{ color: "var(--border-default)" }}>·</span>
              <span style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>{d.t}</span>
            </div>
            <h1 style={{ fontSize: 20, marginBottom: 6 }}>
              <span className="mono">{d.pair}</span> · 做多 <span className="mono" style={{ color: "var(--amber-500)" }}>{(d.verdict.position_scale * 100).toFixed(0)}%</span> 仓位
            </h1>
            <div style={{ fontSize: 13, color: "var(--fg-secondary)", textWrap: "pretty" }}>{d.verdict.thesis}</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
            <StatusPill tone="success"><Icon d={icons.check} size={10} stroke={2.5}/>已执行</StatusPill>
            <div className="mono" style={{ fontSize: 11, color: "var(--long)", fontWeight: 500 }}>+$1,831.20 · +4.89%</div>
          </div>
        </div>

        {/* Section 2 — Context */}
        <div>
          <SectionTitle eyebrow="2 · 市场上下文">决策时快照</SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
            <Card pad={14}><KPI label="价格" value={`$${d.price.toLocaleString()}`}/></Card>
            <Card pad={14}><KPI label="持仓前" value="空仓" sub="可用 $42,180"/></Card>
            <Card pad={14}><KPI label="当日敞口" value="42%" sub={`上限 50%`}/></Card>
            <Card pad={14}><KPI label="熔断器" value="正常" sub="日亏损 -0.8%"/></Card>
          </div>
        </div>

        {/* Section 3 — Agents */}
        <div>
          <SectionTitle eyebrow="3 · 四方分析" right={<span style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>4 Agent 并行 · {d.latency.agents}ms</span>}>四位专业分析师</SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {d.agents.map(a => <AgentReportCard key={a.kind} report={a}/>)}
          </div>
        </div>

        {/* Section 4 — Debate */}
        <div>
          <SectionTitle eyebrow="4 · 辩论"
            right={<button onClick={onOpenDebate} style={{ fontSize: 11, color: "var(--amber-500)", display: "flex", alignItems: "center", gap: 4 }}>
              查看辩论场景示例 <Icon d={icons.external} size={11}/>
            </button>}>
            debate_gate: <span style={{ color: "var(--long)" }}>跳过</span>
          </SectionTitle>
          <div style={{
            padding: 16, background: "var(--bg-inset)", border: "1px solid var(--border-subtle)",
            borderRadius: 8, display: "flex", alignItems: "center", gap: 14,
          }}>
            <div style={{
              width: 36, height: 36, borderRadius: 8,
              background: "color-mix(in oklch, var(--long) 15%, transparent)",
              border: "1px solid color-mix(in oklch, var(--long) 35%, transparent)",
              color: "var(--long)", display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <Icon d={icons.check} size={18} stroke={2.2}/>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 500 }}>强共识 — 辩论被门控跳过</div>
              <div style={{ fontSize: 12, color: "var(--fg-tertiary)", marginTop: 2 }}>
                4/4 Agent 方向一致(看多)· 均值置信度 0.70 · 分歧度 0.18
              </div>
            </div>
            <div className="mono" style={{ fontSize: 11, color: "var(--fg-tertiary)", textAlign: "right" }}>
              <div>节省 <span style={{ color: "var(--amber-500)" }}>9 次 LLM 调用</span></div>
              <div>约 $0.14</div>
            </div>
          </div>
        </div>

        {/* Section 5 — Verdict */}
        <div>
          <SectionTitle eyebrow="5 · AI 裁决" right={<AgentBadge kind="verdict" size={22} showName/>}>首席决策者 · Verdict</SectionTitle>
          <Card pad={0} style={{
            background: "linear-gradient(135deg, color-mix(in oklch, var(--amber-500) 6%, transparent) 0%, var(--bg-card) 60%)",
            border: "1px solid color-mix(in oklch, var(--amber-500) 30%, transparent)",
          }}>
            <div style={{ padding: 20 }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20, paddingBottom: 16, borderBottom: "1px solid var(--border-subtle)" }}>
                <KPI label="动作" value={<DirChip dir={d.verdict.action}/>} mono={false}/>
                <KPI label="置信度" value={`${(d.verdict.confidence * 100).toFixed(0)}%`}/>
                <KPI label="仓位缩放" value={`${(d.verdict.position_scale * 100).toFixed(0)}%`} sub="size = max(floor, scale × ceiling)"/>
              </div>
              <div style={{ paddingTop: 16, display: "flex", flexDirection: "column", gap: 12 }}>
                <div>
                  <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 500, marginBottom: 6 }}>Thesis · 交易论点</div>
                  <div style={{ fontSize: 14, fontWeight: 500, color: "var(--fg-primary)", textWrap: "pretty" }}>{d.verdict.thesis}</div>
                </div>
                <div>
                  <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 500, marginBottom: 6 }}>推理</div>
                  <div style={{ fontSize: 13, color: "var(--fg-secondary)", lineHeight: 1.6, textWrap: "pretty" }}>{d.verdict.reasoning}</div>
                </div>
                <div style={{ display: "flex", gap: 12, padding: 12, background: "var(--bg-inset)", borderRadius: 6, alignItems: "flex-start" }}>
                  <Icon d={icons.alert} size={14} stroke={1.8} style={{ color: "var(--warning)", flexShrink: 0, marginTop: 2 }}/>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 500, color: "var(--warning)", marginBottom: 2 }}>失效条件 · Invalidation</div>
                    <div style={{ fontSize: 12, color: "var(--fg-secondary)" }}>{d.verdict.invalidation}</div>
                  </div>
                </div>

                {/* Bias + experience */}
                <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 12, marginTop: 4 }}>
                  <div style={{ padding: 12, background: "color-mix(in oklch, var(--violet-500) 8%, transparent)", border: "1px solid color-mix(in oklch, var(--violet-500) 25%, transparent)", borderRadius: 6 }}>
                    <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--violet-400)", fontWeight: 500, marginBottom: 4 }}>偏差校正</div>
                    <div style={{ fontSize: 12, color: "var(--fg-secondary)" }}>{d.bias.detected}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 500, marginBottom: 6 }}>注入的历史经验 · {d.experience.length} 条</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      {d.experience.map((ex, i) => (
                        <div key={i} style={{ display: "flex", gap: 10, fontSize: 11, color: "var(--fg-secondary)", padding: "6px 10px", background: "var(--bg-inset)", borderRadius: 4 }}>
                          <span className="mono" style={{ color: "var(--fg-tertiary)", width: 84 }}>{ex.date}</span>
                          <span style={{ flex: 1 }}>{ex.cond} → <span style={{ color: "var(--fg-primary)" }}>{ex.verdict}</span></span>
                          <span className="mono" style={{ color: ex.pnl.startsWith("+") ? "var(--long)" : "var(--short)" }}>{ex.pnl}</span>
                          <span style={{ color: "var(--fg-tertiary)", fontStyle: "italic" }}>{ex.lesson}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Section 6 — Risk audit */}
        <div>
          <SectionTitle eyebrow="6 · 风控审计" right={<StatusPill tone="success">11/11 通过</StatusPill>}>11 项硬规则</SectionTitle>
          <Card>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 20px" }}>
              {d.risk_audit.map(r => (
                <div key={r.n} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: "1px solid var(--border-subtle)" }}>
                  <div className="mono" style={{ fontSize: 10, color: "var(--fg-tertiary)", width: 20 }}>{r.n.toString().padStart(2, "0")}</div>
                  <div style={{
                    width: 18, height: 18, borderRadius: "50%",
                    background: r.status === "pass" ? "color-mix(in oklch, var(--long) 18%, transparent)" :
                                r.status === "warn" ? "color-mix(in oklch, var(--warning) 18%, transparent)" :
                                                       "color-mix(in oklch, var(--short) 18%, transparent)",
                    color: r.status === "pass" ? "var(--long)" : r.status === "warn" ? "var(--warning)" : "var(--short)",
                    display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                  }}>
                    <Icon d={r.status === "pass" ? icons.check : icons.alert} size={10} stroke={2.5}/>
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, fontWeight: 500 }}>{r.name}</div>
                    <div style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>{r.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Section 7 — Execution */}
        <div>
          <SectionTitle eyebrow="7 · 执行">订单填充</SectionTitle>
          <Card>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 16 }}>
              <KPI label="订单 ID" value={<span className="mono" style={{ fontSize: 11 }}>{d.execution.order_id}</span>} mono={false}/>
              <KPI label="方向 · 数量" value={`${d.execution.side.toUpperCase()} ${d.execution.qty} BTC`}/>
              <KPI label="成交价" value={`$${d.execution.filled_price.toLocaleString()}`}/>
              <KPI label="手续费" value={`$${d.execution.fees}`}/>
              <KPI label="滑点" value={`${d.execution.slippage_bps} bps`}/>
              <KPI label="延迟" value={`${d.execution.latency_ms}ms`} sub={d.execution.exchange}/>
            </div>
          </Card>
        </div>

        {/* Section 8 — Meta */}
        <div>
          <SectionTitle eyebrow="8 · 元数据">延迟 · Token · 成本</SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Card title="延迟分解" subtitle={`总 ${d.latency.total}ms`}>
              <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 4 }}>
                {Object.entries(d.latency).filter(([k]) => k !== "total").map(([stage, ms]) => (
                  <div key={stage} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ width: 64, fontSize: 11, color: "var(--fg-tertiary)", textTransform: "capitalize" }}>{stage}</div>
                    <div style={{ flex: 1, height: 6, background: "var(--bg-inset)", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{ width: `${(ms / d.latency.total) * 100}%`, height: "100%", background: stage === "verdict" ? "var(--amber-500)" : stage === "agents" ? "var(--cyan-500)" : "var(--violet-500)" }}/>
                    </div>
                    <div className="mono" style={{ width: 60, textAlign: "right", fontSize: 11 }}>{ms}ms</div>
                  </div>
                ))}
              </div>
            </Card>
            <Card title="Token · 成本" subtitle={`LLM 调用 5 次 · 缓存命中 2 次`}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginTop: 4 }}>
                <KPI label="输入" value={d.tokens.input.toLocaleString()}/>
                <KPI label="输出" value={d.tokens.output.toLocaleString()}/>
                <KPI label="成本" value={`$${d.tokens.cost_usd}`}/>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── One agent's report card ────────────────────────────────────
function AgentReportCard({ report }) {
  const a = AGENTS[report.kind];
  return (
    <Card pad={0} style={{ overflow: "hidden" }}>
      <div style={{
        padding: 14, display: "flex", alignItems: "center", gap: 10,
        background: `linear-gradient(90deg, color-mix(in oklch, ${a.color} 10%, transparent) 0%, transparent 80%)`,
        borderBottom: "1px solid var(--border-subtle)",
      }}>
        <AgentBadge kind={report.kind} size={28}/>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: a.color }}>{a.zh} · {a.en}</div>
          <div style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>{a.role}</div>
        </div>
        <DirChip dir={report.direction} confidence={report.confidence}/>
      </div>
      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ fontSize: 12, color: "var(--fg-secondary)", lineHeight: 1.55, textWrap: "pretty" }}>{report.reasoning}</div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {report.factors.map((f, i) => (
            <span key={i} className="ct-chip" style={{ color: a.color, borderColor: `color-mix(in oklch, ${a.color} 30%, transparent)`, background: `color-mix(in oklch, ${a.color} 10%, transparent)` }}>
              <Icon d={icons.check} size={9} stroke={2.5}/>{f}
            </span>
          ))}
          {report.risks.map((f, i) => (
            <span key={i} className="ct-chip">
              <Icon d={icons.alert} size={9} stroke={2}/>{f}
            </span>
          ))}
        </div>
        <div style={{
          padding: "8px 10px", background: "var(--bg-inset)", borderRadius: 4,
          display: "flex", gap: 12, fontSize: 10, color: "var(--fg-tertiary)", flexWrap: "wrap",
          fontFamily: "var(--font-mono)",
        }}>
          {Object.entries(report.data).slice(0, 4).map(([k, v]) => (
            <span key={k}><span style={{ color: "var(--fg-muted)" }}>{k}</span> <span style={{ color: "var(--fg-secondary)" }}>{typeof v === "number" ? (v > 1000 ? v.toLocaleString() : v) : v}</span></span>
          ))}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--fg-tertiary)" }}>
          <span>数据充足度: <span style={{ color: report.sufficiency === "high" ? "var(--long)" : "var(--warning)" }}>{report.sufficiency === "high" ? "高" : "中"}</span></span>
          <span>置信度 {(report.confidence * 100).toFixed(0)}%</span>
        </div>
      </div>
    </Card>
  );
}

window.DecisionsList = DecisionsList;
window.DecisionDetail = DecisionDetail;
window.AgentReportCard = AgentReportCard;
