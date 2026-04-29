// Debate visualization — conversation-style with divergence meter
// Shows the full 2-round debate scenario for decision 9f2c8e1

function DivergenceMeter({ before, after, target }) {
  // horizontal meter showing divergence decreasing round-by-round
  const width = 280;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--fg-tertiary)", fontWeight: 500 }}>
        <span>分歧度</span><span>收敛目标 {target.toFixed(2)}</span>
      </div>
      <div style={{ position: "relative", height: 8, background: "var(--bg-inset)", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ position: "absolute", left: `${target * 100}%`, top: 0, bottom: 0, width: 1, background: "var(--amber-500)", opacity: 0.6 }}/>
        <div style={{
          position: "absolute", left: 0, top: 0, bottom: 0, width: `${after * 100}%`,
          background: "linear-gradient(90deg, var(--long) 0%, var(--warning) 100%)",
        }}/>
        <div style={{
          position: "absolute", left: `${after * 100}%`, top: -3, bottom: -3, width: 2,
          background: "var(--fg-primary)", boxShadow: "0 0 8px rgba(255,255,255,0.3)",
        }}/>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
        <span className="mono" style={{ color: "var(--fg-tertiary)" }}>开始 {before.toFixed(2)}</span>
        <span className="mono" style={{ color: "var(--warning)", fontWeight: 500 }}>→ 结束 {after.toFixed(2)}</span>
      </div>
    </div>
  );
}

function DebateTurn({ turn, agentKind }) {
  const from = AGENTS[turn.from];
  const to = turn.to ? AGENTS[turn.to] : null;
  const moveTone = turn.move.includes("让步") ? "warning" :
                   turn.move === "保持" ? "default" : "violet";
  return (
    <div style={{
      display: "flex", gap: 12, padding: 14,
      borderRadius: 8, background: "var(--bg-card)",
      border: "1px solid var(--border-subtle)",
      position: "relative",
    }}>
      <AgentBadge kind={turn.from} size={32}/>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6, flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: from.color }}>{from.zh}</span>
          {to && (
            <>
              <Icon d={icons.arrowRight} size={11} stroke={1.8} style={{ color: "var(--fg-tertiary)" }}/>
              <span style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>回应 <span style={{ color: to.color }}>{to.zh}</span></span>
            </>
          )}
          {!to && <span style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>独白</span>}
          <div style={{ flex: 1 }}/>
          <DirChip dir={turn.dir} confidence={turn.conf}/>
        </div>
        <div style={{
          fontSize: 13, color: "var(--fg-primary)", lineHeight: 1.55, textWrap: "pretty",
          padding: "8px 12px", background: "var(--bg-inset)",
          borderLeft: `2px solid ${from.color}`,
          borderRadius: "0 6px 6px 0",
          fontStyle: "italic",
        }}>
          「{turn.critique}」
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8, fontSize: 11 }}>
          <StatusPill tone={moveTone}>
            {turn.move.includes("强化") && <Icon d={icons.arrowUp} size={9} stroke={2.5}/>}
            {turn.move.includes("让步") && <Icon d={icons.arrowDown} size={9} stroke={2.5}/>}
            {turn.move === "保持" && <Icon d={icons.pause} size={9} stroke={2.5}/>}
            {turn.move}
          </StatusPill>
        </div>
      </div>
    </div>
  );
}

function DebatePage({ onBack }) {
  const d = MOCK.debate_scenario;
  const [activeRound, setActiveRound] = useState(0); // 0 = gate, 1 = round1, 2 = round2, 3 = verdict

  return (
    <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: 16 }}>
        <button onClick={onBack} style={{ padding: 6, color: "var(--fg-tertiary)" }}>
          <Icon d={icons.chevLeft} size={18}/>
        </button>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--violet-400)", fontWeight: 500, marginBottom: 4 }}>
            辩论可视化 · 决策 {d.id}
          </div>
          <h1 style={{ fontSize: 22 }}>4 位 Agent · 2 轮交叉挑战辩论</h1>
          <div style={{ fontSize: 13, color: "var(--fg-secondary)", marginTop: 6 }}>
            <span className="mono">{d.pair}</span> @ <span className="mono">${d.price.toLocaleString()}</span> · 初始分歧度 0.73 触发辩论
          </div>
        </div>
        <DivergenceMeter before={d.convergence.before} after={d.convergence.after} target={d.convergence.target}/>
      </div>

      {/* Timeline stepper */}
      <div style={{ display: "flex", alignItems: "stretch", gap: 0, marginTop: 8 }}>
        {[
          { label: "门控", sub: d.gate.decision === "debate" ? "触发辩论" : "跳过", icon: icons.filter, tone: "violet", active: activeRound >= 0 },
          { label: "第 1 轮", sub: "4 Agent 交叉挑战", icon: icons.chat, tone: "violet", active: activeRound >= 1 },
          { label: "第 2 轮", sub: "强化 / 让步 / 保持", icon: icons.chat, tone: "violet", active: activeRound >= 2 },
          { label: "收敛", sub: "0.73 → 0.41", icon: icons.check, tone: "warning", active: activeRound >= 3 },
          { label: "裁决", sub: "做空 30%", icon: icons.bolt, tone: "amber", active: activeRound >= 3 },
        ].map((step, i, arr) => (
          <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", position: "relative" }}>
            {i > 0 && (
              <div style={{
                position: "absolute", top: 18, right: "50%", width: "100%", height: 1,
                background: step.active ? "var(--violet-400)" : "var(--border-subtle)",
              }}/>
            )}
            <button onClick={() => setActiveRound(i)} style={{
              width: 36, height: 36, borderRadius: "50%",
              background: step.active ? `color-mix(in oklch, var(--${step.tone}-500) 20%, transparent)` : "var(--bg-elev)",
              border: `1.5px solid ${step.active ? `var(--${step.tone}-500)` : "var(--border-default)"}`,
              color: step.active ? `var(--${step.tone}-500)` : "var(--fg-tertiary)",
              display: "flex", alignItems: "center", justifyContent: "center",
              position: "relative", zIndex: 1, marginBottom: 6,
            }}>
              <Icon d={step.icon} size={14} stroke={2}/>
            </button>
            <div style={{ fontSize: 12, fontWeight: 500, color: step.active ? "var(--fg-primary)" : "var(--fg-tertiary)" }}>{step.label}</div>
            <div style={{ fontSize: 10, color: "var(--fg-tertiary)", marginTop: 2 }}>{step.sub}</div>
          </div>
        ))}
      </div>

      {/* Gate explanation */}
      <Card pad={0} style={{
        background: "color-mix(in oklch, var(--violet-500) 6%, transparent)",
        border: "1px solid color-mix(in oklch, var(--violet-500) 30%, transparent)",
      }}>
        <div style={{ padding: 16, display: "flex", gap: 14, alignItems: "center" }}>
          <AgentBadge kind="chain" size={40}/>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--violet-400)", fontWeight: 500, marginBottom: 2 }}>
              debate_gate 裁定
            </div>
            <div style={{ fontSize: 14, fontWeight: 500 }}>触发辩论 · {d.gate.reason}</div>
          </div>
          {/* Initial positions */}
          <div style={{ display: "flex", gap: 6 }}>
            {d.initial.map(a => {
              const agent = AGENTS[a.kind];
              const dirTone = a.dir === "bullish" ? "var(--long)" : a.dir === "bearish" ? "var(--short)" : "var(--fg-tertiary)";
              return (
                <div key={a.kind} style={{
                  padding: "6px 10px", borderRadius: 6,
                  background: "var(--bg-elev)", border: "1px solid var(--border-subtle)",
                  textAlign: "center", minWidth: 72,
                }}>
                  <div style={{ fontSize: 10, color: agent.color, fontWeight: 500 }}>{agent.zh}</div>
                  <div className="mono" style={{ fontSize: 11, color: dirTone, marginTop: 2 }}>
                    {a.dir === "bullish" ? "↑" : a.dir === "bearish" ? "↓" : "—"} {a.conf.toFixed(2)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </Card>

      {/* Rounds */}
      {d.rounds.map((round) => (
        <div key={round.n}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
            <div style={{
              width: 32, height: 32, borderRadius: "50%",
              background: "color-mix(in oklch, var(--violet-500) 15%, transparent)",
              border: "1px solid var(--violet-500)", color: "var(--violet-400)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontFamily: "var(--font-display)", fontWeight: 600, fontSize: 14,
            }}>{round.n}</div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, fontFamily: "var(--font-display)" }}>第 {round.n === 1 ? "一" : "二"} 轮 · 交叉挑战</div>
              <div style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>
                {round.n === 1 ? "每个 Agent 必须指出他人最弱论点" : "必须捍卫立场或说明被什么数据改变(反趋同规则)"}
              </div>
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10, paddingLeft: 40, borderLeft: "1px dashed var(--border-default)", marginLeft: 15 }}>
            {round.turns.map((turn, i) => <DebateTurn key={i} turn={turn}/>)}
          </div>
        </div>
      ))}

      {/* Final verdict */}
      <Card pad={0} style={{
        background: "linear-gradient(135deg, color-mix(in oklch, var(--amber-500) 10%, transparent), var(--bg-card) 60%)",
        border: "1px solid color-mix(in oklch, var(--amber-500) 35%, transparent)",
        boxShadow: "var(--shadow-glow-amber)",
      }}>
        <div style={{ padding: 20, display: "flex", gap: 16, alignItems: "flex-start" }}>
          <AgentBadge kind="verdict" size={48}/>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--amber-500)", fontWeight: 500, marginBottom: 4 }}>
              AI 首席决策者 · 辩论后裁决
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
              <DirChip dir={d.final_verdict.action} confidence={d.final_verdict.confidence}/>
              <span style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>仓位 {(d.final_verdict.scale * 100).toFixed(0)}%</span>
            </div>
            <div style={{ fontSize: 14, color: "var(--fg-primary)", lineHeight: 1.6, textWrap: "pretty", fontWeight: 500 }}>
              {d.final_verdict.thesis}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

window.DebatePage = DebatePage;
