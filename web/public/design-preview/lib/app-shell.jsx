// App shell: left sidebar + top bar. Provides navigation context for all pages.

const { useState: useStateShell, useEffect: useEffectShell } = React;

function AppShell({ page, onNav, children, topBarExtra }) {
  const nav = [
    { id: "dashboard", label: "仪表盘",  icon: icons.dashboard },
    { id: "decisions", label: "决策",    icon: icons.decision  },
    { id: "debate",    label: "辩论",    icon: icons.split     },
    { id: "backtest",  label: "回测",    icon: icons.refresh   },
    { id: "risk",      label: "风控",    icon: icons.shield    },
    { id: "metrics",   label: "指标",    icon: icons.gauge     },
    { id: "chat",      label: "会话",    icon: icons.chat      },
    { id: "market",    label: "行情",    icon: icons.market    },
  ];
  return (
    <div className="ct-root" style={{ width: "100%", height: "100%", display: "flex", background: "var(--bg-base)", overflow: "hidden" }}>
      {/* Sidebar */}
      <aside style={{
        width: 208, flexShrink: 0, background: "var(--bg-raised)",
        borderRight: "1px solid var(--border-subtle)", display: "flex", flexDirection: "column",
      }}>
        <div style={{ padding: "18px 16px 14px 16px", display: "flex", alignItems: "center", gap: 10, borderBottom: "1px solid var(--border-subtle)" }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8,
            background: "linear-gradient(135deg, var(--amber-500) 0%, var(--amber-600) 100%)",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "var(--shadow-glow-amber)",
            color: "var(--fg-inverse)", fontWeight: 700, fontSize: 14, fontFamily: "var(--font-display)",
          }}>₵</div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: "-0.01em" }}>CryptoTrader</div>
            <div style={{ fontSize: 10, color: "var(--fg-tertiary)", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 500 }}>AI · v2.4</div>
          </div>
        </div>

        <nav style={{ flex: 1, padding: 8, display: "flex", flexDirection: "column", gap: 1, overflow: "auto" }}>
          {nav.map((n) => {
            const active = n.id === page;
            return (
              <button key={n.id} onClick={() => onNav && onNav(n.id)} style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "7px 10px", borderRadius: 6, textAlign: "left",
                background: active ? "var(--bg-hover)" : "transparent",
                color: active ? "var(--fg-primary)" : "var(--fg-secondary)",
                fontSize: 13, fontWeight: active ? 500 : 400,
                borderLeft: active ? "2px solid var(--amber-500)" : "2px solid transparent",
                paddingLeft: active ? 8 : 10, transition: "background 0.12s",
              }}>
                <Icon d={n.icon} size={15} stroke={1.6}/>
                <span>{n.label}</span>
                {n.id === "decisions" && <span style={{ marginLeft: "auto", fontSize: 10, color: "var(--fg-tertiary)" }}>12</span>}
                {n.id === "risk" && <span className="ct-pulse" style={{ marginLeft: "auto", width: 6, height: 6, borderRadius: "50%", background: "var(--long)" }}/>}
              </button>
            );
          })}
        </nav>

        <div style={{ padding: 12, borderTop: "1px solid var(--border-subtle)", display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{
            padding: "8px 10px", borderRadius: 8,
            background: "var(--bg-elev)", border: "1px solid var(--border-subtle)",
            display: "flex", alignItems: "center", gap: 8,
          }}>
            <span className="ct-pulse" style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--long)", flexShrink: 0 }}/>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 11, fontWeight: 500, color: "var(--fg-primary)" }}>模拟交易 · 运行中</div>
              <div className="mono" style={{ fontSize: 10, color: "var(--fg-tertiary)" }}>下次分析 02:47</div>
            </div>
          </div>
          <button style={{
            display: "flex", alignItems: "center", gap: 8, padding: "6px 10px",
            borderRadius: 6, fontSize: 12, color: "var(--fg-tertiary)",
          }}>
            <Icon d={icons.settings} size={14}/>设置
          </button>
        </div>
      </aside>

      {/* Main */}
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <header style={{
          height: 48, flexShrink: 0,
          borderBottom: "1px solid var(--border-subtle)",
          background: "var(--bg-base)",
          display: "flex", alignItems: "center", gap: 12, padding: "0 20px",
        }}>
          <div style={{ fontSize: 11, color: "var(--fg-tertiary)", fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}>
            <span>CryptoTrader</span>
            <Icon d={icons.chevRight} size={10} stroke={2}/>
            <span style={{ color: "var(--fg-secondary)" }}>{
              { dashboard: "仪表盘", decisions: "决策", debate: "辩论可视化", backtest: "回测", risk: "风控", metrics: "指标", chat: "多 Agent 会话", market: "行情" }[page] || page
            }</span>
          </div>
          <div style={{ flex: 1 }}/>
          {topBarExtra}
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div className="mono" style={{ fontSize: 11, color: "var(--fg-tertiary)" }}>BTC ${MOCK.market.price.toLocaleString()}</div>
            <div style={{ width: 1, height: 14, background: "var(--border-default)" }}/>
            <StatusPill tone="long" live>在线</StatusPill>
          </div>
        </header>
        <main style={{ flex: 1, overflow: "auto", background: "var(--bg-base)" }}>
          {children}
        </main>
      </div>
    </div>
  );
}

window.AppShell = AppShell;
