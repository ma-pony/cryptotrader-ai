// Shared atoms for the CryptoTrader AI mocks
// All components here are stateless; consumers provide data.

const { useState, useEffect, useMemo, useRef } = React;

// ── Icons (hand-rolled, 16px default) ──────────────────────────
const Icon = ({ d, size = 16, stroke = 1.6, fill = "none" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke="currentColor"
       strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
    {typeof d === "string" ? <path d={d} /> : d}
  </svg>
);
const icons = {
  dashboard: "M3 3h7v9H3zM14 3h7v5h-7zM14 12h7v9h-7zM3 16h7v5H3z",
  decision:  "M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83",
  chart:     "M3 3v18h18M7 14l3-3 4 4 5-6",
  shield:    "M12 2l9 4v6c0 5-3.5 9-9 10-5.5-1-9-5-9-10V6l9-4z",
  gauge:     "M12 14l4-4M3 12a9 9 0 0 1 18 0M5 18h14",
  chat:      "M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z",
  market:    "M3 12l4-6 4 4 4-8 6 10",
  play:      "M5 3l14 9-14 9z",
  pause:     "M6 4h4v16H6zM14 4h4v16h-4z",
  check:     "M5 12l5 5L20 6",
  x:         "M6 6l12 12M6 18L18 6",
  alert:     "M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z",
  arrowUp:   "M12 19V5M5 12l7-7 7 7",
  arrowDown: "M12 5v14M5 12l7 7 7-7",
  arrowRight:"M5 12h14M12 5l7 7-7 7",
  search:    "M11 17a6 6 0 1 0 0-12 6 6 0 0 0 0 12zM21 21l-5.2-5.2",
  bot:       "M12 2v2M9 6h6a3 3 0 0 1 3 3v7a3 3 0 0 1-3 3H9a3 3 0 0 1-3-3V9a3 3 0 0 1 3-3zM9 12h.01M15 12h.01M10 16h4",
  spark:     "M12 2l2.4 6.6L21 11l-6.6 2.4L12 20l-2.4-6.6L3 11l6.6-2.4z",
  clock:     "M12 6v6l4 2M12 22a10 10 0 1 1 0-20 10 10 0 0 1 0 20z",
  circle:    "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z",
  dot:       <circle cx="12" cy="12" r="4" fill="currentColor" stroke="none"/>,
  grip:      "M9 6h.01M15 6h.01M9 12h.01M15 12h.01M9 18h.01M15 18h.01",
  chevRight: "M9 6l6 6-6 6",
  chevDown:  "M6 9l6 6 6-6",
  chevLeft:  "M15 6l-6 6 6 6",
  pulse:     "M3 12h4l3-9 4 18 3-9h4",
  network:   "M6 3v4M18 17v4M18 3v4M6 17v4M3 6h4M17 6h4M3 18h4M17 18h4M8 8l8 8M16 8l-8 8",
  flame:     "M12 2s4 4 4 8a4 4 0 0 1-8 0c0-2 1-3 1-4s-1-1-1-3 4-1 4-1z",
  scale:     "M6 9l-3 4h6zM18 9l-3 4h6zM3 9h18M12 3v18",
  zap:       "M13 2L4 14h7l-1 8 9-12h-7z",
  eye:       "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
  book:      "M4 4v16a2 2 0 0 0 2 2h14V4a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2zM8 6h8M8 10h8M8 14h5",
  settings:  "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z",
  filter:    "M22 3H2l8 9.46V19l4 2v-8.54L22 3z",
  refresh:   "M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15",
  external:  "M15 3h6v6M10 14L21 3M21 10v11H3V3h11",
  copy:      "M8 8h12v12H8zM4 4h12v4M4 8v12h4",
  telegram:  "M22 2L2 11l6 3 2 7 4-4 6 5 2-20z",
  hitl:      "M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM23 11h-6M20 8v6",
  seed:      "M12 2C12 8 8 12 4 12c0 6 4 10 8 10s8-4 8-10c-4 0-8-4-8-10z",
  split:     "M6 3v12M18 9v12M6 15a6 6 0 0 0 12 0",
  bolt:      "M13 2L3 14h7v8l10-12h-7z",
  link:      "M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 1 0-7.07-7.07l-1.72 1.71M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",
  plus:      "M12 5v14M5 12h14",
};

// ── Sparkline ──────────────────────────────────────────────────
function Sparkline({ data, width = 80, height = 24, color, fill, strokeWidth = 1.5 }) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data), max = Math.max(...data);
  const range = max - min || 1;
  const stepX = width / (data.length - 1);
  const pts = data.map((v, i) => [i * stepX, height - ((v - min) / range) * height]);
  const d = "M " + pts.map((p) => p.map((n) => n.toFixed(2)).join(",")).join(" L ");
  const area = d + ` L ${width},${height} L 0,${height} Z`;
  const last = pts[pts.length - 1];
  return (
    <svg width={width} height={height} style={{ overflow: "visible", display: "block" }}>
      {fill && <path d={area} fill={fill} />}
      <path d={d} fill="none" stroke={color || "currentColor"} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      {last && <circle cx={last[0]} cy={last[1]} r="2" fill={color || "currentColor"} />}
    </svg>
  );
}

// ── KPI ────────────────────────────────────────────────────────
function KPI({ label, value, delta, sub, mono = true, large = false, trend, trendColor }) {
  const deltaColor = delta == null ? null : delta >= 0 ? "var(--long)" : "var(--short)";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
      <div style={{ fontSize: "var(--t-micro)", color: "var(--fg-tertiary)", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 500 }}>{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
        <div className={mono ? "mono" : ""} style={{
          fontSize: large ? 28 : 20,
          fontWeight: 600,
          fontFamily: large ? "var(--font-display)" : undefined,
          letterSpacing: "-0.02em",
          color: "var(--fg-primary)",
          lineHeight: 1.1,
        }}>
          {value}
        </div>
        {delta != null && (
          <div className="mono" style={{ fontSize: 12, fontWeight: 500, color: deltaColor }}>
            {delta >= 0 ? "+" : ""}{typeof delta === "number" ? delta.toFixed(2) : delta}%
          </div>
        )}
        {trend && (
          <div style={{ marginLeft: "auto", color: trendColor || "var(--fg-tertiary)" }}>
            <Sparkline data={trend} width={60} height={18} />
          </div>
        )}
      </div>
      {sub && <div style={{ fontSize: "var(--t-micro)", color: "var(--fg-tertiary)" }}>{sub}</div>}
    </div>
  );
}

// ── Card wrappers ──────────────────────────────────────────────
function Card({ children, title, subtitle, right, style, pad = 16, className = "" }) {
  return (
    <div className={`ct-card ${className}`} style={{ display: "flex", flexDirection: "column", ...style }}>
      {(title || right) && (
        <div style={{
          display: "flex", alignItems: "center", gap: 12,
          padding: `${pad - 4}px ${pad}px`,
          borderBottom: subtitle || children ? "1px solid var(--border-subtle)" : "none",
        }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            {title && <div style={{ fontSize: 13, fontWeight: 600, letterSpacing: "-0.01em" }}>{title}</div>}
            {subtitle && <div style={{ fontSize: 11, color: "var(--fg-tertiary)", marginTop: 2 }}>{subtitle}</div>}
          </div>
          {right}
        </div>
      )}
      {children && <div style={{ padding: pad, flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>{children}</div>}
    </div>
  );
}

// ── Status pill ────────────────────────────────────────────────
function StatusPill({ tone = "default", live = false, children, style }) {
  const map = {
    default: { color: "var(--fg-secondary)", bg: "var(--bg-elev)", border: "var(--border-subtle)" },
    long:    { color: "var(--long)",  bg: "var(--long-soft)",  border: "color-mix(in oklch, var(--long) 30%, transparent)" },
    short:   { color: "var(--short)", bg: "var(--short-soft)", border: "color-mix(in oklch, var(--short) 30%, transparent)" },
    hold:    { color: "var(--fg-secondary)", bg: "var(--hold-soft)", border: "var(--border-default)" },
    amber:   { color: "var(--amber-500)",  bg: "color-mix(in oklch, var(--amber-500) 15%, transparent)",  border: "color-mix(in oklch, var(--amber-500) 35%, transparent)" },
    violet:  { color: "var(--violet-500)", bg: "color-mix(in oklch, var(--violet-500) 15%, transparent)", border: "color-mix(in oklch, var(--violet-500) 35%, transparent)" },
    cyan:    { color: "var(--cyan-500)",   bg: "color-mix(in oklch, var(--cyan-500) 15%, transparent)",   border: "color-mix(in oklch, var(--cyan-500) 35%, transparent)" },
    danger:  { color: "var(--danger)", bg: "var(--short-soft)", border: "color-mix(in oklch, var(--danger) 35%, transparent)" },
    success: { color: "var(--success)", bg: "var(--long-soft)", border: "color-mix(in oklch, var(--success) 35%, transparent)" },
  };
  const c = map[tone] || map.default;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "2px 8px", borderRadius: 999,
      fontSize: 11, fontWeight: 500, lineHeight: 1.6,
      color: c.color, background: c.bg, border: `1px solid ${c.border}`,
      whiteSpace: "nowrap", fontVariantCaps: "all-small-caps", letterSpacing: "0.04em",
      ...style,
    }}>
      {live && (
        <span className="ct-pulse" style={{
          width: 6, height: 6, borderRadius: "50%", background: c.color, display: "inline-block",
        }}/>
      )}
      {children}
    </span>
  );
}

// ── Agent palette helper ───────────────────────────────────────
const AGENTS = {
  tech:    { zh: "技术面",   en: "Tech",    color: "var(--agent-tech)",    icon: icons.chart,   role: "指标 · RSI/MACD/SMA/BB" },
  chain:   { zh: "链上",     en: "Chain",   color: "var(--agent-chain)",   icon: icons.network, role: "OI · 资金费率 · 鲸鱼" },
  news:    { zh: "新闻",     en: "News",    color: "var(--agent-news)",    icon: icons.flame,   role: "RSS · 情绪 · 社交" },
  macro:   { zh: "宏观",     en: "Macro",   color: "var(--agent-macro)",   icon: icons.scale,   role: "Fed · DXY · FnG · ETF" },
  verdict: { zh: "AI 决策者", en: "Verdict", color: "var(--agent-verdict)", icon: icons.bolt,    role: "首席裁决者" },
};

// ── AgentBadge ─────────────────────────────────────────────────
function AgentBadge({ kind, size = 24, showName = false, dim = false }) {
  const a = AGENTS[kind] || AGENTS.tech;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 8, opacity: dim ? 0.55 : 1 }}>
      <span style={{
        width: size, height: size, borderRadius: 6,
        background: `color-mix(in oklch, ${a.color} 18%, transparent)`,
        border: `1px solid color-mix(in oklch, ${a.color} 40%, transparent)`,
        color: a.color,
        display: "inline-flex", alignItems: "center", justifyContent: "center",
      }}>
        <Icon d={a.icon} size={Math.round(size * 0.55)} stroke={1.8}/>
      </span>
      {showName && <span style={{ fontSize: 12, fontWeight: 500, color: a.color }}>{a.zh}</span>}
    </span>
  );
}

// ── Directional chip (long/short/hold) ─────────────────────────
function DirChip({ dir, confidence }) {
  const map = {
    long:    { tone: "long",  label: "看多", icon: icons.arrowUp },
    bullish: { tone: "long",  label: "看多", icon: icons.arrowUp },
    short:   { tone: "short", label: "看空", icon: icons.arrowDown },
    bearish: { tone: "short", label: "看空", icon: icons.arrowDown },
    hold:    { tone: "hold",  label: "观望", icon: icons.pause },
    neutral: { tone: "hold",  label: "中性", icon: icons.pause },
    close:   { tone: "amber", label: "平仓", icon: icons.x },
  };
  const m = map[dir] || map.hold;
  return (
    <StatusPill tone={m.tone}>
      <Icon d={m.icon} size={10} stroke={2.2}/>
      {m.label}
      {confidence != null && <span className="mono" style={{ opacity: 0.75, marginLeft: 4 }}>{(confidence * 100).toFixed(0)}%</span>}
    </StatusPill>
  );
}

// ── Section header ─────────────────────────────────────────────
function SectionTitle({ children, right, eyebrow }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 12, marginBottom: 12 }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        {eyebrow && <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--fg-tertiary)", fontWeight: 500 }}>{eyebrow}</div>}
        <div style={{ fontSize: 15, fontWeight: 600, fontFamily: "var(--font-display)", letterSpacing: "-0.01em" }}>{children}</div>
      </div>
      {right}
    </div>
  );
}

// ── Progress bar ───────────────────────────────────────────────
function Bar({ value, max = 1, color = "var(--amber-500)", bg = "var(--bg-inset)", height = 4, radius = 2 }) {
  const pct = Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div style={{ width: "100%", height, background: bg, borderRadius: radius, overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: radius, transition: "width 0.3s" }}/>
    </div>
  );
}

// Export to window for cross-script access
Object.assign(window, {
  Icon, icons, Sparkline, KPI, Card, StatusPill, AgentBadge, AGENTS, DirChip, SectionTitle, Bar,
});
