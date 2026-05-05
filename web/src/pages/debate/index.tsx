import { Check, Filter, MessageSquare, Zap } from 'lucide-react';
import { useMemo } from 'react';
import { useNavigate, useParams } from 'react-router';

import { EmptyState } from '@/components/ui/empty-state';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { Skeleton } from '@/components/ui/skeleton';
import { useDecisionDetail } from '@/hooks/use-decision-detail';
import { useDecisions } from '@/hooks/use-decisions';
import type { DebateTurn, DecisionDetail } from '@/types/api';

import { DirChip } from '@/components/ui/dir-chip';

import { AgentBadge } from './components/agent-badge';
import { DebateTurnCard } from './components/debate-turn';
import { DivergenceMeter } from './components/divergence-meter';
import { AGENTS, type AgentKind, type DebateScenario, scoreToDirection } from './constants';

const STEP_COLORS = {
  violet: { fg: 'oklch(62% 0.180 295)', bg: 'oklch(62% 0.180 295 / 0.18)' },
  warning: { fg: 'oklch(78% 0.155 75)', bg: 'oklch(78% 0.155 75 / 0.18)' },
  amber: { fg: 'oklch(74% 0.165 70)', bg: 'oklch(74% 0.165 70 / 0.18)' },
} as const;

const KNOWN_AGENT_KINDS = new Set<string>(['tech', 'chain', 'news', 'macro', 'verdict', 'other']);

const normalizeKind = (raw: string): AgentKind => {
  const low = raw.toLowerCase();
  if (KNOWN_AGENT_KINDS.has(low)) return low as AgentKind;
  if (/(tech|indicator)/.test(low)) return 'tech';
  if (/(chain|onchain|whale|funding)/.test(low)) return 'chain';
  if (/(news|sentiment|social)/.test(low)) return 'news';
  if (/(macro|fed|dxy|etf|fng)/.test(low)) return 'macro';
  return 'other';
};

type NormalizedDir = 'bullish' | 'bearish' | 'neutral';

const normalizeDir = (raw: string): NormalizedDir => {
  const low = raw.toLowerCase();
  if (low === 'bullish' || low === 'long' || low === 'buy') return 'bullish';
  if (low === 'bearish' || low === 'short' || low === 'sell') return 'bearish';
  return 'neutral';
};

const toScenario = (d: DecisionDetail): DebateScenario => {
  const turnsApi = d.debate_turns ?? [];
  const groupedInitial = new Map<AgentKind, { dir: NormalizedDir; conf: number }>();

  // Initial positions = before-state of each agent's first-round turn; fall
  // back to agent_analyses when no turns exist (skipped debate).
  for (const t of turnsApi) {
    if (t.round !== 1) continue;
    const kind = normalizeKind(t.from);
    if (!groupedInitial.has(kind)) {
      groupedInitial.set(kind, {
        dir: normalizeDir(t.before_direction),
        conf: t.before_confidence,
      });
    }
  }
  if (groupedInitial.size === 0) {
    for (const a of d.agent_analyses) {
      const kind = normalizeKind(a.name);
      // Uses shared AGENT_SCORE_DIRECTION_THRESHOLD (±0.3) so this matches the
      // detail grid's scoreToDirection — FE-I11 fixes a label-drift bug where the
      // same score rendered as "bullish" here but "neutral" in agent-analysis-grid.
      const dir: NormalizedDir = scoreToDirection(a.score);
      groupedInitial.set(kind, { dir, conf: a.confidence });
    }
  }

  const roundsMap = new Map<number, DebateTurn[]>();
  for (const t of turnsApi) {
    const arr = roundsMap.get(t.round) ?? [];
    arr.push(t);
    roundsMap.set(t.round, arr);
  }
  const rounds = [...roundsMap.entries()]
    .sort(([a], [b]) => a - b)
    .map(([n, turns]) => ({
      n,
      turns: turns.map((t) => ({
        from: normalizeKind(t.from),
        to: t.to ? normalizeKind(t.to) : null,
        critique: t.reasoning || t.new_findings || '（无论点文本）',
        dir: normalizeDir(t.after_direction),
        conf: t.after_confidence,
        move: t.move,
      })),
    }));

  const gate = d.debate_gate;
  const cm = d.consensus_metrics;
  const before = cm?.dispersion ?? gate?.dispersion ?? 0;
  // FE-I12: default afterDispersion to ``before`` rather than 0. Previously a single-
  // turn final round (e.g. one agent errored out) skipped recomputation and displayed
  // a false perfect convergence (0). Now the UI shows "no change" when the final
  // round has too few turns to compute a real dispersion.
  const finalTurns = rounds.length > 0 ? (rounds[rounds.length - 1]?.turns ?? []) : [];
  let afterDispersion = before;
  if (finalTurns.length >= 2) {
    const scores = finalTurns.map((t) =>
      t.dir === 'bullish' ? t.conf : t.dir === 'bearish' ? -t.conf : 0,
    );
    const mean = scores.reduce((s, v) => s + v, 0) / scores.length;
    const variance = scores.reduce((s, v) => s + (v - mean) ** 2, 0) / scores.length;
    afterDispersion = Math.sqrt(variance);
  }

  const initial = [...groupedInitial.entries()].map(([kind, v]) => ({
    kind,
    dir: v.dir,
    conf: v.conf,
  }));

  const actionLow = d.verdict.action.toLowerCase();
  const finalDir: NormalizedDir =
    actionLow === 'long' || actionLow === 'buy'
      ? 'bullish'
      : actionLow === 'short' || actionLow === 'sell'
        ? 'bearish'
        : 'neutral';

  return {
    id: d.commit_hash,
    pair: d.pair,
    price: d.price,
    gate: {
      decision: gate?.decision === 'debate' ? 'debate' : 'skipped',
      reason: gate?.reason ?? d.debate_skip_reason ?? '',
    },
    initial,
    rounds,
    convergence: {
      before: Number(before.toFixed(3)),
      after: Number(afterDispersion.toFixed(3)),
      target: cm?.confusion_threshold ?? 0.5,
    },
    final_verdict: {
      action: finalDir,
      confidence: d.verdict.confidence,
      scale: d.verdict.size,
      thesis: d.verdict.reasoning || '（无裁决文本）',
    },
  };
};

const DebateEmpty = ({ message }: { message: string }) => (
  <EmptyState icon={<MessageSquare className="h-6 w-6" />} title={message} />
);

const DebateContent = () => {
  const navigate = useNavigate();
  const { commitId } = useParams<{ commitId?: string }>();
  const decisions = useDecisions({ page: 1, size: 20 });

  // Prefer URL commitId; else fall back to most-recent decision that had a debate.
  const targetHash = useMemo(() => {
    if (commitId) return commitId;
    const items = decisions.data?.items ?? [];
    const withDebate = items.find((i) => i.debate_status && !i.debate_status.startsWith('skipped'));
    return withDebate?.commit_hash ?? items[0]?.commit_hash;
  }, [commitId, decisions.data]);

  const detail = useDecisionDetail(targetHash);

  // FE-I7: memoize the scenario normalisation so toScenario does not re-run on
  // every ancestor re-render (React Query polling, URL param changes, etc.).
  // Kept unconditional (hook ordering) by returning null when data is missing.
  const d = useMemo(
    () => (detail.data ? toScenario(detail.data) : null),
    [detail.data],
  );

  if (decisions.isLoading || detail.isLoading) {
    return <Skeleton className="h-96 w-full" />;
  }

  if (!targetHash) {
    return <DebateEmpty message="暂无决策记录，新决策将自动出现在此" />;
  }

  if (detail.isError || !detail.data || d === null) {
    return <DebateEmpty message={`无法加载决策 ${targetHash} 的辩论详情`} />;
  }
  const hasDebate = d.rounds.length > 0;

  const steps = [
    {
      label: '门控',
      sub: hasDebate ? '触发辩论' : '跳过',
      Icon: Filter,
      tone: 'violet' as const,
    },
    { label: '第 1 轮', sub: '4 Agent 交叉挑战', Icon: MessageSquare, tone: 'violet' as const },
    { label: '第 2 轮', sub: '强化 / 让步 / 保持', Icon: MessageSquare, tone: 'violet' as const },
    {
      label: '收敛',
      sub: `${d.convergence.before.toFixed(2)} → ${d.convergence.after.toFixed(2)}`,
      Icon: Check,
      tone: 'warning' as const,
    },
    {
      label: '裁决',
      sub: `${d.final_verdict.action === 'bullish' ? '看多' : d.final_verdict.action === 'bearish' ? '看空' : '中性'} ${(d.final_verdict.scale * 100).toFixed(0)}%`,
      Icon: Zap,
      tone: 'amber' as const,
    },
  ];

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        onBack={() => void navigate(-1)}
        eyebrow={`辩论可视化 · 决策 ${d.id.slice(0, 10)}`}
        title={hasDebate ? `${d.rounds.length} 轮交叉挑战辩论` : '无辩论（门控跳过）'}
        subtitle={
          <>
            <span className="font-mono">{d.pair}</span> @{' '}
            <span className="font-mono">${d.price.toLocaleString()}</span>
            {hasDebate ? <> · 初始分歧度 {d.convergence.before.toFixed(2)} 触发辩论</> : null}
          </>
        }
        actions={
          <DivergenceMeter
            before={d.convergence.before}
            after={d.convergence.after}
            target={d.convergence.target}
          />
        }
      />

      <div className="flex items-stretch mt-2">
        {steps.map((step, i) => {
          const c = STEP_COLORS[step.tone];
          return (
            <div key={i} className="flex-1 flex flex-col items-center relative">
              {i > 0 ? (
                <div
                  className="absolute top-[18px] right-1/2 w-full h-px"
                  style={{ background: c.fg }}
                />
              ) : null}
              <div
                className="w-9 h-9 rounded-full flex items-center justify-center relative z-10 mb-1.5 border-[1.5px]"
                style={{ background: c.bg, borderColor: c.fg, color: c.fg }}
              >
                <step.Icon size={14} strokeWidth={2} />
              </div>
              <div className="text-xs font-medium">{step.label}</div>
              <div className="text-[10px] text-muted-foreground mt-0.5">{step.sub}</div>
            </div>
          );
        })}
      </div>

      <div
        className="rounded-xl border p-4 flex gap-3.5 items-center"
        style={{
          background: 'oklch(62% 0.180 295 / 0.06)',
          borderColor: 'oklch(62% 0.180 295 / 0.30)',
        }}
      >
        <AgentBadge kind={d.initial[0]?.kind ?? 'chain'} size={40} />
        <div className="flex-1">
          <div
            className="text-[11px] uppercase tracking-wider font-medium mb-0.5"
            style={{ color: 'oklch(62% 0.180 295)' }}
          >
            debate_gate 裁定
          </div>
          <div className="text-sm font-medium">
            {hasDebate ? '触发辩论' : '跳过辩论'} · {d.gate.reason || '（无说明）'}
          </div>
        </div>
        <div className="flex gap-1.5 flex-wrap">
          {d.initial.map((a) => {
            const agent = AGENTS[a.kind];
            const dirColor =
              a.dir === 'bullish'
                ? 'oklch(74% 0.160 150)'
                : a.dir === 'bearish'
                  ? 'oklch(66% 0.210 25)'
                  : 'hsl(var(--muted-foreground))';
            return (
              <div
                key={a.kind}
                className="px-2.5 py-1.5 rounded-md border text-center min-w-[72px]"
                style={{ background: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
              >
                <div className="text-[10px] font-medium" style={{ color: agent.color }}>
                  {agent.zh}
                </div>
                <div className="font-mono text-[11px] mt-0.5" style={{ color: dirColor }}>
                  {a.dir === 'bullish' ? '↑' : a.dir === 'bearish' ? '↓' : '—'} {a.conf.toFixed(2)}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {hasDebate ? (
        d.rounds.map((round) => (
          <div key={round.n}>
            <div className="flex items-center gap-3 mb-3">
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center font-semibold text-sm border"
                style={{
                  background: 'oklch(62% 0.180 295 / 0.15)',
                  borderColor: 'oklch(62% 0.180 295)',
                  color: 'oklch(72% 0.140 300)',
                }}
              >
                {round.n}
              </div>
              <div>
                <div className="text-[15px] font-semibold tracking-tight">
                  第 {round.n} 轮 · 交叉挑战
                </div>
                <div className="text-[11px] text-muted-foreground">
                  {round.n === 1
                    ? '每个 Agent 必须指出他人最弱论点'
                    : '必须捍卫立场或说明被什么数据改变（反趋同规则）'}
                </div>
              </div>
            </div>
            <div className="flex flex-col gap-2.5 pl-10 ml-[15px] border-l border-dashed border-border">
              {round.turns.map((turn, i) => (
                <DebateTurnCard key={i} turn={turn} />
              ))}
            </div>
          </div>
        ))
      ) : (
        <DebateEmpty
          message={
            d.gate.reason
              ? `debate_gate 跳过辩论 — ${d.gate.reason}`
              : '此决策未触发辩论'
          }
        />
      )}

      <div
        className="rounded-xl border p-5 flex gap-4 items-start"
        style={{
          background: 'linear-gradient(135deg, oklch(74% 0.165 70 / 0.10), hsl(var(--card)) 60%)',
          borderColor: 'oklch(74% 0.165 70 / 0.35)',
          boxShadow: '0 0 32px oklch(74% 0.165 70 / 0.25)',
        }}
      >
        <AgentBadge kind="verdict" size={48} />
        <div className="flex-1">
          <div
            className="text-[11px] uppercase tracking-wider font-medium mb-1"
            style={{ color: 'oklch(74% 0.165 70)' }}
          >
            AI 首席决策者 · {hasDebate ? '辩论后裁决' : '直接裁决'}
          </div>
          <div className="flex items-center gap-2.5 mb-2.5">
            <DirChip dir={d.final_verdict.action} confidence={d.final_verdict.confidence} />
            <span className="text-[11px] text-muted-foreground">
              仓位 {(d.final_verdict.scale * 100).toFixed(0)}%
            </span>
          </div>
          <div className="text-sm leading-relaxed font-medium">{d.final_verdict.thesis}</div>
        </div>
      </div>
    </div>
  );
};

// FE-m8: wrap with page-scoped PageBoundary so a runtime failure in
// DebateContent stays localised to the content area and doesn't collapse the
// AppShell-level sidebar. DebateContent renders its own loading/empty states
// because it depends on a chained query (decisions list -> detail).
const DebatePage = () => (
  <PageBoundary>
    <DebateContent />
  </PageBoundary>
);

export default DebatePage;
