import { Check, Filter, MessageSquare, Zap } from 'lucide-react';
import { useMemo } from 'react';
import { useNavigate, useParams } from 'react-router';

import { EmptyState } from '@/components/ui/empty-state';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/cn';
import { useDecisionDetail } from '@/hooks/use-decision-detail';
import { useDecisions } from '@/hooks/use-decisions';
import type { DebateTurn, DecisionDetail } from '@/types/api';

import { DirChip } from '@/components/ui/dir-chip';

import { AgentBadge } from './components/agent-badge';
import { DebateTurnCard } from './components/debate-turn';
import { DivergenceMeter } from './components/divergence-meter';
import { AGENTS, type AgentKind, type DebateScenario, scoreToDirection } from './constants';

// Debate visual cleanup (FE-2026-05-06): collapse the violet/warning/amber
// inline-style palette down to Tailwind utility classes. The whole page leans
// to the brand amber accent (was over-using violet, which made the Debate
// page feel like a different product from the rest of the app).
const STEP_TONES = {
  flow: 'border-cyan-500 bg-cyan-500/15 text-cyan-500',
  pivot: 'border-amber-500 bg-amber-500/15 text-amber-500',
  final: 'border-amber-500 bg-amber-500/20 text-amber-500',
} as const;

type StepTone = keyof typeof STEP_TONES;

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

  const steps: { label: string; sub: string; Icon: typeof Filter; tone: StepTone }[] = [
    { label: '门控', sub: hasDebate ? '触发辩论' : '跳过', Icon: Filter, tone: 'flow' },
    { label: '第 1 轮', sub: '4 Agent 交叉挑战', Icon: MessageSquare, tone: 'flow' },
    { label: '第 2 轮', sub: '强化 / 让步 / 保持', Icon: MessageSquare, tone: 'flow' },
    {
      label: '收敛',
      sub: `${d.convergence.before.toFixed(2)} → ${d.convergence.after.toFixed(2)}`,
      Icon: Check,
      tone: 'pivot',
    },
    {
      label: '裁决',
      sub: `${d.final_verdict.action === 'bullish' ? '看多' : d.final_verdict.action === 'bearish' ? '看空' : '中性'} ${(d.final_verdict.scale * 100).toFixed(0)}%`,
      Icon: Zap,
      tone: 'final',
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

      <div className="mt-2 flex items-stretch">
        {steps.map((step, i) => (
          <div key={i} className="relative flex flex-1 flex-col items-center">
            {i > 0 ? (
              <div className="absolute right-1/2 top-[18px] h-px w-full bg-border" />
            ) : null}
            <div
              className={cn(
                'relative z-10 mb-1.5 flex h-9 w-9 items-center justify-center rounded-full border-[1.5px]',
                STEP_TONES[step.tone],
              )}
            >
              <step.Icon size={14} strokeWidth={2} />
            </div>
            <div className="text-xs font-medium">{step.label}</div>
            <div className="mt-0.5 text-[10px] text-muted-foreground">{step.sub}</div>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-3.5 rounded-xl border border-cyan-500/30 bg-cyan-500/5 p-4">
        <AgentBadge kind={d.initial[0]?.kind ?? 'chain'} size={40} />
        <div className="flex-1">
          <div className="mb-0.5 text-[11px] font-medium uppercase tracking-wider text-cyan-500">
            debate_gate 裁定
          </div>
          <div className="text-sm font-medium">
            {hasDebate ? '触发辩论' : '跳过辩论'} · {d.gate.reason || '（无说明）'}
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {d.initial.map((a) => {
            const agent = AGENTS[a.kind];
            const dirClass =
              a.dir === 'bullish'
                ? 'text-trade-long'
                : a.dir === 'bearish'
                  ? 'text-trade-short'
                  : 'text-muted-foreground';
            return (
              <div
                key={a.kind}
                className="min-w-[72px] rounded-md border border-border bg-card px-2.5 py-1.5 text-center"
              >
                <div className="text-[10px] font-medium" style={{ color: agent.color }}>
                  {agent.zh}
                </div>
                <div className={cn('mt-0.5 font-mono text-[11px]', dirClass)}>
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
            <div className="mb-3 flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full border border-amber-500 bg-amber-500/15 text-sm font-semibold text-amber-500">
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
            <div className="ml-[15px] flex flex-col gap-2.5 border-l border-dashed border-border pl-10">
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

      <div className="flex items-start gap-4 rounded-xl border border-amber-500/35 bg-gradient-to-br from-amber-500/10 to-card p-5 shadow-glow-amber">
        <AgentBadge kind="verdict" size={48} />
        <div className="flex-1">
          <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-amber-500">
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
