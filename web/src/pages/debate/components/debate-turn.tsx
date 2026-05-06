import { ArrowDown, ArrowRight, ArrowUp, Pause } from 'lucide-react';

import { AGENTS, type DebateTurn as Turn } from '../constants';

import { AgentBadge } from './agent-badge';
import { DirChip } from '@/components/ui/dir-chip';

// FE-2026-05-06 visual cleanup: replaced inline OKLCH literals with utility
// classes. ``强化`` (strengthen) is the active state -> amber accent; ``让步``
// (concede) is a softer warning -> amber-200; ``保持`` is neutral.
const MOVE_TONE_CLASS = (move: string): string => {
  if (move.includes('强化'))
    return 'border-amber-500/60 bg-amber-500/15 text-amber-500';
  if (move.includes('让步'))
    return 'border-amber-200/60 bg-amber-200/15 text-amber-200';
  return 'border-border bg-muted text-muted-foreground';
};

interface Props {
  turn: Turn;
}

export const DebateTurnCard = ({ turn }: Props) => {
  const from = AGENTS[turn.from];
  const to = turn.to ? AGENTS[turn.to] : null;
  const toneClass = MOVE_TONE_CLASS(turn.move);

  return (
    <article
      tabIndex={0}
      aria-label={`${from.zh} → ${to ? AGENTS[turn.to!].zh : '独白'}: ${turn.move}`}
      className="flex gap-3 rounded-lg border border-border bg-card p-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/40"
    >
      <AgentBadge kind={turn.from} size={32} />
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex flex-wrap items-center gap-2">
          {/* AGENTS[*].color is an OKLCH string by design (see lib/agents.ts) so
              SVG / canvas consumers get the same value across themes. Here we
              accept the inline style for parity with AgentBadge. */}
          <span className="text-xs font-semibold" style={{ color: from.color }}>
            {from.zh}
          </span>
          {to ? (
            <>
              <ArrowRight size={11} strokeWidth={1.8} className="text-muted-foreground" />
              <span className="text-[11px] text-muted-foreground">
                回应 <span style={{ color: to.color }}>{to.zh}</span>
              </span>
            </>
          ) : (
            <span className="text-[11px] text-muted-foreground">独白</span>
          )}
          <span className="flex-1" />
          <DirChip dir={turn.dir} confidence={turn.conf} />
        </div>
        <div
          className="rounded-r-md bg-muted px-3 py-2 text-[13px] italic leading-relaxed text-foreground"
          style={{ borderLeft: `2px solid ${from.color}` }}
        >
          「{turn.critique}」
        </div>
        <div className="mt-2 flex items-center gap-2 text-[11px]">
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-medium ${toneClass}`}
          >
            {turn.move.includes('强化') ? <ArrowUp size={10} strokeWidth={2.4} /> : null}
            {turn.move.includes('让步') ? <ArrowDown size={10} strokeWidth={2.4} /> : null}
            {turn.move === '保持' ? <Pause size={10} strokeWidth={2.4} /> : null}
            {turn.move}
          </span>
        </div>
      </div>
    </article>
  );
};
