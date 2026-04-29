import { ArrowDown, ArrowRight, ArrowUp, Pause } from 'lucide-react';

import { AGENTS, type DebateTurn as Turn } from '../constants';

import { AgentBadge } from './agent-badge';
import { DirChip } from '@/components/ui/dir-chip';

const MOVE_TONE = (move: string) => {
  if (move.includes('让步')) return { color: 'oklch(78% 0.155 75)', bg: 'oklch(78% 0.155 75 / 0.15)' };
  if (move === '保持') return { color: 'hsl(var(--muted-foreground))', bg: 'oklch(65% 0.015 270 / 0.15)' };
  return { color: 'oklch(62% 0.180 295)', bg: 'oklch(62% 0.180 295 / 0.15)' };
};

interface Props {
  turn: Turn;
}

export const DebateTurnCard = ({ turn }: Props) => {
  const from = AGENTS[turn.from];
  const to = turn.to ? AGENTS[turn.to] : null;
  const tone = MOVE_TONE(turn.move);

  return (
    <article
      tabIndex={0}
      aria-label={`${from.zh} → ${to ? AGENTS[turn.to!].zh : '独白'}: ${turn.move}`}
      className="flex gap-3 p-4 rounded-lg bg-card border border-border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/40"
    >
      <AgentBadge kind={turn.from} size={32} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
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
          className="text-[13px] leading-relaxed px-3 py-2 italic"
          style={{
            background: 'hsl(var(--muted))',
            borderLeft: `2px solid ${from.color}`,
            borderRadius: '0 6px 6px 0',
            color: 'hsl(var(--foreground))',
          }}
        >
          「{turn.critique}」
        </div>
        <div className="mt-2 flex items-center gap-2 text-[11px]">
          <span
            className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium border"
            style={{ color: tone.color, background: tone.bg, borderColor: tone.color }}
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
