import { ArrowDown, ArrowRight, ArrowUp, Pause } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { AGENTS, type DebateTurn as Turn } from '../constants';

import { AgentBadge } from './agent-badge';
import { DirChip } from '@/components/ui/dir-chip';

// FE-2026-05-06 visual cleanup: replaced inline OKLCH literals with utility
// classes. ``强化`` (strengthen) is the active state -> amber accent; ``让步``
// (concede) is a softer warning -> amber-200; ``保持`` is neutral.
const MOVE_TONE_CLASS = (move: string): string => {
  if (move === '强化' || move === 'strengthen')
    return 'border-amber-500/60 bg-amber-500/15 text-amber-500';
  if (move === '让步' || move === 'concede')
    return 'border-amber-200/60 bg-amber-200/15 text-amber-200';
  return 'border-border bg-muted text-muted-foreground';
};

interface Props {
  turn: Turn;
}

export const DebateTurnCard = ({ turn }: Props) => {
  const { t } = useTranslation('debate');
  const from = AGENTS[turn.from];
  const to = turn.to ? AGENTS[turn.to] : null;
  const toneClass = MOVE_TONE_CLASS(turn.move);
  const moveKey =
    turn.move === '强化' || turn.move === 'strengthen'
      ? 'strengthen'
      : turn.move === '让步' || turn.move === 'concede'
        ? 'concede'
        : turn.move === '保持' || turn.move === 'hold'
          ? 'hold'
          : null;
  const moveLabel = moveKey ? t(`moves.${moveKey}`) : t('moves.unknown', { move: turn.move });

  return (
    <article
      tabIndex={0}
      aria-label={t('turn.aria', { from: t(`agent.${turn.from}`), to: to ? t(`agent.${turn.to!}`) : t('turn.monologue'), move: moveLabel })}
      className="flex gap-3 rounded-lg border border-border bg-card p-4 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/40"
    >
      <AgentBadge kind={turn.from} size={32} />
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex flex-wrap items-center gap-2">
          {/* AGENTS[*].color is an OKLCH string by design (see lib/agents.ts) so
              SVG / canvas consumers get the same value across themes. Here we
              accept the inline style for parity with AgentBadge. */}
          <span className="text-xs font-semibold" style={{ color: from.color }}>
            {t(`agent.${turn.from}`)}
          </span>
          {to ? (
            <>
              <ArrowRight size={11} strokeWidth={1.8} className="text-muted-foreground" />
              <span className="text-[11px] text-muted-foreground">
                {t('turn.respondsTo')} <span style={{ color: to.color }}>{t(`agent.${turn.to!}`)}</span>
              </span>
            </>
          ) : (
            <span className="text-[11px] text-muted-foreground">{t('turn.monologue')}</span>
          )}
          <span className="flex-1" />
          <DirChip dir={turn.dir} confidence={turn.conf} />
        </div>
        <div
          className="rounded-r-md bg-muted px-3 py-2 text-[13px] italic leading-relaxed text-foreground"
          style={{ borderLeft: `2px solid ${from.color}` }}
        >
          「{turn.critique || t('noArgument')}」
        </div>
        <div className="mt-2 flex items-center gap-2 text-[11px]">
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-medium ${toneClass}`}
          >
            {moveKey === 'strengthen' ? <ArrowUp size={10} strokeWidth={2.4} /> : null}
            {moveKey === 'concede' ? <ArrowDown size={10} strokeWidth={2.4} /> : null}
            {moveKey === 'hold' ? <Pause size={10} strokeWidth={2.4} /> : null}
            {moveLabel}
          </span>
        </div>
      </div>
    </article>
  );
};
