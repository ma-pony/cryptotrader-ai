import { Activity, Flame, Link2, Scale, Zap } from 'lucide-react';

import { AGENTS, type AgentKind } from '../constants';

const ICONS: Record<AgentKind, typeof Activity> = {
  tech: Activity,
  chain: Link2,
  news: Flame,
  macro: Scale,
  verdict: Zap,
  other: Activity,
};

interface Props {
  kind: AgentKind;
  size?: number;
  showName?: boolean;
}

export const AgentBadge = ({ kind, size = 28, showName = false }: Props) => {
  const a = AGENTS[kind];
  const Icon = ICONS[kind];
  const inner = Math.round(size * 0.55);
  return (
    <span className="inline-flex items-center gap-2" data-agent={kind}>
      <span
        className="inline-flex items-center justify-center rounded-md"
        style={{
          width: size,
          height: size,
          background: `color-mix(in oklch, ${a.color} 18%, transparent)`,
          border: `1px solid color-mix(in oklch, ${a.color} 40%, transparent)`,
          color: a.color,
        }}
      >
        <Icon size={inner} strokeWidth={1.8} />
      </span>
      {showName ? (
        <span className="text-xs font-medium" style={{ color: a.color }}>
          {a.zh}
        </span>
      ) : null}
    </span>
  );
};
