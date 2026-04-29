import { ArrowDown, ArrowUp, Pause, X } from 'lucide-react';

export type Direction =
  | 'bullish'
  | 'bearish'
  | 'neutral'
  | 'long'
  | 'short'
  | 'hold'
  | 'close'
  | 'buy'
  | 'sell';

const MAP: Record<Direction, { label: string; tone: 'long' | 'short' | 'hold' | 'amber'; Icon: typeof ArrowUp }> = {
  long: { label: '看多', tone: 'long', Icon: ArrowUp },
  buy: { label: '买入', tone: 'long', Icon: ArrowUp },
  bullish: { label: '看多', tone: 'long', Icon: ArrowUp },
  short: { label: '看空', tone: 'short', Icon: ArrowDown },
  sell: { label: '卖出', tone: 'short', Icon: ArrowDown },
  bearish: { label: '看空', tone: 'short', Icon: ArrowDown },
  hold: { label: '观望', tone: 'hold', Icon: Pause },
  neutral: { label: '中性', tone: 'hold', Icon: Pause },
  close: { label: '平仓', tone: 'amber', Icon: X },
};

const TONE = {
  long: 'text-trade-long bg-trade-long-soft border-trade-long/40',
  short: 'text-trade-short bg-trade-short-soft border-trade-short/40',
  hold: 'text-muted-foreground bg-muted border-border',
  amber: 'text-amber-500 bg-amber-500/15 border-amber-500/35',
} as const;

interface Props {
  dir: string;
  confidence?: number | undefined;
  size?: 'sm' | 'md';
}

export const DirChip = ({ dir, confidence, size = 'sm' }: Props) => {
  const key = (dir?.toLowerCase() ?? 'hold') as Direction;
  const m = MAP[key] ?? MAP.hold;
  const Icon = m.Icon;
  const cls = TONE[m.tone];
  const pad = size === 'md' ? 'px-2.5 py-1 text-xs' : 'px-2 py-0.5 text-[11px]';
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-medium whitespace-nowrap tracking-wide ${pad} ${cls}`}
    >
      <Icon size={size === 'md' ? 12 : 10} strokeWidth={2.4} />
      {m.label}
      {confidence != null ? (
        <span className="font-mono opacity-75 ml-0.5">{(confidence * 100).toFixed(0)}%</span>
      ) : null}
    </span>
  );
};
