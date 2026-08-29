import { ArrowDown, ArrowUp, Pause, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

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

const MAP: Record<Direction, { tone: 'long' | 'short' | 'hold' | 'amber'; Icon: typeof ArrowUp }> = {
  long: { tone: 'long', Icon: ArrowUp }, buy: { tone: 'long', Icon: ArrowUp }, bullish: { tone: 'long', Icon: ArrowUp },
  short: { tone: 'short', Icon: ArrowDown }, sell: { tone: 'short', Icon: ArrowDown }, bearish: { tone: 'short', Icon: ArrowDown },
  hold: { tone: 'hold', Icon: Pause }, neutral: { tone: 'hold', Icon: Pause }, close: { tone: 'amber', Icon: X },
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
  const { t } = useTranslation('debate');
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
      {t(`direction.${key in MAP ? key : 'hold'}`)}
      {confidence != null ? (
        <span className="font-mono opacity-75 ml-0.5">{(confidence * 100).toFixed(0)}%</span>
      ) : null}
    </span>
  );
};
