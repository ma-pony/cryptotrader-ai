import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';

type Tone = 'default' | 'long' | 'short' | 'hold' | 'amber' | 'violet' | 'cyan' | 'danger' | 'success' | 'warning';

const TONE: Record<Tone, string> = {
  default: 'text-muted-foreground bg-muted border-border',
  long: 'text-trade-long bg-trade-long-soft border-trade-long/35',
  short: 'text-trade-short bg-trade-short-soft border-trade-short/35',
  hold: 'text-muted-foreground bg-muted border-border',
  amber: 'text-amber-500 bg-amber-500/15 border-amber-500/35',
  violet: 'text-violet-500 bg-violet-500/15 border-violet-500/35',
  cyan: 'text-cyan-500 bg-cyan-500/15 border-cyan-500/35',
  danger: 'text-trade-short bg-trade-short-soft border-trade-short/35',
  success: 'text-trade-long bg-trade-long-soft border-trade-long/35',
  warning: 'text-amber-500 bg-amber-500/15 border-amber-500/35',
};

const DOT: Record<Tone, string> = {
  default: 'bg-muted-foreground',
  long: 'bg-trade-long',
  short: 'bg-trade-short',
  hold: 'bg-muted-foreground',
  amber: 'bg-amber-500',
  violet: 'bg-violet-500',
  cyan: 'bg-cyan-500',
  danger: 'bg-trade-short',
  success: 'bg-trade-long',
  warning: 'bg-amber-500',
};

interface Props {
  tone?: Tone;
  live?: boolean;
  children: ReactNode;
  className?: string;
}

export const StatusPill = ({ tone = 'default', live = false, children, className }: Props) => (
  <span
    className={cn(
      'inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium whitespace-nowrap tracking-wider uppercase',
      TONE[tone],
      className,
    )}
  >
    {live ? <span className={cn('h-1.5 w-1.5 rounded-full animate-ct-pulse', DOT[tone])} /> : null}
    {children}
  </span>
);
