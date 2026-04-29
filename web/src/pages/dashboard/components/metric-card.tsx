import type { ReactNode } from 'react';

import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Sparkline } from '@/components/ui/sparkline';
import { cn } from '@/lib/cn';

interface MetricCardProps {
  label: ReactNode;
  value: string;
  delta?: string | undefined;
  deltaDirection?: 'up' | 'down' | 'neutral' | undefined;
  sub?: ReactNode;
  trend?: number[] | undefined;
  trendColor?: string;
  isLoading?: boolean | undefined;
}

export const MetricCard = ({
  label,
  value,
  delta,
  deltaDirection = 'neutral',
  sub,
  trend,
  trendColor = 'var(--amber-500)',
  isLoading,
}: MetricCardProps) => {
  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-4 space-y-2">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-8 w-28" />
          <Skeleton className="h-3 w-16" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
          {label}
        </div>
        <div className="mt-1.5 flex items-baseline gap-2">
          <div className="text-2xl font-semibold tabular-nums tracking-tight leading-none">{value}</div>
          {delta ? (
            <div
              className={cn(
                'font-mono text-xs',
                deltaDirection === 'up' && 'text-trade-long',
                deltaDirection === 'down' && 'text-trade-short',
                deltaDirection === 'neutral' && 'text-muted-foreground',
              )}
            >
              {deltaDirection === 'up' ? '↑' : deltaDirection === 'down' ? '↓' : ''} {delta}
            </div>
          ) : null}
          {trend && trend.length > 1 ? (
            <div className="ml-auto" style={{ color: trendColor }}>
              <Sparkline data={trend} width={60} height={18} />
            </div>
          ) : null}
        </div>
        {sub ? <div className="mt-1 text-[10px] text-muted-foreground">{sub}</div> : null}
      </CardContent>
    </Card>
  );
};
