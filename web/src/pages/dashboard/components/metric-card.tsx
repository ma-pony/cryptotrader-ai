import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/cn';

interface MetricCardProps {
  label: string;
  value: string;
  delta?: string | undefined;
  deltaDirection?: 'up' | 'down' | 'neutral' | undefined;
  isLoading?: boolean | undefined;
}

export const MetricCard = ({ label, value, delta, deltaDirection = 'neutral', isLoading }: MetricCardProps) => {
  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-4 space-y-2">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-8 w-28" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className="text-2xl font-semibold tabular-nums tracking-tight">{value}</p>
        {delta && (
          <p
            className={cn(
              'text-xs tabular-nums',
              deltaDirection === 'up' && 'text-success',
              deltaDirection === 'down' && 'text-destructive',
              deltaDirection === 'neutral' && 'text-muted-foreground',
            )}
            aria-label={`${label} delta`}
          >
            {deltaDirection === 'up' ? '↑' : deltaDirection === 'down' ? '↓' : ''} {delta}
          </p>
        )}
      </CardContent>
    </Card>
  );
};
