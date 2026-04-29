import { Card, CardContent } from '@/components/ui/card';

interface Props {
  label: string;
  value: number | null | undefined;
  limit: number;
  unit?: string;
  precision?: number;
}

export const RiskMeter = ({ label, value, limit, unit = '%', precision = 1 }: Props) => {
  const hasValue = value != null;
  const pct = hasValue ? Math.min(1, Math.abs(value) / limit) : 0;
  const tone: 'danger' | 'warning' | 'safe' | 'unknown' = hasValue
    ? pct > 0.8
      ? 'danger'
      : pct > 0.5
        ? 'warning'
        : 'safe'
    : 'unknown';
  // FE-m6: when value is null we can't tell if the state is safe — render in a
  // neutral muted color rather than green (which implies "safe" to users).
  const color =
    tone === 'danger'
      ? 'var(--trade-short)'
      : tone === 'warning'
        ? 'var(--amber-500)'
        : tone === 'safe'
          ? 'var(--trade-long)'
          : 'hsl(var(--muted-foreground))';

  return (
    <Card>
      <CardContent className="flex flex-col gap-2 p-4">
        <div className="flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
            {label}
          </span>
          <span className="font-mono text-[10px] text-muted-foreground">
            上限 {limit}
            {unit}
          </span>
        </div>
        <div
          className="font-mono text-[22px] font-semibold leading-none tracking-tight"
          style={{ color: hasValue ? color : 'hsl(var(--muted-foreground))' }}
        >
          {hasValue ? `${value > 0 ? '+' : ''}${value.toFixed(precision)}${unit}` : '—'}
        </div>
        {/* FE-I1: ARIA meter semantics — screen readers now announce "{label}: 3.4 of 5". */}
        <div
          role="meter"
          aria-label={label}
          aria-valuemin={0}
          aria-valuemax={limit}
          aria-valuenow={hasValue ? Math.round(Math.abs(value) * 100) / 100 : undefined}
          aria-valuetext={
            hasValue ? `${value.toFixed(precision)}${unit} of ${limit}${unit}` : 'unknown'
          }
          className="relative h-1.5 overflow-hidden rounded bg-muted"
        >
          <div
            className="h-full rounded transition-[width] duration-300"
            style={{ width: `${pct * 100}%`, background: color }}
          />
        </div>
      </CardContent>
    </Card>
  );
};
