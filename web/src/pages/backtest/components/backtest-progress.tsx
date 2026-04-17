import { Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useCancelBacktest } from '@/hooks/use-backtest';
import type { BacktestRunStatus } from '@/types/api';

interface Props {
  run: BacktestRunStatus;
}

const statusVariant = (s: string): 'default' | 'success' | 'destructive' | 'warning' | 'secondary' => {
  if (s === 'completed') return 'success';
  if (s === 'failed') return 'destructive';
  if (s === 'running') return 'warning';
  return 'secondary';
};

export const BacktestProgress = ({ run }: Props) => {
  const { t } = useTranslation('backtest');
  const cancel = useCancelBacktest();
  const isActive = run.status === 'queued' || run.status === 'running';

  return (
    <div className="flex items-center gap-3 text-sm">
      {isActive && <Loader2 className="h-4 w-4 animate-spin text-primary" aria-hidden />}
      <Badge variant={statusVariant(run.status)}>{t(`status.${run.status.toUpperCase()}`, { defaultValue: run.status })}</Badge>
      <span className="tabular-nums text-muted-foreground">{Math.round(run.progress * 100)}%</span>
      {run.error && <span className="text-destructive text-xs truncate max-w-sm">{run.error}</span>}
      {isActive && (
        <Button variant="ghost" size="sm" onClick={() => cancel.mutate(run.run_id)} disabled={cancel.isPending}>
          {t('form.cancel')}
        </Button>
      )}
    </div>
  );
};
