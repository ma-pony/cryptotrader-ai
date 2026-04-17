import { CheckCircle2, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/cn';
import type { RiskGate } from '@/types/api';

interface Props {
  gate: RiskGate;
}

export const RiskGateSection = ({ gate }: Props) => {
  const { t } = useTranslation('decisions');
  if (gate.checks.length === 0) return null;

  return (
    <section aria-label={t('detail.risk_gate', { defaultValue: '风控门' })}>
      <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
        {t('detail.risk_gate', { defaultValue: '风控门' })}
        <Badge variant={gate.passed ? 'success' : 'destructive'}>
          {gate.passed ? 'PASS' : 'REJECT'}
        </Badge>
      </h3>
      <div className="space-y-1">
        {gate.checks.map((check) => (
          <div
            key={check.name}
            className={cn(
              'flex items-start gap-2 text-xs p-1.5 rounded',
              check.passed ? 'text-muted-foreground' : 'text-destructive bg-destructive/5',
            )}
          >
            {check.passed ? (
              <CheckCircle2 className="h-3.5 w-3.5 mt-0.5 shrink-0 text-success" aria-hidden />
            ) : (
              <XCircle className="h-3.5 w-3.5 mt-0.5 shrink-0 text-destructive" aria-hidden />
            )}
            <div>
              <span className="font-medium">{check.name}</span>
              {check.reason && <span className="ml-1 text-muted-foreground">— {check.reason}</span>}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
