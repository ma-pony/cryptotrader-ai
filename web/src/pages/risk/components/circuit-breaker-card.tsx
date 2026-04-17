import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ConfirmDialog } from '@/components/ui/dialog';
import { useResetCircuitBreaker } from '@/hooks/use-risk-status';
import { formatRelative } from '@/lib/format';
import type { CircuitBreakerStatus } from '@/types/api';

interface Props {
  cb: CircuitBreakerStatus;
}

export const CircuitBreakerCard = ({ cb }: Props) => {
  const { t } = useTranslation('risk');
  const resetMutation = useResetCircuitBreaker();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const isActive = cb.state === 'active';

  return (
    <>
      <Card>
        <CardHeader className="p-4 pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            {t('circuit_breaker.title')}
            <Badge variant={isActive ? 'destructive' : 'success'}>
              {t(`circuit_breaker.${cb.state.toUpperCase()}`)}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0 space-y-2">
          {isActive && cb.reason && (
            <p className="text-xs text-destructive">{cb.reason}</p>
          )}
          {isActive && cb.expires_at && (
            <div className="text-xs text-muted-foreground">
              <span>{t('circuit_breaker.remaining')}: </span>
              <span className="tabular-nums font-medium">{formatRelative(cb.expires_at)}</span>
            </div>
          )}
          {isActive && (
            <Button variant="destructive" size="sm" onClick={() => setConfirmOpen(true)}>
              {t('circuit_breaker.reset')}
            </Button>
          )}
        </CardContent>
      </Card>

      <ConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title={t('circuit_breaker.confirm_title')}
        body={t('circuit_breaker.confirm_body')}
        confirmLabel={t('circuit_breaker.confirm_action')}
        destructive
        onConfirm={() => void resetMutation.mutateAsync()}
      />
    </>
  );
};
