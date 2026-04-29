import { Shield } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { ConfirmDialog } from '@/components/ui/dialog';
import { StatusPill } from '@/components/ui/status-pill';
import { useResetCircuitBreaker } from '@/hooks/use-risk-status';
import { formatRelative } from '@/lib/format';
import type { CircuitBreakerStatus } from '@/types/api';

interface Props {
  cb: CircuitBreakerStatus;
  redisAvailable: boolean;
  checksOnline: number;
}

export const CircuitBreakerHero = ({ cb, redisAvailable, checksOnline }: Props) => {
  const { t } = useTranslation('risk');
  const resetMutation = useResetCircuitBreaker();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const tripped = cb.state === 'active';

  const gradient = tripped
    ? 'linear-gradient(135deg, color-mix(in oklch, var(--trade-short) 15%, transparent), hsl(var(--card)))'
    : 'linear-gradient(135deg, color-mix(in oklch, var(--trade-long) 6%, transparent), hsl(var(--card)))';
  const borderColor = tripped
    ? 'color-mix(in oklch, var(--trade-short) 35%, transparent)'
    : 'color-mix(in oklch, var(--trade-long) 35%, transparent)';
  const iconBg = tripped
    ? 'color-mix(in oklch, var(--trade-short) 18%, transparent)'
    : 'color-mix(in oklch, var(--trade-long) 18%, transparent)';
  const iconColor = tripped ? 'var(--trade-short)' : 'var(--trade-long)';

  return (
    <>
      <Card style={{ background: gradient, borderColor }} className="overflow-hidden">
        <CardContent className="flex items-center gap-4 p-5">
          <div
            className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border"
            style={{ background: iconBg, borderColor, color: iconColor }}
          >
            <Shield className="h-7 w-7" strokeWidth={1.8} aria-hidden />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-[10px] uppercase tracking-wider font-medium text-muted-foreground">
              {t('circuit_breaker.title', { defaultValue: '熔断器状态' })}
            </div>
            <div className="text-xl font-semibold tracking-tight mt-0.5">
              {tripped
                ? t('circuit_breaker.tripped', { defaultValue: '已触发 — 交易暂停' })
                : t('circuit_breaker.normal', { defaultValue: '正常 · 所有闸门开放' })}
            </div>
            <div className="mt-1.5 flex items-center gap-3 text-xs text-muted-foreground">
              <span>
                Redis:{' '}
                <span className={redisAvailable ? 'text-trade-long' : 'text-trade-short'}>
                  {redisAvailable ? 'healthy' : 'unavailable'}
                </span>
              </span>
              <span className="h-3 w-px bg-border" />
              <span>{checksOnline} 项检查在线</span>
              {tripped && cb.reason ? (
                <>
                  <span className="h-3 w-px bg-border" />
                  <span className="text-trade-short truncate">{cb.reason}</span>
                </>
              ) : null}
              {tripped && cb.expires_at ? (
                <>
                  <span className="h-3 w-px bg-border" />
                  <span className="tabular-nums">
                    剩余 {formatRelative(cb.expires_at)}
                  </span>
                </>
              ) : null}
            </div>
          </div>
          {tripped ? (
            <Button variant="destructive" onClick={() => setConfirmOpen(true)}>
              {t('circuit_breaker.reset', { defaultValue: '重置熔断器' })}
            </Button>
          ) : (
            <StatusPill tone="success" live>
              运行中
            </StatusPill>
          )}
        </CardContent>
      </Card>
      <ConfirmDialog
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title={t('circuit_breaker.confirm_title', { defaultValue: '重置熔断器' })}
        body={t('circuit_breaker.confirm_body', { defaultValue: '确认清空熔断并恢复交易？' })}
        confirmLabel={t('circuit_breaker.confirm_action', { defaultValue: '立即重置' })}
        destructive
        onConfirm={() => void resetMutation.mutateAsync()}
      />
    </>
  );
};
