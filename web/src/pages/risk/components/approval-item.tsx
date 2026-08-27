import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ConfirmDialog } from '@/components/ui/dialog';
import { useHitlRespond } from '@/hooks/use-hitl-approvals';
import { cn } from '@/lib/cn';
import { formatCurrency, formatDateTime } from '@/lib/format';
import type { ApprovalRequest, TargetPosition } from '@/types/api';

interface Props {
  approval: ApprovalRequest;
}

const targetLabel = (target: TargetPosition) => {
  const ratio = `${(target.size_ratio * 100).toFixed(0)}%`;
  if (target.side === 'long') return `目标多仓 ${ratio}`;
  if (target.side === 'short') return `目标空仓 ${ratio}`;
  return '目标清仓';
};

export const ApprovalItem = ({ approval }: Props) => {
  const { t } = useTranslation('risk');
  const respond = useHitlRespond();
  const [confirmAction, setConfirmAction] = useState<'approve' | 'reject' | null>(null);
  const { trade_plan: plan } = approval;

  return (
    <>
      <Card className="overflow-hidden border-amber-500/40">
        <div className="h-1 bg-amber-500" />
        <CardHeader className="p-4 pb-2">
          <CardTitle className="flex flex-wrap items-start justify-between gap-3 text-sm">
            <span>
              <span className="font-semibold">{approval.pair}</span>
              <span className="ml-2 font-mono text-[10px] text-muted-foreground">{approval.cycle_id}</span>
            </span>
            <span className="flex items-center gap-2">
              <Badge variant="secondary">Revision {approval.profile_revision}</Badge>
              <span className="font-mono text-[10px] text-muted-foreground">{formatDateTime(approval.created_at)}</span>
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 p-4 pt-1">
          <div className="flex flex-wrap items-end justify-between gap-4 rounded-lg bg-muted/40 p-3">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">冻结交易计划</div>
              <div className={cn('mt-1 text-lg font-semibold', plan.target.side === 'long' ? 'text-trade-long' : plan.target.side === 'short' ? 'text-trade-short' : 'text-muted-foreground')}>
                {targetLabel(plan.target)}
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">融合分数</div>
              <div className="font-mono text-lg font-semibold">{plan.fused_signal.score >= 0 ? '+' : ''}{plan.fused_signal.score.toFixed(2)}</div>
            </div>
          </div>

          <dl className="grid grid-cols-2 gap-3 text-xs">
            <div className="rounded-md border border-border p-3"><dt className="text-muted-foreground">ATR 止损</dt><dd className="mt-1 font-mono font-medium">{plan.stop_loss === null ? '—' : formatCurrency(plan.stop_loss)}</dd></div>
            <div className="rounded-md border border-border p-3"><dt className="text-muted-foreground">目标止盈</dt><dd className="mt-1 font-mono font-medium">{plan.take_profit === null ? '—' : formatCurrency(plan.take_profit)}</dd></div>
          </dl>

          <p className="text-xs leading-5 text-muted-foreground">批准后会用最新账户状态重新执行风控，再按这份冻结目标计划下单。</p>

          <div className="flex gap-2">
            <Button size="sm" variant="primary" className="bg-success text-success-foreground hover:bg-success/90" onClick={() => setConfirmAction('approve')} disabled={respond.isPending}>{t('hitl.approve')}</Button>
            <Button size="sm" variant="destructive" onClick={() => setConfirmAction('reject')} disabled={respond.isPending}>{t('hitl.reject')}</Button>
          </div>
        </CardContent>
      </Card>

      <ConfirmDialog
        open={confirmAction === 'approve'}
        onOpenChange={(open) => { if (!open) setConfirmAction(null); }}
        title={t('hitl.confirm_approve_title')}
        body={t('hitl.confirm_approve_body')}
        confirmLabel={t('hitl.confirm_approve_action')}
        destructive={false}
        onConfirm={async () => { await respond.mutateAsync({ approvalId: approval.approval_id, decision: 'approve' }); }}
      />
      <ConfirmDialog
        open={confirmAction === 'reject'}
        onOpenChange={(open) => { if (!open) setConfirmAction(null); }}
        title={t('hitl.confirm_reject_title')}
        body={t('hitl.confirm_reject_body')}
        confirmLabel={t('hitl.confirm_reject_action')}
        destructive
        onConfirm={async () => { await respond.mutateAsync({ approvalId: approval.approval_id, decision: 'reject' }); }}
      />
    </>
  );
};
