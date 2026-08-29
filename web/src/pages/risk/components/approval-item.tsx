import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ConfirmDialog } from '@/components/ui/dialog';
import { useHitlRespond } from '@/hooks/use-hitl-approvals';
import { formatDateTime } from '@/lib/format';
import type { ApprovalRequest } from '@/types/api';

interface Props {
  approval: ApprovalRequest;
  onOutcome?: (outcome: { approval_id: string; cycle_id: string; approval_status: string; cycle_status: string; execution_status: string; requires_attention: boolean }) => void;
}

export const ApprovalItem = ({ approval, onOutcome }: Props) => {
  const { t } = useTranslation('risk');
  const respond = useHitlRespond();
  const [confirmAction, setConfirmAction] = useState<'approve' | 'reject' | null>(null);
  const { proposal } = approval;

  return (
    <>
      <Card className="overflow-hidden border-amber-500/40">
        <div className="h-1 bg-amber-500" />
        <CardHeader className="p-4 pb-2">
          <CardTitle className="flex flex-wrap items-start justify-between gap-3 text-sm">
            <span>
              <span className="font-semibold">{approval.book_id} · {approval.pair}</span>
              <span className="ml-2 font-mono text-[10px] text-muted-foreground">{approval.cycle_id}</span>
            </span>
            <span className="flex items-center gap-2">
              <Badge variant="secondary">{t('hitl.config_revision', { defaultValue: 'Config revision' })} {approval.config_revision}</Badge>
              <span className="font-mono text-[10px] text-muted-foreground">{formatDateTime(approval.created_at)}</span>
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 p-4 pt-1">
          <div className="flex flex-wrap items-end justify-between gap-4 rounded-lg bg-muted/40 p-3">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{t('hitl.frozen_book_plan', { defaultValue: 'Frozen book proposal' })}</div>
              <div className="mt-1 text-lg font-semibold">{proposal.capital_scope} · {proposal.target_exposure}</div>
            </div>
          </div>
          <div className="space-y-2 text-xs">{proposal.connection_plans.map((plan) => <div key={plan.connection_id} className="rounded-md border border-border p-3"><div className="font-medium">{plan.connection_id} · {plan.market_type}</div><div className="mt-1 font-mono text-muted-foreground">{plan.side} {plan.amount} · {plan.target_signed_notional}</div></div>)}</div>
          {proposal.errors.map((error) => <p key={error} className="text-xs text-destructive">{error}</p>)}

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
        onConfirm={async () => { onOutcome?.(await respond.mutateAsync({ approvalId: approval.approval_id, decision: 'approve' })); }}
      />
      <ConfirmDialog
        open={confirmAction === 'reject'}
        onOpenChange={(open) => { if (!open) setConfirmAction(null); }}
        title={t('hitl.confirm_reject_title')}
        body={t('hitl.confirm_reject_body')}
        confirmLabel={t('hitl.confirm_reject_action')}
        destructive
        onConfirm={async () => { onOutcome?.(await respond.mutateAsync({ approvalId: approval.approval_id, decision: 'reject' })); }}
      />
    </>
  );
};
