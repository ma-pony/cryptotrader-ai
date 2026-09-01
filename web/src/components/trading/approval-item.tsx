import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ConfirmDialog } from '@/components/ui/dialog';
import { useHitlRespond } from '@/hooks/use-hitl-approvals';
import { formatDateTime } from '@/lib/format';
import { formatCycleStatus } from '@/lib/cycle-status';
import type { ApprovalRequest } from '@/types/api';

interface Props {
  approval: ApprovalRequest;
  onOutcome?: (outcome: {
    approval_id: string;
    cycle_id: string;
    approval_status: string;
    cycle_status: string;
    execution_status: string;
    requires_attention: boolean;
  }) => void;
}

export const ApprovalItem = ({ approval, onOutcome }: Props) => {
  const { t } = useTranslation('risk');
  const respond = useHitlRespond();
  const [confirmAction, setConfirmAction] = useState<'approve' | 'reject' | null>(null);
  const [mutationError, setMutationError] = useState(false);
  const { proposal } = approval;

  return (
    <>
      <Card className="overflow-hidden border-amber-500/40">
        <div className="h-1 bg-amber-500" />
        <CardHeader className="p-4 pb-2">
          <CardTitle className="flex flex-wrap items-start justify-between gap-3 text-sm">
            <span>
              <span className="font-semibold">
                {approval.book_id} · {approval.pair}
              </span>
              <span className="ml-2 font-mono text-[10px] text-muted-foreground">{approval.cycle_id}</span>
            </span>
            <span className="flex items-center gap-2">
              <Badge variant="secondary">
                {t('hitl.config_revision')} {approval.config_revision}
              </Badge>
              <span className="font-mono text-[10px] text-muted-foreground">{formatDateTime(approval.created_at)}</span>
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 p-4 pt-1">
          <div className="flex flex-wrap items-end justify-between gap-4 rounded-lg bg-muted/40 p-3">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                {t('hitl.frozen_book_plan')}
              </div>
              <div className="mt-1 text-lg font-semibold">
                {proposal.capital_scope} · {proposal.target_exposure}
              </div>
            </div>
          </div>
          <details className="rounded border border-border p-3" open>
            <summary className="font-medium">{t('hitl.frozen_book_plan')}</summary>
            <dl className="mt-2 grid gap-1 text-xs">
              <div>
                {t('hitl.scope')}: {proposal.capital_scope}
              </div>
              <div>
                {t('hitl.status')}: {formatCycleStatus(t, approval.status)}
              </div>
              <div>
                {t('hitl.created_at')}: {approval.created_at}
              </div>
              {approval.decided_at ? (
                <div>
                  {t('hitl.decided_at')}: {approval.decided_at}
                </div>
              ) : null}
              <div>
                {t('hitl.requested')}: {proposal.requested_target_exposure}
              </div>
              <div>
                {t('hitl.target')}: {proposal.target_exposure}
              </div>
              <div>
                {t('hitl.risk')}:{' '}
                {proposal.risk.cap_source === 'max_drawdown' && Number(proposal.target_exposure) === 0
                  ? '风险限制：目标持仓降至零'
                  : proposal.risk.reason}
              </div>
              {proposal.risk.state ? (
                <div>
                  整池风控观察时间：{proposal.risk.state.observed_at} · 当前占用{' '}
                  {proposal.risk.state.gross_notional ?? '未知'} · 待成交增仓{' '}
                  {proposal.risk.state.pending_increase_notional ?? '未知'} {proposal.risk.state.valuation_currency}
                </div>
              ) : null}
              <p>批准前将刷新全部成员账户。原数量不能安全执行时审批失效，不会自动调整数量。</p>
              <div>
                {t('hitl.connection_targets')}:{' '}
                {proposal.risk.connection_targets
                  .map((target) => `${target.connection_id} ${target.target_exposure}`)
                  .join(', ')}
              </div>
              <div>
                {t('hitl.connection_risks')}:{' '}
                {proposal.connection_risks.map((risk) => `${risk.connection_id} ${risk.reason}`).join(', ')}
              </div>
              <div>
                {t('hitl.unavailable')}: {proposal.unavailable_connections.join(', ') || '—'}
              </div>
            </dl>
            <div className="mt-3 space-y-2 text-xs">
              {proposal.connection_plans.map((plan) => (
                <article key={plan.connection_id} className="rounded-md border border-border p-3">
                  <div className="font-medium">
                    {plan.connection_id} · {plan.pair.symbol}
                  </div>
                  <div>
                    {t('hitl.quote')}: {plan.quote.bid}/{plan.quote.ask}/{plan.quote.last}
                  </div>
                  <div>
                    {t('hitl.amount')}: {plan.amount}; {t('hitl.notional')}: {plan.target_signed_notional}
                  </div>
                  <div>
                    {t('hitl.side')}: {plan.side}; {t('hitl.reduce_only')}: {String(plan.reduce_only)};{' '}
                    {t('hitl.market')}: {plan.market_type}
                  </div>
                  <div>
                    {t('hitl.protection')}: {plan.stop_loss}/{plan.take_profit}
                  </div>
                  <div>
                    {t('hitl.old_protections')}: {plan.old_protection_ids.join(', ') || '—'}
                  </div>
                  <div>
                    {t('hitl.capabilities')}: {plan.capabilities.market_types.join(', ')} ·{' '}
                    {plan.capabilities.supported_order_types.join(', ')}
                  </div>
                </article>
              ))}
            </div>
          </details>
          {proposal.errors.map((error) => (
            <p key={error} className="text-xs text-destructive">
              {error}
            </p>
          ))}
          {mutationError ? <p className="text-xs text-destructive">{t('hitl.response_error')}</p> : null}

          <div className="flex gap-2">
            <Button
              size="sm"
              variant="primary"
              className="bg-success text-success-foreground hover:bg-success/90"
              onClick={() => setConfirmAction('approve')}
              disabled={respond.isPending}
            >
              {t('hitl.approve')}
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={() => setConfirmAction('reject')}
              disabled={respond.isPending}
            >
              {t('hitl.reject')}
            </Button>
          </div>
        </CardContent>
      </Card>

      <ConfirmDialog
        open={confirmAction === 'approve'}
        onOpenChange={(open) => {
          if (!open) setConfirmAction(null);
        }}
        title={t('hitl.confirm_approve_title')}
        body={t('hitl.confirm_approve_body')}
        confirmLabel={t('hitl.confirm_approve_action')}
        destructive={false}
        initialFocus="confirm"
        onConfirm={async () => {
          try {
            setMutationError(false);
            onOutcome?.(await respond.mutateAsync({ approvalId: approval.approval_id, decision: 'approve' }));
          } catch {
            setMutationError(true);
          }
        }}
      />
      <ConfirmDialog
        open={confirmAction === 'reject'}
        onOpenChange={(open) => {
          if (!open) setConfirmAction(null);
        }}
        title={t('hitl.confirm_reject_title')}
        body={t('hitl.confirm_reject_body')}
        confirmLabel={t('hitl.confirm_reject_action')}
        destructive
        onConfirm={async () => {
          try {
            setMutationError(false);
            onOutcome?.(await respond.mutateAsync({ approvalId: approval.approval_id, decision: 'reject' }));
          } catch {
            setMutationError(true);
          }
        }}
      />
    </>
  );
};
