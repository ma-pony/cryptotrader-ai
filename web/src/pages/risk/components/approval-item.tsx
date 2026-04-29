import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ConfirmDialog } from '@/components/ui/dialog';
import { useHitlRespond } from '@/hooks/use-hitl-approvals';
import type { ApprovalRequest } from '@/types/api';

interface Props {
  approval: ApprovalRequest;
}

const actionColor: Record<string, string> = {
  long: 'text-success',
  short: 'text-destructive',
  close: 'text-warning',
};

const directionIcon: Record<string, string> = {
  bullish: '\u25B2',
  bearish: '\u25BC',
  neutral: '\u25CF',
};

export const ApprovalItem = ({ approval }: Props) => {
  const { t } = useTranslation('risk');
  const respond = useHitlRespond();
  const [confirmAction, setConfirmAction] = useState<'approve' | 'reject' | null>(null);
  const [remaining, setRemaining] = useState('');

  useEffect(() => {
    if (!approval.expires_at) return;
    const tick = () => {
      const diff = new Date(approval.expires_at!).getTime() - Date.now();
      if (diff <= 0) {
        setRemaining(t('hitl.expired'));
        return;
      }
      const m = Math.floor(diff / 60_000);
      const s = Math.floor((diff % 60_000) / 1000);
      setRemaining(`${m}:${s.toString().padStart(2, '0')}`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [approval.expires_at, t]);

  const isUrgent = (() => {
    if (!approval.expires_at) return false;
    return new Date(approval.expires_at).getTime() - Date.now() < 60_000;
  })();

  const { verdict_snapshot: v, agent_analyses_snapshot: agents } = approval;

  const reasonKey = `hitl.reason.${approval.trigger_reason}` as const;

  return (
    <>
      <Card className="border-warning/40">
        <CardHeader className="p-4 pb-2">
          <CardTitle className="text-sm flex items-center justify-between">
            <span className="flex items-center gap-2">
              <span className="font-semibold">{approval.pair}</span>
              <Badge variant="outline">{t(reasonKey)}</Badge>
            </span>
            <span className={`text-xs tabular-nums font-mono ${isUrgent ? 'text-destructive animate-pulse' : 'text-muted-foreground'}`}>
              {remaining}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="p-4 pt-0 space-y-3">
          <div className="flex items-center gap-4 text-sm">
            <span className={`font-semibold uppercase ${actionColor[v.action] ?? 'text-foreground'}`}>
              {v.action}
            </span>
            {v.position_scale != null && (
              <span className="text-muted-foreground">
                {t('hitl.position_scale')}: <span className="font-medium tabular-nums">{(v.position_scale * 100).toFixed(0)}%</span>
              </span>
            )}
            {v.confidence != null && (
              <span className="text-muted-foreground">
                {t('hitl.confidence')}: <span className="font-medium tabular-nums">{(v.confidence * 100).toFixed(0)}%</span>
              </span>
            )}
          </div>

          {agents.length > 0 && (
            <div className="flex gap-3 text-xs text-muted-foreground">
              {agents.map((a) => (
                <span key={a.agent} className="flex items-center gap-1">
                  <span>{directionIcon[a.direction] ?? '?'}</span>
                  <span className="capitalize">{a.agent}</span>
                  <span className="tabular-nums">{(a.confidence * 100).toFixed(0)}%</span>
                </span>
              ))}
            </div>
          )}

          {v.reasoning && (
            <p className="text-xs text-muted-foreground line-clamp-2">{v.reasoning}</p>
          )}

          <div className="flex gap-2">
            <Button
              size="sm"
              variant="primary"
              className="bg-success hover:bg-success/90 text-success-foreground"
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
