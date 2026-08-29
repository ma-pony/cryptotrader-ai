import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import { useHitlPending } from '@/hooks/use-hitl-approvals';
import type { HitlRespond } from '@/types/api';

import { ApprovalItem } from './approval-item';

export const ApprovalQueueCard = () => {
  const { t } = useTranslation('risk');
  const { data: pending, isLoading, isError } = useHitlPending();
  const [outcome, setOutcome] = useState<HitlRespond | null>(null);

  return (
    <div className="space-y-3">
      <h2 className="text-lg font-semibold text-foreground">{t('hitl.title')}</h2>
      {isLoading ? <p className="text-sm text-muted-foreground">{t('hitl.loading')}</p> : null}
      {isError ? <p className="text-sm text-destructive">{t('hitl.error')}</p> : null}
      {outcome ? (
        <div className="rounded border border-amber-500 bg-amber-500/10 p-3 text-sm">
          <p>
            {t(`hitl.outcome_status.${outcome.approval_status}`, { defaultValue: outcome.approval_status })} ·{' '}
            {t(`hitl.outcome_status.${outcome.cycle_status}`, { defaultValue: outcome.cycle_status })} ·{' '}
            {t(`hitl.outcome_status.${outcome.execution_status}`, { defaultValue: outcome.execution_status })}
          </p>
          {outcome.requires_attention ? <p className="font-medium">{t('hitl.requires_attention')}</p> : null}
          <Link className="underline" to={`/cycles/${outcome.cycle_id}`}>
            {t('hitl.view_cycle')}
          </Link>
          <button type="button" className="ml-2 underline" onClick={() => setOutcome(null)}>
            {t('hitl.dismiss')}
          </button>
        </div>
      ) : null}
      {pending?.map((approval) => (
        <ApprovalItem key={approval.approval_id} approval={approval} onOutcome={setOutcome} />
      ))}
      {!isLoading && !isError && !pending?.length ? (
        <p className="text-sm text-muted-foreground">{t('hitl.empty')}</p>
      ) : null}
    </div>
  );
};
