import { useState } from 'react';
import { Link } from 'react-router';
import { useTranslation } from 'react-i18next';

import { useHitlPending } from '@/hooks/use-hitl-approvals';

import { ApprovalItem } from './approval-item';

export const ApprovalQueueCard = () => {
  const { t } = useTranslation('risk');
  const { data: pending, isLoading, isError } = useHitlPending();
  const [outcome, setOutcome] = useState<{ approval_id: string; cycle_id: string; approval_status: string; cycle_status: string; execution_status: string; requires_attention: boolean } | null>(null);

  if (isLoading) return <p className="text-sm text-muted-foreground">{t('hitl.loading', { defaultValue: 'Loading approvals…' })}</p>;
  if (isError) return <p className="text-sm text-destructive">{t('hitl.error', { defaultValue: 'Unable to load approvals.' })}</p>;

  return (
    <div className="space-y-3">
      <h2 className="text-lg font-semibold text-foreground">{t('hitl.title')}</h2>
      {outcome ? <div className="rounded border border-amber-500 bg-amber-500/10 p-3 text-sm">{outcome.approval_status} · {outcome.cycle_status} · {outcome.execution_status}{outcome.requires_attention ? ' · requires attention' : ''} <Link className="underline" to={`/cycles/${outcome.cycle_id}`}>View cycle</Link><button className="ml-2 underline" onClick={() => setOutcome(null)}>Dismiss</button></div> : null}
      {pending?.map((a) => (
        <ApprovalItem key={a.approval_id} approval={a} onOutcome={setOutcome} />
      ))}
      {!pending?.length ? <p className="text-sm text-muted-foreground">{t('hitl.empty', { defaultValue: 'No approvals pending.' })}</p> : null}
    </div>
  );
};
