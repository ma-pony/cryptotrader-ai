import { useTranslation } from 'react-i18next';

import { useHitlPending } from '@/hooks/use-hitl-approvals';

import { ApprovalItem } from './approval-item';

export const ApprovalQueueCard = () => {
  const { t } = useTranslation('risk');
  const { data: pending } = useHitlPending();

  if (!pending?.length) return null;

  return (
    <div className="space-y-3">
      <h2 className="text-lg font-semibold text-foreground">{t('hitl.title')}</h2>
      {pending.map((a) => (
        <ApprovalItem key={a.approval_id} approval={a} />
      ))}
    </div>
  );
};
