import { useTranslation } from 'react-i18next';

import { cn } from '@/lib/cn';
import type { DebateRound } from '@/types/api';

interface Props {
  rounds: DebateRound[];
}

export const DebateSection = ({ rounds }: Props) => {
  const { t } = useTranslation('decisions');
  if (rounds.length === 0) return null;

  return (
    <section aria-label={t('detail.debate', { defaultValue: '多空辩论' })}>
      <h3 className="text-sm font-medium mb-2">{t('detail.debate', { defaultValue: '多空辩论' })}</h3>
      <div className="space-y-3">
        {rounds.map((round) => (
          <div key={round.round} className="space-y-1.5">
            <p className="text-[10px] text-muted-foreground font-medium">
              {t('detail.round', { defaultValue: '第 {{n}} 轮', n: round.round })}
            </p>
            {round.bull_message && (
              <div className={cn('text-xs p-2 rounded-md bg-success/10 border border-success/20')}>
                <span className="font-medium text-success">Bull:</span> {round.bull_message}
              </div>
            )}
            {round.bear_message && (
              <div className={cn('text-xs p-2 rounded-md bg-destructive/10 border border-destructive/20')}>
                <span className="font-medium text-destructive">Bear:</span> {round.bear_message}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
};
