import { useTranslation } from 'react-i18next';

import type { VerdictPartialData, VerdictReadyData } from '@/types/analysis-events';

interface VerdictCardProps {
  verdict: VerdictReadyData | VerdictPartialData;
}

function isPartial(v: VerdictReadyData | VerdictPartialData): v is VerdictPartialData {
  return 'is_partial' in v && v.is_partial === true;
}

const ACTION_COLORS: Record<string, string> = {
  long: 'text-green-600 dark:text-green-400',
  short: 'text-red-600 dark:text-red-400',
  hold: 'text-yellow-600 dark:text-yellow-400',
};

export function VerdictCard({ verdict }: VerdictCardProps) {
  const { t } = useTranslation('chat');
  const partial = isPartial(verdict);

  return (
    <div className="rounded-lg border p-4 bg-card text-card-foreground">
      {partial && (
        <div className="mb-2 rounded bg-orange-100 dark:bg-orange-900/30 px-3 py-1 text-xs text-orange-700 dark:text-orange-300">
          {t('partial_verdict', { defaultValue: 'Partial verdict (analysis interrupted)' })}
        </div>
      )}

      <div className="flex items-center gap-4">
        <span className={`text-lg font-bold ${ACTION_COLORS[verdict.action] ?? ''}`}>
          {verdict.action.toUpperCase()}
        </span>
        <span className="text-sm text-muted-foreground">
          {(verdict.confidence * 100).toFixed(0)}%
        </span>
        <span className="text-sm text-muted-foreground">
          {t('scale', { defaultValue: 'Scale' })}: {(verdict.position_scale * 100).toFixed(1)}%
        </span>
      </div>

      {verdict.reasoning && (
        <p className="mt-2 text-sm text-muted-foreground line-clamp-3">{verdict.reasoning}</p>
      )}

      {partial && (
        <div className="mt-2 text-xs text-muted-foreground">
          {t('completed_agents', { defaultValue: 'Completed' })}: {verdict.completed_agents.join(', ')}
          {verdict.missing_agents.length > 0 && (
            <span className="ml-2">
              {t('missing_agents', { defaultValue: 'Missing' })}: {verdict.missing_agents.join(', ')}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
