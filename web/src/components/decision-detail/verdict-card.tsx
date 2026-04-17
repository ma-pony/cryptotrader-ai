import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { VerdictSlim } from '@/types/api';

interface Props {
  verdict: VerdictSlim;
}

export const VerdictCard = ({ verdict }: Props) => {
  const { t } = useTranslation('decisions');

  const isWeightedDowngrade = verdict.source === 'weighted' || verdict.source === 'WEIGHTED_DOWNGRADE';

  return (
    <section aria-label={t('detail.verdict', { defaultValue: '裁定' })}>
      <Card>
        <CardHeader className="p-3 pb-1">
          <CardTitle className="text-sm flex items-center gap-2">
            {t('detail.verdict', { defaultValue: '裁定' })}
            {isWeightedDowngrade && <Badge variant="secondary">weighted-downgrade</Badge>}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-3 pt-1 space-y-1 text-sm">
          <div className="flex gap-4 text-xs">
            <span>
              <span className="text-muted-foreground">{t('detail.action', { defaultValue: '动作' })}:</span>{' '}
              <span className="font-medium uppercase">{verdict.action}</span>
            </span>
            <span>
              <span className="text-muted-foreground">{t('detail.size', { defaultValue: '仓位' })}:</span>{' '}
              <span className="font-medium tabular-nums">{(verdict.size * 100).toFixed(0)}%</span>
            </span>
            <span>
              <span className="text-muted-foreground">{t('detail.confidence', { defaultValue: '置信度' })}:</span>{' '}
              <span className="font-medium tabular-nums">{(verdict.confidence * 100).toFixed(0)}%</span>
            </span>
          </div>
          {verdict.reasoning && <p className="text-xs leading-relaxed text-muted-foreground">{verdict.reasoning}</p>}
        </CardContent>
      </Card>
    </section>
  );
};
