import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { AgentAnalysis } from '@/types/api';

interface Props {
  analyses: AgentAnalysis[];
}

const scoreLabel = (score: number) => {
  if (score > 0.3) return 'bullish';
  if (score < -0.3) return 'bearish';
  return 'neutral';
};

const scoreBadgeVariant = (score: number): 'default' | 'destructive' | 'secondary' => {
  if (score > 0.3) return 'default';
  if (score < -0.3) return 'destructive';
  return 'secondary';
};

export const AgentAnalysisGrid = ({ analyses }: Props) => {
  const { t } = useTranslation('decisions');
  if (analyses.length === 0) return null;

  return (
    <section aria-label={t('detail.agents', { defaultValue: '代理分析' })}>
      <h3 className="text-sm font-medium mb-2">{t('detail.agents', { defaultValue: '代理分析' })}</h3>
      <div className="grid grid-cols-2 gap-3">
        {analyses.map((a) => (
          <Card key={a.name} className="text-sm">
            <CardHeader className="p-3 pb-1">
              <CardTitle className="text-xs flex items-center gap-2">
                {a.name}
                <Badge variant={scoreBadgeVariant(a.score)}>{scoreLabel(a.score)}</Badge>
                {a.is_mock && <Badge variant="secondary">mock</Badge>}
              </CardTitle>
            </CardHeader>
            <CardContent className="p-3 pt-1 space-y-1">
              <p className="text-xs text-muted-foreground">
                {t('detail.confidence', { defaultValue: '置信度' })}: {(a.confidence * 100).toFixed(0)}%
              </p>
              <p className="text-xs leading-relaxed">{a.reasoning}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
};
