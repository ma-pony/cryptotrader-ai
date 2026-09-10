/**
 * spec 019 — SkillsGrid component
 * skill 列表：name / scope / importance / access_count / last_accessed_at / regime_tags
 */

import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { PageBoundary } from '@/components/ui/page-boundary';
import { formatDateTime } from '@/lib/format';

import type { SkillItem } from './queries';
import { useSkills } from './queries';

const SkillRow = ({ item }: { item: SkillItem }) => {
  const { t } = useTranslation('memory');
  const importanceColor =
    item.importance >= 0.7
      ? 'text-trade-long'
      : item.importance >= 0.4
        ? 'text-foreground/80'
        : 'text-muted-foreground';

  return (
    <div className="flex items-start gap-3 border-b border-border py-2 last:border-0">
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="break-all text-sm font-medium text-foreground/90">{item.name}</span>
          <span className="text-sm text-muted-foreground">{item.scope}</span>
          {item.manually_edited && <Badge variant="outline">{t('skills.edited')}</Badge>}
        </div>
        {item.description && <p className="text-sm text-muted-foreground">{item.description}</p>}
        <div className="flex items-center gap-3 flex-wrap text-sm text-muted-foreground">
          <span className={importanceColor}>
            {t('skills.importance')} {item.importance.toFixed(2)}
          </span>
          <span>{t('skills.accessCount', { count: item.access_count })}</span>
          {item.last_accessed_at && (
            <span>
              {t('skills.lastAccess')} {formatDateTime(item.last_accessed_at)}
            </span>
          )}
        </div>
        {item.regime_tags.length > 0 && (
          <div className="flex gap-1 flex-wrap">
            {item.regime_tags.map((tag) => (
              <Badge key={tag} variant="secondary" aria-label={t('skills.regime', { tag })}>
                {tag}
              </Badge>
            ))}
          </div>
        )}
        {item.triggers_keywords.length > 0 && (
          <div className="flex gap-1 flex-wrap" data-testid="triggers-keywords">
            {item.triggers_keywords.slice(0, 5).map((kw) => (
              <Badge
                key={kw}
                variant="outline"
                className="text-muted-foreground"
                aria-label={t('skills.keyword', { keyword: kw })}
              >
                {kw}
              </Badge>
            ))}
            {item.triggers_keywords.length > 5 && (
              <span className="text-sm text-muted-foreground self-center">
                {t('skills.more', { count: item.triggers_keywords.length - 5 })}
              </span>
            )}
          </div>
        )}
        {item.inference_failed && (
          <Badge
            variant="destructive"
            className="w-fit"
            data-testid="inference-failed-badge"
            aria-label={t('skills.inferenceFailed')}
          >
            {t('skills.inferenceFailed')}
          </Badge>
        )}
      </div>
    </div>
  );
};

export const SkillsGrid = () => {
  const { t } = useTranslation('memory');
  const { data, isLoading, isError, refetch } = useSkills({});
  const items = data?.items ?? [];

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle id="memory-skills-heading" className="text-sm font-medium">
          {t('skills.title')}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <PageBoundary
          loading={isLoading}
          isError={isError}
          onRetry={() => void refetch()}
          loadingFallback={<p role="status">{t('loading')}</p>}
          errorTitle={t('skills.loadFailed', { defaultValue: '技能资料加载失败' })}
        >
          {items.length === 0 ? (
            <EmptyState title={t('skills.empty')} />
          ) : (
            <div className="space-y-2">
              {items.map((item) => (
                <SkillRow key={item.name} item={item} />
              ))}
            </div>
          )}
        </PageBoundary>
      </CardContent>
    </Card>
  );
};
