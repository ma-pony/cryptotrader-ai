/**
 * spec 019 — SkillsGrid component
 * skill 列表：name / scope / importance / access_count / last_accessed_at / regime_tags
 */

import { useTranslation } from 'react-i18next';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatDateTime } from '@/lib/format';

import type { SkillItem } from '../queries';
import { useSkills } from '../queries';

const SkillRow = ({ item }: { item: SkillItem }) => {
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
          <span className="text-xs font-medium text-foreground/90">{item.name}</span>
          <span className="text-[10px] text-muted-foreground">{item.scope}</span>
          {item.manually_edited && (
            <Badge variant="outline" className="text-[9px] px-1 py-0">
              edited
            </Badge>
          )}
        </div>
        {item.description && (
          <p className="text-[10px] text-muted-foreground line-clamp-1">{item.description}</p>
        )}
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-muted-foreground">
          <span className={importanceColor}>
            importance {item.importance.toFixed(2)}
          </span>
          <span>access {item.access_count}</span>
          {item.last_accessed_at && (
            <span>最后访问 {formatDateTime(item.last_accessed_at)}</span>
          )}
        </div>
        {item.regime_tags.length > 0 && (
          <div className="flex gap-1 flex-wrap">
            {item.regime_tags.map((tag) => (
              <Badge key={tag} variant="secondary" className="text-[9px] px-1 py-0">
                {tag}
              </Badge>
            ))}
          </div>
        )}
        {item.triggers_keywords.length > 0 && (
          <div className="flex gap-1 flex-wrap" data-testid="triggers-keywords">
            {item.triggers_keywords.slice(0, 5).map((kw) => (
              <Badge key={kw} variant="outline" className="text-[9px] px-1 py-0 text-muted-foreground">
                {kw}
              </Badge>
            ))}
            {item.triggers_keywords.length > 5 && (
              <span className="text-[9px] text-muted-foreground self-center">
                +{item.triggers_keywords.length - 5} more
              </span>
            )}
          </div>
        )}
        {item.inference_failed && (
          <Badge variant="destructive" className="text-[9px] px-1 py-0 w-fit" data-testid="inference-failed-badge">
            inference failed
          </Badge>
        )}
      </div>
    </div>
  );
};

export const SkillsGrid = () => {
  const { t } = useTranslation('memory');
  const { data, isLoading } = useSkills({});
  const items = data?.items ?? [];

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium">
          {t('skills.title', { defaultValue: 'Skills 列表' })}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="text-xs text-muted-foreground">{t('loading', { defaultValue: '加载中…' })}</div>
        ) : items.length === 0 ? (
          <div className="py-6 text-center text-xs text-muted-foreground">
            {t('skills.empty', { defaultValue: '暂无 skill 记录' })}
          </div>
        ) : (
          <div className="max-h-72 overflow-y-auto pr-1">
            {items.map((item) => (
              <SkillRow key={item.name} item={item} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
