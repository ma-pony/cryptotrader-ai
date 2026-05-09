/**
 * spec 018 — ArchivedRules component
 * archived rules 列表（rule_name / agent / archived_at / fundamental_streak / final_pnl_track 摘要）
 */

import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatDateTime } from '@/lib/format';

import type { ArchivedRuleItem } from '../queries';
import { useArchivedRules } from '../queries';

const ArchivedRow = ({ item }: { item: ArchivedRuleItem }) => {
  const pnlColor = item.final_pnl_track.total_pnl >= 0 ? 'text-trade-long' : 'text-trade-short';
  const pnlStr =
    `${item.final_pnl_track.total_pnl >= 0 ? '+' : ''}${item.final_pnl_track.total_pnl.toFixed(2)}`;

  return (
    <div className="flex items-center gap-3 border-b border-border py-2 last:border-0">
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-medium text-foreground/80 line-through">{item.name}</span>
          <span className="text-[10px] text-muted-foreground">{item.agent}</span>
        </div>
        <div className="flex items-center gap-3 flex-wrap text-[10px] text-muted-foreground">
          <span>
            归档：{item.archived_at ? formatDateTime(item.archived_at) : '—'}
          </span>
          <span className="text-red-400 font-medium">
            连续基本面失败 {item.fundamental_failure_streak} 次
          </span>
          <span>
            胜：{item.final_pnl_track.successes} / 败：{item.final_pnl_track.losses}
          </span>
          <span className={pnlColor}>累计 PnL {pnlStr}</span>
        </div>
      </div>
    </div>
  );
};

export const ArchivedRules = () => {
  const { t } = useTranslation('memory');
  const { data, isLoading } = useArchivedRules();
  const items = data?.items ?? [];

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium">
          {t('archived_rules.title', { defaultValue: '已归档规则' })}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="text-xs text-muted-foreground">{t('loading', { defaultValue: '加载中…' })}</div>
        ) : items.length === 0 ? (
          <div className="py-6 text-center text-xs text-muted-foreground">
            {t('archived_rules.empty', { defaultValue: '暂无归档规则' })}
          </div>
        ) : (
          <div className="max-h-72 overflow-y-auto pr-1">
            {items.map((item) => (
              <ArchivedRow key={`${item.agent}::${item.name}`} item={item} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
};
