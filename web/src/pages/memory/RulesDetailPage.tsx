/**
 * spec 018 / 021 — RulesDetailPage
 *
 * /memory/rules?agent=<agent>&status=<maturity>
 *
 * 完整 rule 数据表（字段：name / description / maturity / pnl_track /
 * regime_tags / importance / access_count / last_accessed_at /
 * fundamental_failure_streak / version / manually_edited）。
 *
 * 从 RulesGrid cell 点击进入；不带筛选时显示全部 rules。
 */

import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate, useSearchParams } from 'react-router';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/cn';
import { formatCurrency, formatDateTime } from '@/lib/format';

import { useMemoryRules, type RuleItem } from './queries';

const AGENTS = ['', 'tech', 'chain', 'news', 'macro'] as const;
const MATURITIES = ['', 'observed', 'probationary', 'active', 'deprecated', 'archived'] as const;

const maturityColor: Record<string, string> = {
  observed: 'bg-muted text-muted-foreground',
  probationary: 'bg-blue-500/15 text-blue-400',
  active: 'bg-trade-long/15 text-trade-long',
  deprecated: 'bg-amber-500/15 text-amber-400',
  archived: 'bg-muted/40 text-muted-foreground line-through',
};

const winRate = (rule: RuleItem): number | null => {
  const total = rule.pnl_track.successes + rule.pnl_track.losses;
  if (total === 0) return null;
  return rule.pnl_track.successes / total;
};

const Filter = ({
  label,
  value,
  options,
  onChange,
  labelFor,
}: {
  label: string;
  value: string;
  options: readonly string[];
  onChange: (v: string) => void;
  labelFor: (opt: string) => string;
}) => (
  <label className="flex items-center gap-2 text-xs">
    <span className="text-muted-foreground">{label}</span>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-border bg-background px-2 py-1 text-xs"
    >
      {options.map((opt) => (
        <option key={opt || '__all__'} value={opt}>
          {opt ? labelFor(opt) : '全部'}
        </option>
      ))}
    </select>
  </label>
);

const RulesDetailContent = () => {
  const { t } = useTranslation('memory');
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();

  const agent = params.get('agent') ?? '';
  const status = params.get('status') ?? '';

  const { data, isLoading } = useMemoryRules({
    ...(agent ? { agent } : {}),
    ...(status ? { status } : {}),
  });

  const rules = useMemo(() => data?.items ?? [], [data]);

  const setParam = (key: 'agent' | 'status', value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  };

  const subtitleParts: string[] = [];
  if (agent) subtitleParts.push(`agent: ${agent}`);
  if (status)
    subtitleParts.push(`${t('rules_grid.agent', { defaultValue: '' })}${t(`maturity.${status}`, { defaultValue: status })}`);
  const subtitle = subtitleParts.length
    ? subtitleParts.join(' · ') + ` · ${rules.length} 条`
    : `全部 ${rules.length} 条`;

  return (
    <div className="space-y-6">
      <PageHeader
        title={t('rules_detail.title', { defaultValue: '规则详情' })}
        subtitle={subtitle}
        onBack={() => void navigate('/memory')}
        actions={
          <div className="flex flex-wrap gap-3">
            <Filter
              label="Agent"
              value={agent}
              options={AGENTS}
              onChange={(v) => setParam('agent', v)}
              labelFor={(v) => v}
            />
            <Filter
              label="状态"
              value={status}
              options={MATURITIES}
              onChange={(v) => setParam('status', v)}
              labelFor={(v) => t(`maturity.${v}`, { defaultValue: v }) as string}
            />
          </div>
        }
      />

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">
            {t('rules_detail.list_title', { defaultValue: '规则列表' })}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : rules.length === 0 ? (
            <EmptyState
              size="compact"
              title={t('rules_detail.empty', { defaultValue: '没有匹配的规则' })}
              description={t('rules_detail.empty_hint', {
                defaultValue: '尝试切换 agent / 状态过滤条件',
              })}
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs">
                <thead>
                  <tr className="border-b border-border text-left text-[10px] uppercase tracking-wider text-muted-foreground">
                    <th className="px-3 py-2">名称</th>
                    <th className="px-3 py-2">Agent</th>
                    <th className="px-3 py-2">状态</th>
                    <th className="px-3 py-2 text-right">胜/负</th>
                    <th className="px-3 py-2 text-right">胜率</th>
                    <th className="px-3 py-2 text-right">累计 PnL</th>
                    <th className="px-3 py-2 text-right">重要性</th>
                    <th className="px-3 py-2 text-right">访问</th>
                    <th className="px-3 py-2 text-right">连失</th>
                    <th className="px-3 py-2 text-right">版本</th>
                    <th className="px-3 py-2">市场 regime</th>
                    <th className="px-3 py-2">最近访问</th>
                  </tr>
                </thead>
                <tbody>
                  {rules.map((r) => {
                    const wr = winRate(r);
                    return (
                      <tr
                        key={`${r.agent}-${r.name}`}
                        className="cursor-pointer border-b border-border/50 align-top hover:bg-muted/30"
                        onClick={() =>
                          void navigate(
                            `/memory/patterns/${encodeURIComponent(r.agent)}/${encodeURIComponent(r.name)}`,
                          )
                        }
                      >
                        <td className="px-3 py-2">
                          <div
                            className="font-mono text-foreground underline-offset-2 hover:underline"
                            title={r.description}
                          >
                            {r.name}
                            {r.manually_edited ? (
                              <Badge variant="outline" className="ml-2 text-[10px]">
                                manual
                              </Badge>
                            ) : null}
                          </div>
                          <div className="mt-0.5 max-w-md truncate text-[10px] text-muted-foreground">
                            {r.description}
                          </div>
                        </td>
                        <td className="px-3 py-2 font-mono text-muted-foreground">{r.agent}</td>
                        <td className="px-3 py-2">
                          <span
                            className={cn(
                              'rounded-full px-1.5 py-0.5 text-[10px] font-semibold',
                              maturityColor[r.maturity] ?? '',
                            )}
                          >
                            {t(`maturity.${r.maturity}`, { defaultValue: r.maturity })}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          <span className="text-trade-long">{r.pnl_track.successes}</span>
                          {' / '}
                          <span className="text-trade-short">{r.pnl_track.losses}</span>
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {wr === null ? (
                            <span className="text-muted-foreground">—</span>
                          ) : (
                            <span
                              className={cn(
                                wr >= 0.5 ? 'text-trade-long' : 'text-trade-short',
                              )}
                            >
                              {(wr * 100).toFixed(1)}%
                            </span>
                          )}
                        </td>
                        <td
                          className={cn(
                            'px-3 py-2 text-right tabular-nums',
                            r.pnl_track.total_pnl > 0
                              ? 'text-trade-long'
                              : r.pnl_track.total_pnl < 0
                                ? 'text-trade-short'
                                : 'text-muted-foreground',
                          )}
                        >
                          {r.pnl_track.total_pnl === 0
                            ? '—'
                            : formatCurrency(r.pnl_track.total_pnl)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                          {r.importance.toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                          {r.access_count}
                        </td>
                        <td
                          className={cn(
                            'px-3 py-2 text-right tabular-nums',
                            r.fundamental_failure_streak > 0
                              ? 'text-trade-short'
                              : 'text-muted-foreground',
                          )}
                        >
                          {r.fundamental_failure_streak}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                          v{r.version}
                        </td>
                        <td className="px-3 py-2">
                          {r.regime_tags.length === 0 ? (
                            <span className="text-muted-foreground">—</span>
                          ) : (
                            <div className="flex flex-wrap gap-1">
                              {r.regime_tags.map((tag) => (
                                <Badge
                                  key={tag}
                                  variant="outline"
                                  className="text-[10px] px-1.5 py-0"
                                >
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </td>
                        <td className="px-3 py-2 text-[10px] text-muted-foreground">
                          {r.last_accessed_at ? formatDateTime(r.last_accessed_at) : '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

const RulesDetailPage = () => (
  <PageBoundary>
    <RulesDetailContent />
  </PageBoundary>
);

export default RulesDetailPage;
