/**
 * spec 021 — PatternDetailPage
 *
 * /memory/patterns/:agent/:name
 *
 * 展示单条 pattern 的完整链路：
 *   - 头部：maturity / pnl_track / regime_tags / 元数据
 *   - body 全文
 *   - source_cycles 列表：每行可展开 CaseChainCard，显示完整 5-维链条
 *     (agent_analyses / verdict / risk_gate / execution / final_pnl)
 */

import { useTranslation } from 'react-i18next';
import { useNavigate, useParams } from 'react-router';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/cn';
import { formatCurrency, formatDateTime } from '@/lib/format';

import { CaseChainCard } from './components/CaseChainCard';
import { usePatternDetail } from './queries';

const maturityColor: Record<string, string> = {
  observed: 'bg-muted text-muted-foreground',
  probationary: 'bg-blue-500/15 text-blue-400',
  active: 'bg-trade-long/15 text-trade-long',
  deprecated: 'bg-amber-500/15 text-amber-400',
  archived: 'bg-muted/40 text-muted-foreground line-through',
};

const StatRow = ({ label, value }: { label: string; value: React.ReactNode }) => (
  <div className="flex items-center justify-between gap-4 border-b border-border/40 py-1.5 last:border-0">
    <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</span>
    <span className="text-xs tabular-nums text-foreground">{value}</span>
  </div>
);

const PatternDetailContent = () => {
  const { t } = useTranslation('memory');
  const navigate = useNavigate();
  const { agent = '', name = '' } = useParams<{ agent: string; name: string }>();
  const { data, isLoading, isError } = usePatternDetail(agent, name);

  if (isLoading) {
    return (
      <div className="space-y-6">
        <PageHeader title={name} onBack={() => void navigate(-1)} />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-48 w-full" />
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="space-y-6">
        <PageHeader title={name} onBack={() => void navigate(-1)} />
        <EmptyState
          title={t('pattern_detail.not_found', { defaultValue: '规则不存在' })}
          description={`${agent}/${name}`}
        />
      </div>
    );
  }

  const totalCases = data.pnl_track.successes + data.pnl_track.losses;
  const winRate = totalCases > 0 ? data.pnl_track.successes / totalCases : null;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow={`${data.agent} · ${t(`maturity.${data.maturity}`, { defaultValue: data.maturity })}`}
        title={data.name}
        subtitle={data.description}
        onBack={() => void navigate(-1)}
        actions={
          <span
            className={cn(
              'rounded-full px-2 py-1 text-[11px] font-semibold',
              maturityColor[data.maturity] ?? '',
            )}
          >
            {t(`maturity.${data.maturity}`, { defaultValue: data.maturity })}
          </span>
        }
      />

      {/* 摘要 stats */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              {t('pattern_detail.stats', { defaultValue: '统计' })}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-0 text-xs">
            <StatRow
              label="胜 / 负"
              value={
                <>
                  <span className="text-trade-long">{data.pnl_track.successes}</span>
                  {' / '}
                  <span className="text-trade-short">{data.pnl_track.losses}</span>
                </>
              }
            />
            <StatRow
              label="胜率"
              value={
                winRate === null ? (
                  <span className="text-muted-foreground">—（尚无闭环交易）</span>
                ) : (
                  <span className={winRate >= 0.5 ? 'text-trade-long' : 'text-trade-short'}>
                    {(winRate * 100).toFixed(1)}%
                  </span>
                )
              }
            />
            <StatRow
              label="累计 PnL"
              value={
                data.pnl_track.total_pnl === 0 ? (
                  <span className="text-muted-foreground">—</span>
                ) : (
                  <span
                    className={
                      data.pnl_track.total_pnl > 0 ? 'text-trade-long' : 'text-trade-short'
                    }
                  >
                    {formatCurrency(data.pnl_track.total_pnl)}
                  </span>
                )
              }
            />
            <StatRow label="重要性" value={data.importance.toFixed(2)} />
            <StatRow label="访问次数" value={data.access_count} />
            <StatRow
              label="连续根本性失败"
              value={
                <span
                  className={
                    data.fundamental_failure_streak > 0 ? 'text-trade-short' : 'text-muted-foreground'
                  }
                >
                  {data.fundamental_failure_streak}
                </span>
              }
            />
            <StatRow label="版本" value={`v${data.version}`} />
            <StatRow
              label="手动编辑"
              value={data.manually_edited ? '是' : <span className="text-muted-foreground">否</span>}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              {t('pattern_detail.meta', { defaultValue: '元数据' })}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-0 text-xs">
            <StatRow
              label="市场 regime"
              value={
                data.regime_tags.length === 0 ? (
                  <span className="text-muted-foreground">—</span>
                ) : (
                  <span className="flex flex-wrap justify-end gap-1">
                    {data.regime_tags.map((tag) => (
                      <Badge key={tag} variant="outline" className="text-[10px] px-1.5 py-0">
                        {tag}
                      </Badge>
                    ))}
                  </span>
                )
              }
            />
            <StatRow
              label="创建"
              value={data.created ? formatDateTime(data.created) : '—'}
            />
            <StatRow
              label="最近修改"
              value={data.last_modified_at ? formatDateTime(data.last_modified_at) : '—'}
            />
            <StatRow
              label="最近访问"
              value={data.last_accessed_at ? formatDateTime(data.last_accessed_at) : '—'}
            />
            <StatRow
              label="来源 cycle 数"
              value={data.source_cycles.length}
            />
          </CardContent>
        </Card>
      </div>

      {/* body 全文 */}
      {data.body ? (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              {t('pattern_detail.body', { defaultValue: '规则正文' })}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded bg-muted/40 p-3 font-mono text-[11px] text-foreground/90">
              {data.body}
            </pre>
          </CardContent>
        </Card>
      ) : null}

      {/* source cycles 链条 */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">
            {t('pattern_detail.cases_chain', {
              defaultValue: '来源 cycle 完整链条',
            })}
            <span className="ml-2 text-[10px] font-normal text-muted-foreground">
              （点击展开查看 agent 分析 / 决议 / 风控 / 撮合 / final_pnl）
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {data.source_cycles.length === 0 ? (
            <EmptyState
              size="compact"
              title="该 pattern 暂无 source_cycles"
              description="cold-start 阶段未填充或已被清理"
            />
          ) : (
            <ul className="space-y-2">
              {data.source_cycles.map((cycleId) => (
                <li key={cycleId}>
                  <CaseChainCard cycleId={cycleId} />
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

const PatternDetailPage = () => (
  <PageBoundary>
    <PatternDetailContent />
  </PageBoundary>
);

export default PatternDetailPage;
