import { AlertTriangle, Clock, ShieldAlert } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { StatusPill } from '@/components/ui/status-pill';
import { useRiskStatus } from '@/hooks/use-risk-status';
import { formatDateTime } from '@/lib/format';

import { ApprovalQueueCard } from './components/approval-queue-card';
import { CircuitBreakerHero } from './components/circuit-breaker-hero';
import { RiskMeter } from './components/risk-meter';
import { ThresholdsCard } from './components/thresholds-card';

const RiskContent = () => {
  const { t } = useTranslation('risk');
  const { data } = useRiskStatus();

  // RiskPage's PageBoundary already handles loading + error; if data is still
  // null here it's a transient gap (data refetching in the background) — render
  // nothing rather than flash a skeleton.
  if (!data) return null;

  const thresholds = data.thresholds;

  return (
    <div className="space-y-8">
      <PageHeader title={t('title')} />

      {!data.redis_available ? (
        <div className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-500">
          <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden />
          {t('redis_warning')}
        </div>
      ) : null}

      {/* ── Section 1: Live status — what's happening right now ── */}
      <section aria-labelledby="risk-section-live" className="space-y-4">
        <SectionHeader
          id="risk-section-live"
          label={t('section.live', { defaultValue: '实时状态' })}
          description={t('section.live_hint', {
            defaultValue: '熔断器、关键风险指标',
          })}
        />

        <CircuitBreakerHero cb={data.circuit_breaker} redisAvailable={data.redis_available} />

        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <RiskMeter
            label={t('meters.daily_loss', { defaultValue: '当日亏损' })}
            value={data.daily_loss_pct ?? null}
            unit="%"
          />
          <RiskMeter
            label={t('meters.drawdown', { defaultValue: '当前回撤' })}
            value={data.drawdown_pct ?? null}
            unit="%"
          />
          <RiskMeter
            label={t('meters.exposure', { defaultValue: '总敞口' })}
            value={data.total_exposure_pct ?? null}
            unit="%"
            precision={0}
          />
          <RiskMeter
            label={t('meters.cvar', { defaultValue: '95% CVaR' })}
            value={data.cvar_95 ?? null}
            unit="%"
            precision={2}
          />
        </div>
      </section>

      {/* ── Section 2: Limits / config — boundaries the engine enforces ── */}
      <section aria-labelledby="risk-section-limits" className="space-y-4">
        <SectionHeader
          id="risk-section-limits"
          label={t('section.limits', { defaultValue: '观察与配置' })}
          description={t('section.limits_hint', {
            defaultValue: '统计仅供观察，实际限制见下方配置',
          })}
        />

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Card>
            <CardHeader className="p-4 pb-2">
              <CardTitle className="text-sm">{t('corr.title', { defaultValue: '相关性分组' })}</CardTitle>
              <div className="text-[11px] text-muted-foreground">{t('reporting_only')}</div>
            </CardHeader>
            <CardContent className="flex flex-col gap-2.5 p-4 pt-0">
              {data.correlation_groups.length === 0 ? (
                <EmptyState size="compact" title={t('corr.empty', { defaultValue: '暂无相关性数据' })} />
              ) : (
                data.correlation_groups.map((g) => (
                  <div key={g.name} className="flex items-center gap-2.5">
                    <div className="w-28 text-xs font-medium">{g.name}</div>
                    <div className="flex-1 text-right font-mono text-xs">{g.open}</div>
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="p-4 pb-2">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Clock size={14} />
                {t('cooldown.title', { defaultValue: '交易计数与冷却记录' })}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-3 p-4 pt-0">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-md bg-muted p-2.5">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    {t('rate.hourly', { defaultValue: '本小时' })}
                  </div>
                  <div className="font-mono text-base tabular-nums">{data.trade_count_hour ?? '—'}</div>
                </div>
                <div className="rounded-md bg-muted p-2.5">
                  <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    {t('rate.daily', { defaultValue: '今日' })}
                  </div>
                  <div className="font-mono text-base tabular-nums">{data.trade_count_day ?? '—'}</div>
                </div>
              </div>
              {data.cooldowns.length === 0 ? (
                <EmptyState size="compact" title={t('cooldown.empty', { defaultValue: '暂无冷却记录' })} />
              ) : (
                <div className="space-y-1.5">
                  {data.cooldowns.map((c) => (
                    <div key={c.pair} className="flex items-center gap-2 text-xs">
                      <span className="font-mono font-medium w-24">{c.pair}</span>
                      {c.until_seconds === 0 ? (
                        <span className="text-muted-foreground">{t('cooldown.clear')}</span>
                      ) : (
                        <>
                          <StatusPill tone="warning">冷却中</StatusPill>
                          <span className="font-mono text-amber-500">
                            {Math.floor(c.until_seconds / 60)}m {c.until_seconds % 60}s
                          </span>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <ThresholdsCard thresholds={thresholds} />
      </section>

      {/* ── Section 3: Event stream — recent blocks & pending approvals ── */}
      <section aria-labelledby="risk-section-events" className="space-y-4">
        <SectionHeader
          id="risk-section-events"
          label={t('section.events', { defaultValue: '事件流' })}
          description={t('section.events_hint', {
            defaultValue: '最近拦截 · 待审批',
          })}
        />

        <Card>
          <CardHeader className="p-4 pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <ShieldAlert size={14} />
              {t('blocks.title', { defaultValue: '最近风控拦截' })}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-0">
            {data.recent_blocks.length === 0 ? (
              <EmptyState
                size="compact"
                icon={<ShieldAlert className="h-5 w-5" />}
                title={t('blocks.empty', { defaultValue: '近期无拦截记录' })}
                description={t('blocks.empty_hint', { defaultValue: '此处仅展示已记录的风控拦截' })}
              />
            ) : (
              <div className="divide-y divide-border">
                {data.recent_blocks.map((b) => (
                  <div key={b.cycle_id} className="flex items-center gap-3 py-2.5 text-xs">
                    <span className="w-20 font-mono text-muted-foreground">{formatDateTime(b.ts).slice(-8)}</span>
                    <span className="w-20 font-mono text-muted-foreground">{b.cycle_id.slice(0, 8)}</span>
                    <StatusPill tone="danger">{b.rule}</StatusPill>
                    <span className="flex-1 truncate text-muted-foreground">{b.detail}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <ApprovalQueueCard />
      </section>
    </div>
  );
};

interface SectionHeaderProps {
  id: string;
  label: string;
  description?: string;
}

/** Subtle section divider for grouping risk page content. */
const SectionHeader = ({ id, label, description }: SectionHeaderProps) => (
  <div className="flex items-baseline gap-3 border-b border-border pb-1.5">
    <h2 id={id} className="text-[11px] font-semibold uppercase tracking-wider text-foreground">
      {label}
    </h2>
    {description ? <span className="text-[11px] text-muted-foreground">{description}</span> : null}
  </div>
);

const RiskPage = () => {
  const { t } = useTranslation('risk');
  const { isLoading, isError, refetch } = useRiskStatus();
  return (
    <PageBoundary loading={isLoading} isError={isError} onRetry={() => void refetch()} errorTitle={t('title')}>
      <RiskContent />
    </PageBoundary>
  );
};

export default RiskPage;
