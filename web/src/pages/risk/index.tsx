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

// Mirrors the 11 checks registered in ``src/cryptotrader/risk/gate.py``.
// TODO(contract): expose ``checks_total`` via /api/risk/status so this stays in sync.
const RISK_CHECK_COUNT = 11;

const RiskContent = () => {
  const { t } = useTranslation('risk');
  const { data } = useRiskStatus();

  // RiskPage's PageBoundary already handles loading + error; if data is still
  // null here it's a transient gap (data refetching in the background) — render
  // nothing rather than flash a skeleton.
  if (!data) return null;

  const thresholds = data.thresholds;

  return (
    <div className="space-y-6">
      <PageHeader title={t('title')} />

      {!data.redis_available ? (
        <div className="flex items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-xs text-amber-500">
          <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden />
          {t('redis_warning')}
        </div>
      ) : null}

      <CircuitBreakerHero
        cb={data.circuit_breaker}
        redisAvailable={data.redis_available}
        // FE-m7: The 11 risk checks are statically registered in src/cryptotrader/risk/gate.py;
        // backend does not expose a count, so we mirror the static value here. Changes to
        // risk gate registration must update this literal (or expose checks_total via API).
        checksOnline={RISK_CHECK_COUNT}
      />

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <RiskMeter
          label={t('meters.daily_loss', { defaultValue: '当日亏损' })}
          value={data.daily_loss_pct ?? null}
          limit={thresholds.max_daily_loss_pct * 100}
          unit="%"
        />
        <RiskMeter
          label={t('meters.drawdown', { defaultValue: '当前回撤' })}
          value={data.drawdown_pct ?? null}
          limit={10}
          unit="%"
        />
        <RiskMeter
          label={t('meters.exposure', { defaultValue: '总敞口' })}
          value={data.total_exposure_pct ?? null}
          limit={100}
          unit="%"
          precision={0}
        />
        <RiskMeter
          label={t('meters.cvar', { defaultValue: '95% CVaR' })}
          value={data.cvar_95 ?? null}
          limit={5}
          unit="%"
          precision={2}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card>
          <CardHeader className="p-4 pb-2">
            <CardTitle className="text-sm">
              {t('corr.title', { defaultValue: '相关性分组' })}
            </CardTitle>
            <div className="text-[11px] text-muted-foreground">
              {t('corr.subtitle', { defaultValue: '每组最多 {{n}} 仓位', n: 2 })}
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-2.5 p-4 pt-0">
            {data.correlation_groups.length === 0 ? (
              <EmptyState
                size="compact"
                title={t('corr.empty', { defaultValue: '暂无相关性数据' })}
              />
            ) : (
              data.correlation_groups.map((g) => (
                <div key={g.name} className="flex items-center gap-2.5">
                  <div className="w-28 text-xs font-medium">{g.name}</div>
                  <div className="flex flex-1 gap-1">
                    {Array.from({ length: g.max }).map((_, i) => (
                      <div
                        key={i}
                        className="h-5 flex-1 rounded border border-border"
                        style={{
                          background: i < g.open ? 'var(--amber-500)' : 'hsl(var(--muted))',
                        }}
                      />
                    ))}
                  </div>
                  <div className="w-12 text-right font-mono text-[11px] text-muted-foreground">
                    {g.open}/{g.max}
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="p-4 pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Clock size={14} />
              {t('cooldown.title', { defaultValue: '冷却 · 频率限制' })}
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3 p-4 pt-0">
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-md bg-muted p-2.5">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  {t('rate.hourly', { defaultValue: '本小时' })}
                </div>
                <div className="font-mono text-base tabular-nums">
                  {data.trade_count_hour ?? 0}
                  <span className="text-xs text-muted-foreground ml-1">
                    / {thresholds.max_trades_per_hour}
                  </span>
                </div>
              </div>
              <div className="rounded-md bg-muted p-2.5">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  {t('rate.daily', { defaultValue: '今日' })}
                </div>
                <div className="font-mono text-base tabular-nums">
                  {data.trade_count_day ?? 0}
                  <span className="text-xs text-muted-foreground ml-1">
                    / {thresholds.max_trades_per_day}
                  </span>
                </div>
              </div>
            </div>
            {data.cooldowns.length === 0 ? (
              <EmptyState
                size="compact"
                title={t('cooldown.empty', { defaultValue: '所有交易对均可交易' })}
              />
            ) : (
              <div className="space-y-1.5">
                {data.cooldowns.map((c) => (
                  <div key={c.pair} className="flex items-center gap-2 text-xs">
                    <span className="font-mono font-medium w-24">{c.pair}</span>
                    {c.until_seconds === 0 ? (
                      <StatusPill tone="success">可交易</StatusPill>
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
              description={t('blocks.empty_hint', { defaultValue: '风控引擎运行正常' })}
            />
          ) : (
            <div className="divide-y divide-border">
              {data.recent_blocks.map((b) => (
                <div
                  key={b.commit_hash}
                  className="flex items-center gap-3 py-2.5 text-xs"
                >
                  <span className="font-mono text-muted-foreground w-20">
                    {formatDateTime(b.ts).slice(-8)}
                  </span>
                  <span className="font-mono text-muted-foreground w-20">
                    {b.commit_hash.slice(0, 8)}
                  </span>
                  <StatusPill tone="danger">{b.rule}</StatusPill>
                  <span className="flex-1 truncate text-muted-foreground">{b.detail}</span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <ApprovalQueueCard />
    </div>
  );
};

const RiskPage = () => {
  const { t } = useTranslation('risk');
  const { isLoading, isError, refetch } = useRiskStatus();
  return (
    <PageBoundary
      loading={isLoading}
      isError={isError}
      onRetry={() => void refetch()}
      errorTitle={t('title')}
    >
      <RiskContent />
    </PageBoundary>
  );
};

export default RiskPage;
