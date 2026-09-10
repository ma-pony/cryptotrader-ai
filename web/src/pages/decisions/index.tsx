import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';

import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { Button } from '@/components/ui/button';
import { EmptyState } from '@/components/ui/empty-state';
import { useDecisions } from '@/hooks/use-decisions';
import { cycleStatusTone, formatCycleStatus } from '@/lib/cycle-status';
import { formatDateTime } from '@/lib/format';

const DecisionsContent = () => {
  const { t } = useTranslation(['decisions', 'cycles']);
  const [page, setPage] = useState(1);

  const cycles = useDecisions(page);
  const items = cycles.data?.items ?? [];

  return (
    <div className="space-y-6">
      <PageHeader title={t('title')} subtitle="按时间查看组件观点、融合依据与执行结果，内部辩论保留在决策详情中。" />
      {cycles.isLoading ? <p role="status">{t('loading', { ns: 'cycles' })}</p> : null}
      {cycles.isError ? (
        <p role="alert" className="text-destructive">
          {t('loadError', { ns: 'cycles' })}
        </p>
      ) : null}
      {!cycles.isLoading && !cycles.isError && items.length === 0 ? (
        <EmptyState
          title={t('empty', { ns: 'cycles' })}
          description="开始一次分析后，观点与依据会记录在这里。"
          action={
            <Button asChild variant="outline">
              <Link to="/research/analysis">前往仅分析</Link>
            </Button>
          }
        />
      ) : null}
      {items.length > 0 ? (
        <>
          {items.map((cycle) => (
            <Link key={cycle.decision_id} to={`/decisions/${cycle.decision_id}`} className="decision-list-row">
              <b>{cycle.pair ?? '交易对未知'}</b>
              <span>
                {cycle.mode === 'analysis' ? '仅分析' : cycle.mode === 'trading' ? '交易运行' : '回测'} · R
                {cycle.config_revision}
              </span>
              <time className="text-sm text-muted-foreground" dateTime={cycle.created_at}>
                {formatDateTime(cycle.created_at)}
              </time>
              <span className="text-sm text-muted-foreground">
                {cycle.books
                  .map((book) => `${book.book_id}（${book.capital_scope === 'simulated' ? '模拟' : '真实'}）`)
                  .join('、')}
              </span>
              <span
                className={`text-sm ${cycleStatusTone(cycle.status) === 'danger' ? 'text-destructive' : cycleStatusTone(cycle.status) === 'warning' ? 'text-amber-600' : 'text-muted-foreground'}`}
              >
                {formatCycleStatus(t, cycle.status)}
              </span>
              {cycle.books.some((book) => book.execution?.requires_attention) ? (
                <span className="ml-2 text-amber-600">{t('attention', { ns: 'cycles' })}</span>
              ) : null}
            </Link>
          ))}
          <nav className="flex items-center justify-between" aria-label={t('title')}>
            <Button
              variant="outline"
              onClick={() => setPage((current) => Math.max(1, current - 1))}
              disabled={page === 1}
            >
              {t('previous', { ns: 'cycles' })}
            </Button>
            <span>{t('page', { ns: 'cycles', page })}</span>
            <Button
              variant="outline"
              onClick={() => setPage((current) => current + 1)}
              disabled={!cycles.data?.has_next}
            >
              {t('next', { ns: 'cycles' })}
            </Button>
          </nav>
        </>
      ) : null}
    </div>
  );
};

const DecisionsPage = () => (
  <PageBoundary>
    <DecisionsContent />
  </PageBoundary>
);

export default DecisionsPage;
