import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useNavigate } from 'react-router';
import { AttentionList } from '@/components/alerts/attention-list';
import { RunDialog } from '@/components/trading/run-dialog';
import { Button } from '@/components/ui/button';
import { PageHeader } from '@/components/ui/page-header';
import { EmptyState } from '@/components/ui/empty-state';
import { useAccounts } from '@/hooks/use-accounts';
import { useDecisions, useStartAnalysis } from '@/hooks/use-decisions';
import { useRuntimeStatus } from '@/hooks/use-runtime-status';
import { useSchedulerStatus } from '@/hooks/use-scheduler-status';
import { formatCycleStatus } from '@/lib/cycle-status';
import { formatDateTime } from '@/lib/format';
import { MoneyValues } from '@/pages/accounts/money';
import { ReadinessNextStep } from './first-run';

const modeLabel = { analysis: '仅分析', trading: '交易运行', backtest: '历史回测' } as const;

export default function WorkbenchPage() {
  const { t } = useTranslation('cycles');
  const readiness = useRuntimeStatus();
  const scheduler = useSchedulerStatus();
  const decisions = useDecisions(1);
  const accounts = useAccounts();
  const startAnalysis = useStartAnalysis();
  const navigate = useNavigate();
  const [pairDraft, setPairDraft] = useState<string | null>(null);
  const [tradingOpen, setTradingOpen] = useState(false);
  const pair = pairDraft ?? readiness.data?.execution_pairs[0] ?? '';
  const reasons = readiness.data
    ? readiness.data.analysis.ready
      ? readiness.data.trading.reasons
      : readiness.data.analysis.reasons
    : [];
  const runningDecision = decisions.data?.items.some((item) => ['queued', 'running'].includes(item.status));

  return (
    <div className="workbench-page text-sm">
      <PageHeader
        title="工作台"
        subtitle="查看运行状态、待处理事项和最近决策。"
        actions={readiness.data ? <span className="status-line">配置版本 R{readiness.data.saved_revision}</span> : null}
      />

      <section className="operational-panel" aria-labelledby="runtime-title">
        <div className="operational-panel-heading">
          <div>
            <h2 id="runtime-title" className="text-base font-semibold">
              运行状态与操作
            </h2>
            {readiness.isPending ? <p role="status">正在读取后端就绪状态…</p> : null}
            {readiness.isError ? (
              <p role="alert" className="configuration-error">
                就绪状态读取失败。不会沿用旧状态。
              </p>
            ) : null}
          </div>
          {readiness.data ? (
            <div className="flex flex-wrap gap-2" aria-label="就绪摘要">
              <span className={readiness.data.analysis.ready ? 'status-ready' : 'status-muted'}>
                分析{readiness.data.analysis.ready ? '已就绪' : '未就绪'}
              </span>
              <span className={readiness.data.trading.ready ? 'status-ready' : 'status-muted'}>
                交易{readiness.data.trading.ready ? '已就绪' : '未就绪'}
              </span>
            </div>
          ) : null}
        </div>
        {readiness.data ? (
          <div className="runtime-facts">
            <p>{readiness.data.automation_enabled ? '自动运行已开启' : '自动运行已暂停'}</p>
            <p>
              定时来源：
              {scheduler.isPending
                ? '读取中'
                : scheduler.data?.enabled
                  ? '已启用'
                  : scheduler.isError
                    ? '状态未知'
                    : '未启用'}
            </p>
            <p>{runningDecision ? '有决策任务正在运行' : '当前没有已知运行中任务'}</p>
            <p>最近运行：{readiness.data.latest_run_at ? formatDateTime(readiness.data.latest_run_at) : '尚无记录'}</p>
          </div>
        ) : null}
        <form
          className="workbench-actions"
          onSubmit={(event) => {
            event.preventDefault();
            if (!readiness.data?.analysis.ready || !pair.trim() || startAnalysis.isPending) return;
            startAnalysis.mutate(
              { pair: pair.trim(), expected_revision: readiness.data.saved_revision },
              { onSuccess: ({ decision_id }) => void navigate(`/decisions/${decision_id}`) },
            );
          }}
        >
          <label className="configuration-field min-w-[15rem] flex-1" htmlFor="workbench-pair">
            <span>交易对</span>
            <input
              id="workbench-pair"
              name="workbench-pair"
              className="configuration-control"
              value={pair}
              placeholder="例如 BTC/USDT:USDT…"
              autoComplete="off"
              spellCheck={false}
              onChange={(event) => setPairDraft(event.target.value)}
            />
          </label>
          <Button type="submit" disabled={!readiness.data?.analysis.ready || !pair.trim() || startAnalysis.isPending}>
            {startAnalysis.isPending ? '正在提交…' : '仅分析一次'}
          </Button>
          <Button variant="outline" disabled={!readiness.data?.trading.ready} onClick={() => setTradingOpen(true)}>
            运行一次交易
          </Button>
        </form>
        {startAnalysis.isError ? <p role="alert">分析未能启动，请核对后端就绪原因后重试。</p> : null}
        <p className="text-muted-foreground">读取状态、查看历史和保存配置都不会发起分析或交易。</p>
      </section>

      <ReadinessNextStep reasons={reasons} />

      <AttentionList />

      <section aria-labelledby="recent-title">
        <div className="section-heading-row">
          <h2 id="recent-title" className="section-title">
            最近决策
          </h2>
          <Link className="text-primary whitespace-nowrap" to="/decisions">
            查看全部
          </Link>
        </div>
        {decisions.isPending ? <p role="status">正在读取决策…</p> : null}
        {decisions.isError ? <p role="alert">决策记录读取失败。</p> : null}
        {decisions.data?.items.length === 0 ? (
          <EmptyState
            size="compact"
            title="还没有决策记录。"
            description="准备好引擎配置后，可在上方仅分析一次；结果与组件依据会保存在这里。"
          />
        ) : null}
        {decisions.data?.items.slice(0, 5).map((decision) => (
          <Link className="decision-row" key={decision.decision_id} to={`/decisions/${decision.decision_id}`}>
            <span className="font-medium">{decision.pair ?? '交易对未知'}</span>
            <span>{modeLabel[decision.mode]}</span>
            <span>R{decision.config_revision}</span>
            <time dateTime={decision.created_at}>{formatDateTime(decision.created_at)}</time>
            <span>{formatCycleStatus(t, decision.status)}</span>
          </Link>
        ))}
      </section>

      <section aria-labelledby="accounts-title">
        <div className="section-heading-row">
          <h2 id="accounts-title" className="section-title">
            账户概况
          </h2>
          <Link className="text-primary whitespace-nowrap" to="/accounts">
            查看账户
          </Link>
        </div>
        {accounts.isPending ? <p role="status">正在读取账户事实…</p> : null}
        {accounts.isError ? <p role="alert">账户事实读取失败，不显示过期金额。</p> : null}
        {accounts.data ? (
          <div className="account-overview-grid">
            {(['simulated', 'real'] as const).map((scope) => {
              const scoped = accounts.data.items.filter((item) => item.capital_scope === scope);
              const unassigned = scoped.filter((item) => item.book_ids.length === 0).length;
              return (
                <article key={scope} className="account-overview-row">
                  <div>
                    <h3 className="font-semibold">{scope === 'simulated' ? '模拟账户' : '真实账户'}</h3>
                    <p className="text-muted-foreground">
                      {scoped.length ? `${scoped.length} 个连接` : '尚未添加账户'}
                    </p>
                  </div>
                  <div>
                    <MoneyValues values={accounts.data[scope]} />
                    {unassigned ? <p className="mt-1 text-amber-600">尚未分配到资金池：{unassigned} 个</p> : null}
                  </div>
                </article>
              );
            })}
          </div>
        ) : null}
      </section>
      {tradingOpen ? (
        <RunDialog pairs={readiness.data?.execution_pairs ?? []} onClose={() => setTradingOpen(false)} />
      ) : null}
    </div>
  );
}
