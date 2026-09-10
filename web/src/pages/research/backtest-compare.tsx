import { Link, useSearchParams } from 'react-router';
import { Button } from '@/components/ui/button';
import { PageHeader } from '@/components/ui/page-header';
import { useBacktestComparison } from '@/hooks/use-backtest';
import { ReadableValues, ResearchNav, RunMetrics, fieldLabel } from './presentation';

export default function BacktestCompare() {
  const [params] = useSearchParams();
  const query = useBacktestComparison(params.get('left'), params.get('right'));
  if (!params.get('left') || !params.get('right'))
    return (
      <p>
        请在
        <Link to="/research" className="text-primary">
          研究历史
        </Link>
        选择两场回测。
      </p>
    );
  if (query.isPending) return <p role="status">正在读取两场实验…</p>;
  if (query.isError)
    return (
      <p role="alert">
        比较记录读取失败。
        <Button className="min-h-10" onClick={() => void query.refetch()}>
          重新读取
        </Button>
      </p>
    );
  const comparison = query.data;
  const differences = { ...comparison.condition_differences, ...comparison.configuration_differences };
  return (
    <div className="space-y-6 text-sm">
      <ResearchNav />
      <PageHeader title="两次回测比较" />
      <section className="space-y-3 rounded border border-border bg-card p-4">
        <h2 className="font-semibold">
          {comparison.comparable ? '基础条件一致，仍需核对配置与证据' : '实验条件不同，不作排名'}
        </h2>
        <p>结果只描述这两次实验。模型、提示词、数据覆盖和缺失上下文会影响解释，不自动选择最佳配置。</p>
        {Object.entries(differences).map(([key, value]) => (
          <article key={key} className="border-t border-border pt-3">
            <h3 className="mb-2 font-medium">{fieldLabel(key)}</h3>
            {'reason' in value ? <p>{value.reason}</p> : null}
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <p className="mb-1 text-muted-foreground">左侧</p>
                <ReadableValues value={value.left} />
              </div>
              <div>
                <p className="mb-1 text-muted-foreground">右侧</p>
                <ReadableValues value={value.right} />
              </div>
            </div>
          </article>
        ))}
        {!Object.keys(differences).length ? (
          <p>已保存的实验条件和配置未见差异；未知模型身份仍不代表相同模型。</p>
        ) : null}
      </section>
      <div className="grid gap-6 xl:grid-cols-2">
        {[comparison.left, comparison.right].map((run, index) => (
          <section className="space-y-3" key={`${index}-${run.run_id}`}>
            <h2 className="font-semibold">
              {index === 0 ? '左侧' : '右侧'} · {run.params.name ?? run.params.pair}
            </h2>
            <Link className="inline-flex min-h-10 items-center text-primary" to={`/research/backtests/${run.run_id}`}>
              查看详情 {run.run_id}
            </Link>
            <RunMetrics run={run} />
            <ReadableValues value={run.params} />
          </section>
        ))}
      </div>
    </div>
  );
}
