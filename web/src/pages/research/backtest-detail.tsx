import { Link, useParams } from 'react-router';
import { EquityChart } from '@/components/charts/equity-chart';
import { Button } from '@/components/ui/button';
import { PageHeader } from '@/components/ui/page-header';
import { isActiveBacktest, useBacktestRun, useCancelBacktest } from '@/hooks/use-backtest';
import { ReadableValues, ResearchNav, RunMetrics, SnapshotSummary, runLabels } from './presentation';

export default function BacktestDetail() {
  const { runId } = useParams<{ runId: string }>();
  const query = useBacktestRun(runId);
  const cancel = useCancelBacktest();
  if (query.isPending) return <p role="status">正在读取回测记录…</p>;
  if (query.isError)
    return (
      <div role="alert">
        回测记录读取失败或不存在。
        <Button className="min-h-10" onClick={() => void query.refetch()}>
          重新读取
        </Button>
        <Link to="/research">返回研究</Link>
      </div>
    );
  const run = query.data;
  const result = run.result;
  return (
    <div className="space-y-6 text-sm">
      <ResearchNav />
      <PageHeader
        title={run.params.name ?? `${run.params.pair} 回测详情`}
        subtitle={
          <>
            <p>
              <span>{runLabels[run.status]}</span> · 进度 {Math.round(run.progress * 100)}%
            </p>
            <p className="text-muted-foreground">运行记录 {run.run_id} · 刷新或关闭页面不会丢失已保存结果。</p>
          </>
        }
      />
      <div className="flex flex-wrap items-center gap-4">
        {run.config_snapshot !== null ? (
          <Link className="configuration-button" to={`/research?reuse=${encodeURIComponent(run.run_id)}`}>
            复用此配置
          </Link>
        ) : (
          <button className="configuration-button" disabled>
            配置不完整，无法复用
          </button>
        )}
        <Link to="/research" className="inline-flex min-h-11 items-center text-primary">
          选择另一场回测比较
        </Link>
        {isActiveBacktest(run.status) ? (
          <Button className="min-h-10" disabled={cancel.isPending} onClick={() => cancel.mutate(run.run_id)}>
            取消回测
          </Button>
        ) : null}
      </div>
      {cancel.isError ? <p role="alert">取消失败，请重新读取运行状态后重试。</p> : null}
      {run.error ? <p role="alert">{run.error}</p> : null}
      {run.status === 'interrupted' ? <p>服务重启后不会自动续跑。可复用完整快照，明确开始一场新实验。</p> : null}
      {run.incomplete_fields.length ? (
        <section aria-label="缺失证据">
          <h2 className="font-semibold">这份历史记录缺少证据</h2>
          <ReadableValues value={run.incomplete_fields} />
        </section>
      ) : null}
      <section aria-label="实验条件">
        <h2 className="mb-2 font-semibold">实验条件</h2>
        <ReadableValues value={run.params} />
      </section>
      <RunMetrics run={run} />
      {result ? (
        <>
          <section aria-label="历史权益曲线">
            <h2 className="font-semibold">历史权益曲线</h2>
            {result.equity_curve.length ? (
              <>
                <p className="my-2 tabular-nums">
                  {result.equity_curve[0]!.ts} · {result.equity_curve[0]!.equity} → {result.equity_curve.at(-1)!.ts} ·{' '}
                  {result.equity_curve.at(-1)!.equity}
                </p>
                <EquityChart data={result.equity_curve} height={300} />
              </>
            ) : (
              <p>没有可用的历史权益曲线；不使用当前行情重建。</p>
            )}
          </section>
          <section className="space-y-2">
            <h2 className="font-semibold">成交与成本</h2>
            <p>
              手续费 {run.params.fee_rate === null ? '未知' : result.fees} · 资金费{' '}
              {run.params.funding_assumption === null ? '未知' : result.funding}
            </p>
            {result.fills.length ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr>
                      {['历史时间', '方向', '数量', '成交价', '费用', '订单'].map((label) => (
                        <th className="p-2" key={label}>
                          {label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.fills.map((fill) => (
                      <tr key={fill.venue_fill_id} className="border-t border-border">
                        <td className="p-2">{fill.occurred_at}</td>
                        <td>{fill.side === 'buy' ? '买入' : '卖出'}</td>
                        <td>{fill.amount}</td>
                        <td>{fill.price}</td>
                        <td>{fill.fee.amount ?? '未知'}</td>
                        <td>{fill.venue_order_id}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p>暂无成交流水。</p>
            )}
            <details>
              <summary className="min-h-10 cursor-pointer">成本假设与未建模成本</summary>
              <ReadableValues value={result.cost_assumptions} />
              <ReadableValues value={result.unmodeled_costs} />
            </details>
          </section>
          <section>
            <h2 className="mb-2 font-semibold">数据覆盖与缺失上下文</h2>
            <ReadableValues value={result.data_coverage} />
          </section>
          <section className="space-y-2">
            <h2 className="font-semibold">关联决策</h2>
            {result.decision_ids.length ? (
              <ul>
                {result.decision_ids.map((id) => (
                  <li key={id}>
                    <Link
                      className="inline-flex min-h-10 items-center text-primary"
                      to={`/decisions/${encodeURIComponent(id)}`}
                    >
                      查看原决策 {id}
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <p>尚无关联决策。</p>
            )}
          </section>
        </>
      ) : null}
      <section className="space-y-2">
        <h2 className="font-semibold">模型请求与提示词证据</h2>
        {run.model_evidence.length ? (
          run.model_evidence.map((item, index) => (
            <div key={index} className="rounded border border-border p-3">
              <p>
                请求模型：{item.requested_model ?? '未知'} ·{' '}
                {item.actual_model ? `实际模型：${item.actual_model}` : '实际模型未知：响应未提供身份'}
              </p>
              <p className="break-all">提示词 SHA256：{item.prompt_hash}</p>
              <p>
                安全指纹版本：{item.prompt_version} ·{' '}
                {item.status === 'completed' ? '请求完成' : item.status === 'failed' ? '请求失败' : '请求未完成'}
              </p>
            </div>
          ))
        ) : (
          <p>没有已捕获的模型请求证据；实际模型和提示词哈希未知。不会用当前配置回填。</p>
        )}
      </section>

      <SnapshotSummary run={run} />
    </div>
  );
}
