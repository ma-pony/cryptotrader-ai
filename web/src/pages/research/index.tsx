import { useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router';
import { Button } from '@/components/ui/button';
import { PageHeader } from '@/components/ui/page-header';
import { useBacktestRun, useBacktestRuns } from '@/hooks/use-backtest';
import { BacktestForm } from './backtest-form';
import { ResearchNav, runLabels } from './presentation';

export default function ResearchPage() {
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<string[]>([]);
  const [params] = useSearchParams();
  const reuseId = params.get('reuse') ?? undefined;
  const reuse = useBacktestRun(reuseId);
  const history = useBacktestRuns(offset);
  const navigate = useNavigate();
  return (
    <div className="space-y-6 text-sm">
      <ResearchNav />
      <PageHeader title="研究与回测" subtitle="先明确条件，再看结果。运行与快照自动保存，可随时回来核对。" />
      {reuseId && reuse.isPending ? (
        <p role="status">正在读取待复用快照…</p>
      ) : reuseId && reuse.isError ? (
        <p role="alert">
          历史快照读取失败。
          <Link to="/research" className="text-primary">
            改用当前配置
          </Link>
        </p>
      ) : (
        <BacktestForm
          key={reuseId ?? 'current'}
          initialRun={reuse.data}
          onRunStarted={(id) => {
            void navigate(`/research/backtests/${id}`);
          }}
        />
      )}
      <section className="space-y-3" aria-label="回测历史">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-semibold">回测历史</h2>
          <Button
            className="min-h-10"
            disabled={selected.length !== 2}
            onClick={() => {
              void navigate(
                `/research/compare?left=${encodeURIComponent(selected[0]!)}&right=${encodeURIComponent(selected[1]!)}`,
              );
            }}
          >
            比较已选两次
          </Button>
        </div>
        {history.isPending ? (
          <p role="status">正在读取历史…</p>
        ) : history.isError ? (
          <p role="alert">
            历史读取失败。
            <Button className="min-h-10" onClick={() => void history.refetch()}>
              重新读取
            </Button>
          </p>
        ) : !history.data.items.length ? (
          <p>还没有回测记录。填写上方实验条件并明确点击“运行回测”，第一场结果会保存在这里。</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[620px] text-left">
              <thead>
                <tr>
                  {['比较', '实验 / 交易对', '历史区间', '周期', '状态', '详情'].map((title) => (
                    <th className="p-2" key={title}>
                      {title}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.data.items.map((run) => (
                  <tr key={run.run_id} className="border-t border-border">
                    <td className="p-2">
                      <label className="flex min-h-11 items-center">
                        <input
                          type="checkbox"
                          aria-label={`比较 ${run.params.name ?? run.run_id}`}
                          checked={selected.includes(run.run_id)}
                          disabled={!selected.includes(run.run_id) && selected.length >= 2}
                          onChange={(event) =>
                            setSelected((ids) =>
                              event.target.checked ? [...ids, run.run_id] : ids.filter((id) => id !== run.run_id),
                            )
                          }
                        />
                      </label>
                    </td>
                    <td>
                      {run.params.name ?? run.params.pair}
                      <p className="text-muted-foreground">
                        {run.config_snapshot !== null ? `R${run.config_snapshot.revision}` : '旧记录 · 配置未知'}
                      </p>
                    </td>
                    <td>
                      {run.params.start} → {run.params.end}
                    </td>
                    <td>{run.params.interval ?? '未知'}</td>
                    <td>{runLabels[run.status]}</td>
                    <td>
                      <Link
                        className="inline-flex min-h-11 items-center text-primary"
                        to={`/research/backtests/${run.run_id}`}
                      >
                        查看详情
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="flex gap-3">
          <Button
            className="min-h-10"
            variant="outline"
            disabled={offset === 0}
            onClick={() => setOffset((value) => Math.max(0, value - 20))}
          >
            上一页
          </Button>
          <Button
            className="min-h-10"
            variant="outline"
            disabled={!history.data?.has_next}
            onClick={() => setOffset((value) => value + 20)}
          >
            下一页
          </Button>
        </div>
      </section>
    </div>
  );
}
