import { useState } from 'react';
import { Link } from 'react-router';
import { ResultBlocks } from '@/components/signals/result-blocks';
import { useComponentEvaluations, type EvaluationFilters } from '@/hooks/use-component-evaluations';
import type { ComponentEvaluation, EvaluationGroup, ResultBlock } from '@/types/api';

const modes = { analysis: '实时分析', trading: '实时交易决策', backtest: '回测' };
const statuses = {
  pending: '未到期',
  evaluated: '已评估',
  missing_market: '缺行情',
  not_directional: '无方向',
  skipped: '已跳过',
  failed: '失败',
};
const directions = { long: '看多', short: '看空', neutral: '中性' };
const date = (value: string) => new Date(value).toLocaleString('zh-CN', { hour12: false });

function GroupSummary({ group }: { group: EvaluationGroup }) {
  return (
    <article className="rounded-lg border border-border p-4 space-y-3">
      <h3 className="font-medium">
        {modes[group.mode]} · {group.pair ?? '品种未知'} · {group.interval ?? '周期未知'} · 配置 {group.config_revision}
      </h3>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <p className="text-xl tabular-nums font-semibold">
          {group.hit_rate === null ? '尚无可评估样本' : `${(group.hit_rate * 100).toFixed(1)}%`}
        </p>
        <p>
          命中 {group.hits} / 有效方向样本 {group.matured_directional}
        </p>
        <p>总样本 {group.total}</p>
      </div>
      <dl className="grid grid-cols-2 gap-2 text-muted-foreground sm:grid-cols-5">
        {(
          [
            ['未到期', group.pending],
            ['无方向', group.neutral],
            ['缺行情', group.missing_market],
            ['失败', group.failed],
            ['已跳过', group.skipped],
          ] as const
        ).map(([label, count]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd className="text-foreground tabular-nums">{count}</dd>
          </div>
        ))}
      </dl>
    </article>
  );
}

function EvaluationDetail({ record }: { record: ComponentEvaluation }) {
  const blocks: ResultBlock[] = record.comparisons.map((comparison) => ({
    kind: 'series',
    title: `${comparison.title} · ${comparison.name}对照`,
    forecast_start: null,
    evaluation_target: null,
    series: [
      { name: '原预测', unit: null, points: comparison.points.map((p) => ({ time: p.time, value: p.predicted })) },
      { name: '后续实际', unit: null, points: comparison.points.map((p) => ({ time: p.time, value: p.actual })) },
    ],
  }));
  return (
    <article className="rounded-lg border border-border p-4 space-y-3">
      <header className="flex flex-wrap justify-between gap-2">
        <Link
          className="inline-flex min-h-10 items-center text-primary underline"
          to={`/decisions/${encodeURIComponent(record.decision_id)}`}
        >
          {date(record.created_at)} · 查看原决策
        </Link>
        <p>
          {statuses[record.status]} · {directions[record.direction]}
        </p>
      </header>
      {record.reference ? (
        <p className="text-muted-foreground">
          参考收盘 {date(record.reference.reference_time)} · {record.reference.reference_price} → 截止{' '}
          {date(record.reference.due_at)} · {record.actual_price ?? '尚无截止行情'}
        </p>
      ) : (
        <p>缺少原始参考行情，无法补造评估起点。</p>
      )}
      {record.hit !== null ? (
        <p>
          {record.hit ? '方向命中' : '方向未命中'} · 市场价格变化{' '}
          {record.return_ratio === null ? '未知' : `${(Number(record.return_ratio) * 100).toFixed(2)}%`}（不是交易收益）
        </p>
      ) : null}
      <p>{record.cost === null ? '费用未知' : `原记录费用 ${record.cost} USD`}</p>
      {record.reason === 'frozen_market_configuration_missing' ? (
        <p>原行情源配置未冻结，无法评估。不会使用当前配置回填。</p>
      ) : null}
      {record.reason === 'market_read_failed' ? (
        <p role="status">历史行情读取失败，后台将重试；原预测保持不变。</p>
      ) : null}
      {record.reason === 'curve_initialization_failed' ? (
        <p>原预测曲线参数无法解析，本样本未参与评估；可查看原决策。</p>
      ) : null}
      {record.status === 'missing_market' && record.reason === 'due_candle_missing' ? (
        <p>原来源缺少截止收盘行情，待同步后自动补评。</p>
      ) : null}
      <ResultBlocks blocks={blocks} />
      {record.comparisons.map((comparison, index) => (
        <details key={`${comparison.name}-${index}`} className="rounded border border-border px-3">
          <summary className="min-h-10 cursor-pointer py-2">
            {comparison.name} · 匹配 {comparison.matched}/{comparison.total} ·{' '}
            {comparison.mae === null ? '完整曲线误差待齐全' : `MAE ${comparison.mae} / RMSE ${comparison.rmse}`}
          </summary>
          <p className="py-2 text-muted-foreground">
            按原 {comparison.timeframe} K线开盘时间对齐，收盘后补实际值；差值 = 原预测 − 实际。
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr>
                  {['预测K线开盘', '实际收盘时刻', '原预测', '实际', '差值', '状态'].map((label) => (
                    <th key={label} scope="col" className="p-2 whitespace-nowrap">
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparison.points.map((point, pointIndex) => (
                  <tr key={`${point.time}-${pointIndex}`} className="border-t border-border">
                    <td className="p-2 whitespace-nowrap">{date(point.time)}</td>
                    <td className="p-2 whitespace-nowrap">{date(point.close_time)}</td>
                    <td className="p-2">{point.predicted ?? '未知'}</td>
                    <td className="p-2">{point.actual ?? '—'}</td>
                    <td className="p-2">{point.difference ?? '—'}</td>
                    <td className="p-2 whitespace-nowrap">
                      {{ matched: '已匹配', pending: '未收盘', missing_market: '缺行情' }[point.status]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      ))}
    </article>
  );
}

export function ComponentEvaluation({ componentId }: { componentId: string }) {
  const [filters, setFilters] = useState<EvaluationFilters>({
    pair: '',
    mode: '',
    config_revision: '',
    interval: '',
    status: '',
  });
  const [page, setPage] = useState(1);
  const result = useComponentEvaluations(componentId, filters, page);
  function change(key: keyof EvaluationFilters, value: string) {
    setFilters((previous) => ({ ...previous, [key]: value }));
    setPage(1);
  }
  return (
    <section className="space-y-4" aria-label="组件效果">
      <header>
        <h2 className="text-lg font-semibold">方向命中率</h2>
        <p className="mt-2 text-muted-foreground">
          分母仅含已到期、有方向且取得截止行情的样本；持平不命中。按品种、来源、配置版本与周期分别统计，不混算回测。命中率不是收益率，不自动调整信任权重。
        </p>
      </header>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <label>
          样本来源
          <select
            className="configuration-input mt-1 min-h-10 w-full"
            value={filters.mode}
            onChange={(e) => change('mode', e.target.value)}
          >
            <option value="">全部（分别统计）</option>
            {Object.entries(modes).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        {(
          [
            ['pair', '品种', 'BTC/USDT'],
            ['config_revision', '配置版本', '7'],
            ['interval', '评估周期', '2h'],
          ] as const
        ).map(([key, label, placeholder]) => (
          <label key={key}>
            {label}
            <input
              className="configuration-input mt-1 min-h-10 w-full"
              type={key === 'config_revision' ? 'number' : 'text'}
              min={key === 'config_revision' ? 1 : undefined}
              placeholder={placeholder}
              value={filters[key]}
              onChange={(e) => change(key, e.target.value)}
            />
          </label>
        ))}
        <label>
          记录状态
          <select
            className="configuration-input mt-1 min-h-10 w-full"
            value={filters.status}
            onChange={(e) => change('status', e.target.value)}
          >
            <option value="">全部状态</option>
            {Object.entries(statuses).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="text-muted-foreground">
        状态筛选仅筛选下方记录，不改变命中率分母。后台每分钟补评，浏览不会启动推理或交易。
      </p>
      {result.isPending ? <p role="status">正在读取效果评估…</p> : null}
      {result.isError ? (
        <div role="alert">
          <p>效果数据暂不可读，请重试。</p>
          <button className="configuration-button min-h-10" onClick={() => void result.refetch()}>
            重试读取
          </button>
        </div>
      ) : null}
      {result.data ? (
        <>
          {result.data.summary.groups.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border p-6">
              <h3 className="font-medium">尚无可评估样本</h3>
              <p className="mt-2 text-muted-foreground">
                当前筛选下没有评估记录。可调整筛选，或等待已保存的预测到期后由后台补评。
              </p>
            </div>
          ) : (
            result.data.summary.groups.map((group) => (
              <GroupSummary
                key={[group.pair, group.mode, group.config_revision, group.interval].join('|')}
                group={group}
              />
            ))
          )}
          <h3 className="font-medium">评估记录 · {result.data.total} 条</h3>
          {result.data.items.map((record) => (
            <EvaluationDetail key={record.decision_id} record={record} />
          ))}
          <nav aria-label="评估分页" className="flex items-center justify-between">
            <button
              className="configuration-button min-h-10"
              disabled={page === 1 || result.isFetching}
              onClick={() => setPage((value) => value - 1)}
            >
              上一页
            </button>
            <span>第 {page} 页</span>
            <button
              className="configuration-button min-h-10"
              disabled={!result.data.has_next || result.isFetching}
              onClick={() => setPage((value) => value + 1)}
            >
              下一页
            </button>
          </nav>
        </>
      ) : null}
    </section>
  );
}
