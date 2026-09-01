import { useState } from 'react';
import { Link, useParams } from 'react-router';
import { ResultBlocks } from '@/components/signals/result-blocks';
import { useDecisions } from '@/hooks/use-decisions';
import { ConfigurationEditorGate, useConfiguration } from '@/pages/settings/configuration-context';
import SettingsPage from '@/pages/settings';
import { ComponentEvaluation } from './component-evaluation';
import type { Decision, SavedComponentSignal } from '@/types/api';

const statusLabels = { completed: '已完成', skipped: '已跳过', failed: '失败' };
const directions = { long: '看多', short: '看空', neutral: '中性' };
const date = (value: string) => new Date(value).toLocaleString('zh-CN', { hour12: false });

function SavedResult({ signal, record }: { signal: SavedComponentSignal; record: Decision }) {
  const reference = signal.evaluation_reference;
  return (
    <article className="space-y-4 rounded-lg border border-border p-4">
      <header className="flex flex-wrap justify-between gap-2">
        <time dateTime={record.created_at}>{date(record.created_at)}</time>
        <span>
          配置版本 {record.config_revision} · {statusLabels[signal.status]}
        </span>
      </header>
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div>
          <dt className="text-muted-foreground">方向</dt>
          <dd>{directions[signal.direction]}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">置信度</dt>
          <dd>{(signal.confidence * 100).toFixed(1)}%</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">耗时</dt>
          <dd>{signal.duration_ms === null ? '耗时未知' : `${signal.duration_ms} ms`}</dd>
        </div>
        <div>
          <dt className="text-muted-foreground">费用</dt>
          <dd>{signal.cost === null ? '费用未知' : `${signal.cost} USD`}</dd>
        </div>
      </dl>
      {signal.usage ? (
        <p>
          输入 {signal.usage.input_tokens} / 输出 {signal.usage.output_tokens} tokens
        </p>
      ) : null}
      <p className="whitespace-pre-wrap break-words">{signal.reasoning}</p>
      {record.target?.size_ratio === 0 ? <p>目标持仓为零</p> : null}
      <Link className="inline-flex min-h-10 items-center text-primary" to={`/decisions/${record.decision_id}`}>
        查看完整决策
      </Link>
      {reference ? (
        <p className="text-muted-foreground">
          参考收盘 {date(reference.reference_time)} · {reference.reference_price} · 评估到期 {date(reference.due_at)}（
          {reference.interval}）
        </p>
      ) : (
        <p className="text-muted-foreground">参考行情缺失，无法评估本次结果。</p>
      )}
      <ResultBlocks blocks={signal.blocks} />
    </article>
  );
}

export function ComponentDetail({ componentId }: { componentId: string }) {
  const [section, setSection] = useState<'overview' | 'history' | 'evaluation' | 'configuration'>('overview');
  const [page, setPage] = useState(1);
  const config = useConfiguration();
  const history = useDecisions(page, componentId);
  const definition = config.catalog.data?.components.find((component) => component.id === componentId);
  const label = definition?.label.zh_CN ?? componentId;
  const rows = (history.data?.items ?? []).flatMap((record) => {
    const signal = record.components.find((item) => item.component_id === componentId);
    return signal ? [{ record, signal }] : [];
  });
  return (
    <main className="min-w-0 space-y-5 text-sm">
      <Link to="/engine" className="inline-flex min-h-10 items-center text-primary">
        返回信号引擎
      </Link>
      <header>
        <h1 className="text-xl font-semibold">{label}</h1>
        <p className="mt-2 text-muted-foreground">
          {definition?.description.zh_CN ?? '当前组件未注册；仍可查看已保存历史。'}
        </p>
      </header>
      <nav className="flex gap-2" aria-label="组件视图">
        {(
          [
            ['overview', '概览'],
            ['history', '历史'],
            ['evaluation', '效果'],
            ['configuration', '配置'],
          ] as const
        ).map(([key, text]) => (
          <button
            key={key}
            type="button"
            aria-pressed={section === key}
            className="configuration-button min-h-10"
            onClick={() => {
              setSection(key);
              setPage(1);
            }}
          >
            {text}
          </button>
        ))}
      </nav>
      {section === 'configuration' ? (
        <ConfigurationEditorGate>
          <SettingsPage section="signals" componentId={componentId} />
        </ConfigurationEditorGate>
      ) : section === 'evaluation' ? (
        <ComponentEvaluation componentId={componentId} />
      ) : (
        <>
          {history.isPending ? <p role="status">正在读取保存结果…</p> : null}
          {history.isError ? (
            <div role="alert">
              <p>历史暂不可读，请检查访问权限或重试；读取不会启动推理。</p>
              <button className="configuration-button" onClick={() => void history.refetch()}>
                重试读取
              </button>
            </div>
          ) : null}
          {!history.isPending && !history.isError && rows.length === 0 ? (
            <section className="rounded-lg border border-dashed border-border p-6">
              <h2 className="font-semibold">暂无运行历史</h2>
              <p className="my-2 text-muted-foreground">先检查组件配置。此页面不会启动分析或交易。</p>
              <Link className="inline-flex min-h-10 items-center text-primary underline" to="/engine#configuration">
                前往引擎配置
              </Link>
            </section>
          ) : null}
          {(section === 'overview' ? rows.slice(0, 1) : rows).map(({ record, signal }) => (
            <SavedResult key={record.decision_id} signal={signal} record={record} />
          ))}
          {section === 'history' && rows.length > 0 ? (
            <nav className="flex items-center justify-between" aria-label="历史分页">
              <button
                className="configuration-button"
                disabled={page === 1 || history.isFetching}
                onClick={() => setPage((value) => value - 1)}
              >
                上一页
              </button>
              <span>
                第 {page} 页 · 共 {history.data?.total} 条
              </span>
              <button
                className="configuration-button"
                disabled={!history.data?.has_next || history.isFetching}
                onClick={() => setPage((value) => value + 1)}
              >
                下一页
              </button>
            </nav>
          ) : null}
        </>
      )}
    </main>
  );
}

export default function ComponentDetailPage() {
  const { componentId } = useParams<{ componentId: string }>();
  return componentId ? <ComponentDetail key={componentId} componentId={componentId} /> : null;
}
