import { Link, useParams } from 'react-router';
import { useTranslation } from 'react-i18next';
import { ResultBlocks } from '@/components/signals/result-blocks';
import { useDecision } from '@/hooks/use-decisions';
import { formatCycleStatus } from '@/lib/cycle-status';
import { BookAudit } from '@/components/trading/book-audit';

const modes = { analysis: '仅分析', trading: '交易运行', backtest: '历史回测' };
const sides: Record<string, string> = { long: '看多', short: '看空', flat: '空仓', neutral: '中性' };
const date = (value: string) => new Date(value).toLocaleString('zh-CN', { hour12: false });

export default function DecisionDetailPage() {
  const { decisionId } = useParams<{ decisionId: string }>();
  const { t } = useTranslation('cycles');
  const query = useDecision(decisionId);
  const item = query.data;
  if (query.isPending) return <p role="status">正在读取决策记录…</p>;
  if (query.isError || !item)
    return (
      <div role="alert">
        <p>记录暂不可读，请检查访问权限或重试。</p>
        <button className="configuration-button" onClick={() => void query.refetch()}>
          重新读取
        </button>
      </div>
    );
  const pending = ['queued', 'running'].includes(item.status);
  return (
    <main className="min-w-0 space-y-5 text-sm">
      <Link to="/decisions" className="inline-flex min-h-10 items-center text-primary">
        返回决策记录
      </Link>
      <header className="space-y-2">
        <h1 className="text-xl font-semibold">
          {modes[item.mode]} · {item.pair ?? '交易对未知'}
        </h1>
        <p>
          配置版本 {item.config_revision} · {formatCycleStatus(t, item.status)}
        </p>
        <p className="text-muted-foreground">
          开始 {date(item.created_at)}
          {item.finished_at ? ` · 结束 ${date(item.finished_at)}` : ''}
        </p>
      </header>
      {pending ? (
        <p role="status" className="rounded border border-border p-4">
          分析在后台进行。可以离开此页，稍后从决策记录重新打开。
        </p>
      ) : null}
      {item.failure ? (
        <div role="alert" className="rounded border border-destructive p-4">
          <p>{item.failure.message}</p>
          <p className="mt-2 text-muted-foreground">
            阶段：{item.failure.stage} · {item.failure.code}
          </p>
        </div>
      ) : null}
      {item.incomplete_fields.length ? (
        <p className="rounded border border-amber-500 p-3">旧记录部分信息缺失，未使用当前配置或重新推理补造。</p>
      ) : null}
      <section className="space-y-3" aria-label="组件结果">
        {item.components.map((component) => (
          <article key={component.component_id} className="space-y-3 rounded-lg border border-border p-4">
            <h2 className="font-semibold">
              <Link to={`/engine/components/${encodeURIComponent(component.component_id)}`}>
                {component.component_id}
              </Link>
            </h2>
            <p>
              {sides[component.direction]} · 置信度 {(component.confidence * 100).toFixed(1)}% ·{' '}
              {formatCycleStatus(t, component.status)}
            </p>
            <p>{component.reasoning}</p>
            <p className="text-muted-foreground">
              {component.duration_ms === null ? '耗时未知' : `${component.duration_ms} ms`} ·{' '}
              {component.cost === null ? '费用未知' : `${component.cost} USD`}
            </p>
            <ResultBlocks blocks={component.blocks} />
          </article>
        ))}
      </section>
      {item.fusion ? (
        <section className="space-y-2 rounded-lg border border-border p-4">
          <h2 className="font-semibold">融合结果</h2>
          <p>{item.fusion.reasoning}</p>
          <p>方向得分 {item.fusion.score}</p>
          <ul>
            {item.fusion.contributions.map((component) => (
              <li key={component.component_id}>
                {component.component_id} · 当次权重 {(component.weight * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      {item.target ? (
        <section className="space-y-2 rounded-lg border border-border p-4">
          <h2 className="font-semibold">目标持仓</h2>
          <p>
            {sides[item.target.side]} · {(item.target.size_ratio * 100).toFixed(1)}%
          </p>
        </section>
      ) : null}
      {item.mode === 'analysis' && item.status === 'completed' ? (
        <p>仅分析已完成，未访问交易账户，也未创建审批或订单。</p>
      ) : null}
      {item.mode !== 'analysis' ? (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">资金池执行</h2>
          {item.books.map((book) => (
            <BookAudit key={book.book_id} book={book} />
          ))}
        </section>
      ) : null}
    </main>
  );
}
