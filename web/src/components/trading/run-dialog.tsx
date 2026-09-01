import { useState } from 'react';
import { useNavigate } from 'react-router';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useTradingScope, useStartTrading } from '@/hooks/use-trading-runs';

export function RunDialog({ pairs, onClose }: { pairs: string[]; onClose: () => void }) {
  const [pair, setPair] = useState(pairs[0] ?? '');
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-h-[90vh] overflow-y-auto text-sm">
        <DialogHeader>
          <DialogTitle>确认交易范围</DialogTitle>
          <DialogDescription>
            一份全局决策将分发至下列全部可执行资金池。真实资金授权与逐池审批独立生效。
          </DialogDescription>
        </DialogHeader>
        <label className="grid gap-2">
          交易品种
          <select
            className="configuration-control min-h-11"
            value={pair}
            onChange={(event) => setPair(event.target.value)}
          >
            {!pairs.length ? <option value="">请先配置交易品种范围</option> : null}
            {pairs.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        {pair ? <ScopeConfirmation key={pair} pair={pair} /> : <p>请到资金池配置添加交易品种。</p>}
      </DialogContent>
    </Dialog>
  );
}

function ScopeConfirmation({ pair }: { pair: string }) {
  const scope = useTradingScope(pair);
  const start = useStartTrading();
  const navigate = useNavigate();
  const [confirmed, setConfirmed] = useState<Record<string, boolean>>({});
  const [confirmedRevision, setConfirmedRevision] = useState<number>();
  if (scope.isPending) return <p role="status">正在读取交易范围…</p>;
  if (scope.isError || !scope.data)
    return (
      <div role="alert">
        范围读取失败。
        <button className="configuration-button min-h-10" onClick={() => void scope.refetch()}>
          重试
        </button>
      </div>
    );
  const data = scope.data;
  const eligible = data.books.filter((book) => book.eligible);
  const ready =
    data.ready &&
    eligible.length > 0 &&
    confirmedRevision === data.saved_revision &&
    eligible.every((book) => confirmed[book.book_id]);
  return (
    <div className="space-y-3">
      <p>
        配置版本 {data.saved_revision} · {eligible.length} 个可执行资金池
      </p>
      {data.reasons.map((reason) => (
        <p key={`${reason.code}:${reason.path}`} className="text-destructive">
          {reason.message}
        </p>
      ))}
      {data.books.map((book) => (
        <section className="rounded-md border p-3" key={book.book_id}>
          <label className="flex min-h-11 items-center gap-3">
            <input
              type="checkbox"
              disabled={!book.eligible || start.isPending}
              checked={confirmedRevision === data.saved_revision && Boolean(confirmed[book.book_id])}
              onChange={(event) => {
                setConfirmed({
                  ...(confirmedRevision === data.saved_revision ? confirmed : {}),
                  [book.book_id]: event.target.checked,
                });
                setConfirmedRevision(data.saved_revision);
              }}
            />
            <span>
              {book.label} · {book.capital_scope === 'real' ? '真实资金' : '模拟资金'} ·{' '}
              {book.hitl_required ? '需要人工审批' : '无需人工审批'}
            </span>
          </label>
          <p className="text-muted-foreground">
            账户：
            {book.connections
              .map(
                (connection) =>
                  `${connection.label} [${connection.connection_id}]（${connection.environment}，${connection.enabled ? '已启用' : '已停用'}）`,
              )
              .join('、') || '未分配'}
          </p>
          {!book.eligible ? <p>{book.enabled ? '本次跳过' : '资金池已停用，本次跳过'}</p> : null}
          {book.reasons.map((reason) => (
            <p key={`${reason.code}:${reason.path}`} className="text-muted-foreground">
              {reason.message}
            </p>
          ))}
        </section>
      ))}
      {start.isError ? <p role="alert">交易未启动。请刷新范围、检查就绪原因并重新确认。</p> : null}
      <button
        className="configuration-button configuration-primary min-h-11"
        disabled={!ready || start.isPending || scope.isFetching}
        onClick={() => {
          start.mutate(
            { pair, expected_revision: data.saved_revision, confirmed_book_ids: eligible.map((book) => book.book_id) },
            {
              onSuccess: ({ decision_id }) => {
                void navigate(`/decisions/${decision_id}`);
              },
              onError: () => {
                setConfirmed({});
              },
            },
          );
        }}
      >
        {start.isPending ? '正在提交…' : '确认全部范围并发起交易'}
      </button>
    </div>
  );
}
