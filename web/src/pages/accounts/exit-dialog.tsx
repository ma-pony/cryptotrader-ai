import { useState } from 'react';
import { useSearchParams } from 'react-router';
import { Dialog, DialogContent, DialogDescription, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { useAccountOperation, useAccountOperations } from '@/hooks/use-account-operations';
import { useConfiguration } from '@/pages/settings/configuration-context';
import type { Account } from '@/types/api';
import { formatDateTime } from '@/lib/format';

export function ExitDialog({ account }: { account: Account }) {
  const runtime = useConfiguration();
  const [search, setSearch] = useSearchParams();
  const operationId = search.get('operation_account') === account.connection_id ? search.get('operation') : null;
  const [open, setOpen] = useState(Boolean(operationId));
  const [kind, setKind] = useState<'flatten' | 'cancel_orders'>('flatten');
  const pairs = [
    ...new Set(
      [
        ...(account.snapshot?.positions.filter((p) => p.instrument.tradable).map((p) => p.instrument.pair) ?? []),
        ...(account.snapshot?.orders.filter((o) => o.instrument.tradable).map((o) => o.instrument.pair) ?? []),
        ...(runtime.baseline?.execution.pairs ?? []),
      ].filter((p): p is string => Boolean(p)),
    ),
  ];
  const [selectedPair, setPair] = useState('');
  const pair = selectedPair || pairs[0] || '';
  const query = useAccountOperation(operationId);
  const operation = query.data;
  const api = useAccountOperations(account.connection_id);
  const books = runtime.baseline?.execution.books.filter((b) => account.book_ids.includes(b.id)) ?? [];
  const plan = operation?.plan;
  const busy =
    api.prepare.isPending || api.execute.isPending || ['preparing', 'executing'].includes(operation?.status ?? '');
  const terminal = operation && ['completed', 'failed', 'invalidated'].includes(operation.status);
  const error = api.prepare.error || api.execute.error || query.error;
  return (
    <div>
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <button className="configuration-button" disabled={account.archived || (!runtime.revision && !operationId)}>
          人工退出
        </button>
      </DialogTrigger>
      <DialogContent className="max-h-[90dvh] w-[calc(100%-2rem)] overflow-y-auto text-sm">
        <DialogTitle>人工退出 · {account.label}</DialogTitle>
        <DialogDescription>
          先确认停用范围，再核对平台实际计划。其他资金池不受影响，不调用模型或策略审批。
        </DialogDescription>
        {!operationId ? (
          <>
            <section className="rounded border border-border p-3 space-y-2">
              <p className="font-medium">
                将停用：{books.length ? books.map((b) => b.label).join('、') : account.label}
              </p>
              {books.map((book) => (
                <p key={book.id}>
                  池内账户：
                  {book.allocations
                    .filter((a) => a.enabled)
                    .map(
                      (a) =>
                        runtime.baseline?.execution.connections.find((c) => c.id === a.connection_id)?.label ??
                        a.connection_id,
                    )
                    .join('、')}
                </p>
              ))}
              <p>只退出当前账户的所选品种；完成后保持停用，需明确启用才会恢复策略。</p>
            </section>
            <label>
              品种
              <select
                name="account-exit-pair"
                className="configuration-input block w-full"
                value={pair}
                autoComplete="off"
                onChange={(e) => setPair(e.target.value)}
              >
                {!pairs.length ? (
                  <option value="">请先配置品种或刷新账户</option>
                ) : (
                  pairs.map((p) => <option key={p}>{p}</option>)
                )}
              </select>
            </label>
            <label>
              操作
              <select
                name="account-exit-kind"
                className="configuration-input block w-full"
                value={kind}
                autoComplete="off"
                onChange={(e) => setKind(e.target.value as typeof kind)}
              >
                <option value="flatten">撤普通挂单并平仓</option>
                <option value="cancel_orders">只撤普通挂单</option>
              </select>
            </label>
            <p className="configuration-help">保护在确认零仓之前保留；只撤普通挂单不会平仓或移除保护。</p>
            <button
              className="configuration-button configuration-primary"
              disabled={busy || !pair || !runtime.revision}
              onClick={() => {
                void (async () => {
                  try {
                    const accepted = await api.prepare.mutateAsync({
                      pair,
                      kind,
                      expected_revision: runtime.revision!,
                      confirm_stop: true,
                    });
                    setSearch(
                      (current) => {
                        const next = new URLSearchParams(current);
                        next.set('operation', accepted.operation_id);
                        next.set('operation_account', account.connection_id);
                        return next;
                      },
                      { replace: true },
                    );
                  } catch {
                    /* mutation exposes error */
                  }
                })();
              }}
            >
              确认停用并读取计划
            </button>
          </>
        ) : null}
        {operationId && (query.isPending || operation?.status === 'preparing') ? (
          <p role="status">已受理，正在停用并等待在途操作结束…</p>
        ) : null}
        {plan ? (
          <section className="rounded border border-border p-3 space-y-2 break-words">
            <p className="font-medium">
              退出计划 v{plan.version} · {plan.pair} · {plan.capital_scope === 'real' ? '真实资金' : '模拟资金'}
            </p>
            <p>
              账户持仓：{plan.position_amount} · 预计平仓数量：{plan.close_amount}
            </p>
            <p>先撤普通挂单：{plan.ordinary_order_ids.length} 笔</p>
            {plan.ordinary_order_ids.length ? (
              <p className="configuration-help">{plan.ordinary_order_ids.join('、')}</p>
            ) : null}
            <p>保留至零仓确认的保护：{plan.protection_ids.length} 笔</p>
            <p>快照时间：{formatDateTime(plan.snapshot_time)}</p>
            {operation?.status === 'awaiting_confirmation' ? (
              <button
                className="configuration-button configuration-primary"
                disabled={busy}
                onClick={() => {
                  void api.execute
                    .mutateAsync({ id: operation.operation_id, version: plan.version })
                    .catch(() => undefined);
                }}
              >
                确认执行此计划
              </button>
            ) : null}
          </section>
        ) : null}
        {operation?.status === 'executing' || api.execute.isPending ? (
          <p role="status">正在执行，保护将保留至确认零仓。请勿重复提交。</p>
        ) : null}
        {operation?.result.failure_reason ? (
          <p role="alert" className="configuration-error">
            {operation.result.failure_reason}
          </p>
        ) : null}
        {terminal ? (
          <section className="space-y-2" aria-live="polite">
            {operation.status === 'completed' ? (
              <p className="font-medium">
                {operation.kind === 'flatten' ? '退出已完成，账户保持停用' : '普通挂单已撤，持仓与保护保持不变'}
              </p>
            ) : (
              <p>未确认退出完成，请核对实际账户。</p>
            )}
            <p>剩余持仓：{operation.result.remaining_position ?? '未知，需刷新核对'}</p>
            <p>
              已撤普通挂单：{operation.result.canceled_order_ids.length} 笔 · 已撤保护：
              {operation.result.canceled_protection_ids.length} 笔
            </p>
            {operation.result.orders.map((order) => (
              <p key={order.id}>
                实际订单 {order.id} · 已成交 {order.filled_amount} / {order.amount}
              </p>
            ))}
            <button className="configuration-button" onClick={() => void query.refetch()}>
              重新读取状态
            </button>
            <button
              className="configuration-button"
              onClick={() =>
                setSearch(
                  (current) => {
                    const next = new URLSearchParams(current);
                  next.delete('operation');
                  next.delete('operation_account');
                    return next;
                  },
                  { replace: true },
                )
              }
            >
              重新准备计划
            </button>
          </section>
        ) : null}
        {error ? (
          <p role="alert" className="configuration-error">
            请求未完成，请重新读取配置与操作状态后再试。未自动重试下单。
          </p>
        ) : null}
        {operationId ? (
          <p className="configuration-help break-all">操作记录：{operationId}。此页链接可用于稍后查看结果。</p>
        ) : null}
      </DialogContent>
    </Dialog>
    {!account.archived && !runtime.revision && !operationId ? (
      <p className="configuration-help">配置版本不可用，无法安全准备人工退出；请先解锁并重新读取配置。</p>
    ) : null}
    </div>
  );
}
