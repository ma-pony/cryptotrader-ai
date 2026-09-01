import { Link } from 'react-router';
import { useAlerts } from '@/hooks/use-alerts';

const labels: Record<string, string> = { approval_pending: '等待审批', execution_failed: '执行需核对', protection_failed: '保护需核对', risk_adjusted: '风险限制已应用', component_failed: '组件失败', connection_failed: '连接失败', daily_summary: '每日摘要' };
export function AttentionList({ connectionId, bookId }: { connectionId?: string; bookId?: string }) {
  const alerts = useAlerts({ connectionId, bookId });
  if (alerts.isPending) return <p role="status" className="text-sm text-muted-foreground">正在读取待办事项…</p>;
  if (alerts.isError) return <p role="alert" className="text-sm text-destructive">待办事项读取失败。刷新页面后重试。</p>;
  if (!alerts.data?.items.length) return <p className="text-sm text-muted-foreground">当前没有需要关注的事项。</p>;
  return <section aria-label="待办事项" className="rounded-lg border border-border bg-card p-4"><h2 className="text-sm font-semibold">待办事项</h2><div className="mt-3 space-y-3">{alerts.data.items.map((alert) => <article key={alert.id} className="border-t border-border pt-3 first:border-0 first:pt-0"><div className="flex flex-wrap items-center gap-2"><strong className="text-sm">{labels[alert.type] ?? alert.type}</strong><span className="text-sm text-muted-foreground">{alert.resolution === 'open' ? (alert.read_at ? '已读，仍待处理' : '未读，待处理') : '已处理'}</span></div><p className="mt-1 text-sm">{alert.message}</p><div className="mt-2 flex gap-2">{alert.decision_id ? <Link className="configuration-button" to={`/decisions/${encodeURIComponent(alert.decision_id)}`}>查看决策</Link> : null}{alert.connection_id ? <Link className="configuration-button" to={`/accounts/connections/${encodeURIComponent(alert.connection_id)}`}>查看账户</Link> : null}{!alert.read_at ? <button className="configuration-button" disabled={alerts.isMarkingRead} onClick={() => alerts.markRead(alert.id)}>标为已读</button> : null}</div></article>)}</div></section>;
}
