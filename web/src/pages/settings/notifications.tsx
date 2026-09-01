import { useRef } from 'react';
import { NumberField, TextField } from '@/components/configuration/field';
import { ConfigurationSaveBar } from '@/pages/settings';
import { useConfiguration } from '@/pages/settings/configuration-context';
import { useAlertOverview } from '@/hooks/use-alerts';

const eventLabels = { approval_pending: '等待人工审批', execution_failed: '执行失败或需核对', protection_failed: '保护处理失败', risk_adjusted: '风险限制已应用', component_failed: '信号组件失败', connection_failed: '账户连接失败', daily_summary: '每日运行摘要' } as const;
type EventName = keyof typeof eventLabels;
export default function NotificationSettings() {
  const runtime = useConfiguration();
  const overview = useAlertOverview();
  const form = useRef<HTMLFormElement>(null);
  const value = runtime.document?.notifications;
  if (!value) return null;
  const update = (patch: Partial<typeof value>) => runtime.update('notifications', { ...value, ...patch });
  return <form id="notifications" ref={form} className="space-y-6 text-sm" onSubmit={(event) => { event.preventDefault(); void runtime.save('notifications', form.current ?? undefined); }}>
    <header><h1 className="text-xl font-semibold">告警与通知</h1><p className="configuration-help">站内事项始终保留；这里选择哪些事项额外发送到 Webhook。通知失败不会回滚交易。</p></header>
    <section className="configuration-section space-y-4"><label className="flex min-h-11 items-center gap-3"><input type="checkbox" checked={value.enabled} onChange={(event) => update({ enabled: event.target.checked })} /><span>启用 Webhook 投递</span></label><TextField name="notifications.webhook_url" label="Webhook 地址" type="url" value={value.webhook_url} onChange={(webhook_url) => update({ webhook_url })} error={runtime.errors['notifications.webhook_url']} /><NumberField name="notifications.webhook_timeout" label="超时（秒）" min={1} step={1} value={value.webhook_timeout} onChange={(webhook_timeout) => update({ webhook_timeout })} error={runtime.errors['notifications.webhook_timeout']} /><fieldset><legend className="font-medium">发送事件</legend><div className="mt-2 grid gap-2 sm:grid-cols-2">{(Object.keys(eventLabels) as EventName[]).map((name) => <label className="flex min-h-11 items-center gap-3" key={name}><input type="checkbox" checked={value.events.includes(name)} onChange={(event) => update({ events: event.target.checked ? [...value.events, name] : value.events.filter((item) => item !== name) })} />{eventLabels[name]}</label>)}</div></fieldset></section>
    <section className="configuration-section"><h2 className="text-base font-semibold">最近投递</h2>{overview.isPending ? <p role="status">正在读取投递状态…</p> : overview.isError ? <p role="alert" className="configuration-error">投递状态读取失败，请刷新页面。</p> : !overview.data?.deliveries.length ? <p className="configuration-help">尚无投递记录。保存后，匹配的新事项会在后台发送。</p> : <div className="mt-3 space-y-3">{overview.data.deliveries.slice(0, 10).map((item) => <article className="border-t border-border pt-3" key={item.id}><p>{item.status === 'delivered' ? '已送达' : item.status === 'failed' ? '投递失败' : item.status === 'sending' ? '正在投递' : '等待投递'} · 尝试 {item.attempts} 次</p>{item.last_error ? <p className="configuration-error">{item.last_error}</p> : null}{item.status === 'failed' ? <button type="button" className="configuration-button mt-2" disabled={overview.isRetrying} onClick={() => overview.retry(item.id)}>仅重发通知</button> : null}</article>)}</div>}</section>
    <ConfigurationSaveBar section="notifications" form={form.current} />
  </form>;
}
