import { useRuntimeStatus, useSetAutomation } from '@/hooks/use-runtime-status';
import { useRuntimeConfigConflict } from '@/hooks/runtime-config-conflict';

export function ReadinessPanel() {
  const status = useRuntimeStatus();
  const automation = useSetAutomation();
  const conflict = useRuntimeConfigConflict();
  if (status.isPending) return <p role="status">正在读取就绪状态…</p>;
  if (!status.data)
    return (
      <div role="alert">
        就绪状态读取失败。
        <button className="configuration-button min-h-10" onClick={() => void status.refetch()}>
          重试
        </button>
      </div>
    );
  const data = status.data;
  return (
    <section className="space-y-3 rounded-lg border p-4 text-sm" aria-label="运行就绪">
      <h2 className="font-semibold">运行就绪</h2>
      <p>
        配置已保存：版本 {data.saved_revision} · 已应用：
        {data.applied_revision === null ? '尚未应用' : `版本 ${data.applied_revision}`}
      </p>
      {data.apply_error ? <p role="alert">配置应用失败：{data.apply_error}</p> : null}
      {conflict ? <p role="alert">配置状态待重新核对，请重新加载配置。</p> : null}
      <p>
        {data.automation_enabled
          ? data.trading.ready
            ? '自动运行已开启，等待定时或触发来源'
            : '自动运行已开启，等待交易就绪'
          : '自动运行已暂停'}
      </p>
      <button
        className="configuration-button min-h-11"
        disabled={automation.isPending || conflict}
        onClick={() => automation.mutate({ enabled: !data.automation_enabled, expected_revision: data.saved_revision })}
      >
        {data.automation_enabled ? '暂停自动运行' : '开启自动运行'}
      </button>
      <p className="text-muted-foreground">
        总开关只控制新定时与触发运行；不取消已开始的交易，也不代替真实资金授权和人工审批。
      </p>
      {automation.isError ? <p role="alert">开关未能保存，请刷新配置后重试。</p> : null}
      <p>
        分析：{data.analysis.ready ? '已就绪' : '未就绪'} · 交易：{data.trading.ready ? '已就绪' : '未就绪'}
      </p>
      {data.trading.reasons.map((reason) => (
        <p key={`${reason.code}:${reason.path}`} className="text-muted-foreground">
          {reason.message}
        </p>
      ))}
      <p>最近运行：{data.latest_run_at ? new Date(data.latest_run_at).toLocaleString() : '尚无记录'}</p>
    </section>
  );
}
