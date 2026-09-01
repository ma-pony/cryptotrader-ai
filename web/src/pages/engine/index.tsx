import { useState } from 'react';
import { Link, useNavigate } from 'react-router';
import { useStartAnalysis } from '@/hooks/use-decisions';
import { ConfigurationEditorGate, useConfiguration } from '@/pages/settings/configuration-context';
import SettingsPage from '@/pages/settings';
import { useRuntimeStatus } from '@/hooks/use-runtime-status';
import { ReadinessPanel } from '@/components/trading/readiness-panel';
import { RunDialog } from '@/components/trading/run-dialog';
import AutomationRules from './automation';

function EngineConfiguration({ readiness }: { readiness: ReturnType<typeof useRuntimeStatus> }) {
  const config = useConfiguration();
  const document = config.document!;
  const catalog = config.catalog.data!;
  return (
    <>
      <div className="grid gap-4 md:grid-cols-2">
        {catalog.components.map((component) => {
          const configured = document.signals.components.find((item) => item.component_id === component.id);
          const capability = readiness.data?.components.find((item) => item.component_id === component.id);
          return (
            <article className="rounded-lg border border-border bg-card p-4" key={component.id}>
              <h2 className="font-semibold">{component.label.zh_CN}</h2>
              <p className="my-2 text-muted-foreground">{component.description.zh_CN}</p>
              <p>{configured?.enabled ? '已启用' : '未启用'} · 信任权重 {configured?.weight ? Number(configured.weight * 100).toFixed(0) : 0}%</p>
              {capability ? (
                <div className="mt-2 space-y-1">
                  <p>
                    依赖：{capability.ready ? '已就绪' : '未就绪'}
                    {!capability.enabled ? '（未启用，不阻止分析）' : ''}
                  </p>
                  {capability.dependencies.map((dependency) => (
                    <p key={`${dependency.kind}:${dependency.key}`} className="text-muted-foreground">
                      {dependency.label}：{dependency.ready ? '可用' : '缺失'}
                      {dependency.reasons.map((reason) => ` · ${reason.message}`).join('')}
                    </p>
                  ))}
                </div>
              ) : null}
              <Link className="mt-3 inline-flex min-h-10 items-center text-primary underline" to={`/engine/components/${encodeURIComponent(component.id)}`}>
                查看 {component.label.zh_CN} 结果
              </Link>
            </article>
          );
        })}
      </div>
      <section id="configuration" className="space-y-4">
        <h2 className="text-lg font-semibold">引擎配置</h2>
        <div id="market" className="configuration-section"><h3 className="mb-3 text-base font-semibold">行情与上下文</h3><SettingsPage section="market" /></div>
        <div id="signals" className="configuration-section"><h3 className="mb-3 text-base font-semibold">信号、Kronos 与大模型委员会</h3><SettingsPage section="signals" /></div>
        <div id="risk" className="configuration-section"><h3 className="mb-3 text-base font-semibold">融合与风险</h3><SettingsPage section="risk" /></div>
        <div id="automation" className="configuration-section space-y-4">
          <h3 className="mb-3 text-base font-semibold">自动运行来源</h3>
          <SettingsPage section="scheduler" />
          <AutomationRules />
        </div>
      </section>
    </>
  );
}

export default function EnginePage() {
  const config = useConfiguration();
  const [pair, setPair] = useState('BTC/USDT:USDT');
  const start = useStartAnalysis();
  const readiness = useRuntimeStatus();
  const [tradingOpen, setTradingOpen] = useState(false);
  const navigate = useNavigate();
  return (
    <main className="space-y-6 text-sm">
      <header>
        <h1 className="text-xl font-semibold">引擎</h1>
        <p className="mt-2 text-muted-foreground">行情、信号、融合风控与自动来源共享一套全局决策引擎。</p>
      </header>
      <ReadinessPanel />
      <button
        className="configuration-button min-h-11"
        disabled={!readiness.data?.trading.ready}
        onClick={() => setTradingOpen(true)}
      >
        发起交易
      </button>
      {tradingOpen ? (
        <RunDialog pairs={readiness.data?.execution_pairs ?? []} onClose={() => setTradingOpen(false)} />
      ) : null}
      <section className="space-y-3 rounded-lg border border-border p-4" aria-label="独立分析">
        <h2 className="font-semibold">仅分析</h2>
        <p className="text-muted-foreground">使用已保存的引擎配置，不读取交易账户，不创建审批或订单。</p>
        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            if (!readiness.data?.analysis.ready || start.isPending) return;
            start.mutate(
              { pair: pair.trim(), expected_revision: readiness.data.saved_revision },
              {
                onSuccess: ({ decision_id }) => {
                  void navigate(`/decisions/${decision_id}`);
                },
              },
            );
          }}
        >
          <label className="grid gap-1">
            交易对
            <input
              className="configuration-input min-h-10"
              value={pair}
              onChange={(event) => setPair(event.target.value)}
              required
            />
          </label>
          <button
            type="submit"
            className="configuration-button min-h-10"
            disabled={start.isPending || !readiness.data?.analysis.ready || !pair.trim()}
          >
            {start.isPending ? '正在提交…' : '仅分析，不交易'}
          </button>
        </form>
        {config.hasUnsaved ? <p>有未保存更改；本次分析仍使用已保存配置。</p> : null}
        {readiness.data?.analysis.reasons.map((reason) => (
          <p key={`${reason.code}:${reason.path}`}>{reason.message}</p>
        ))}
        {start.isError ? <p role="alert">分析未能启动，请刷新配置版本并检查访问权限后重试。</p> : null}
        <Link to="/decisions" className="inline-flex min-h-10 items-center text-primary">
          查看决策记录
        </Link>
      </section>
      <ConfigurationEditorGate>
        <EngineConfiguration readiness={readiness} />
      </ConfigurationEditorGate>
    </main>
  );
}
