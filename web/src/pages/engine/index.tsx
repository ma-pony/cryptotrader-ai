import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router';
import { Button } from '@/components/ui/button';
import { PageHeader } from '@/components/ui/page-header';
import { SectionTabs } from '@/components/ui/section-tabs';
import { TextField } from '@/components/configuration/field';
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
  const { hash } = useLocation();
  const navigate = useNavigate();
  const active = hash.startsWith('#signals')
    ? 'signals'
    : hash.startsWith('#risk') || hash.startsWith('#hitl')
      ? 'risk'
      : hash.startsWith('#automation') || hash.startsWith('#scheduler') || hash.startsWith('#triggers')
        ? 'automation'
        : 'market';
  return (
    <>
      <section className="space-y-3" aria-label="信号组件概况">
        <h2 className="section-title">信号组件</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {catalog.components.map((component) => {
            const configured = document.signals.components.find((item) => item.component_id === component.id);
            const capability = readiness.data?.components.find((item) => item.component_id === component.id);
            return (
              <article className="signal-summary" key={component.id}>
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <h3 className="font-semibold">{component.label.zh_CN}</h3>
                  <span className="status-muted">{configured?.enabled ? '已启用' : '未启用'}</span>
                </div>
                <p className="my-2 text-muted-foreground">{component.description.zh_CN}</p>
                <p className="tabular-nums">
                  信任权重 <strong>{configured?.weight ? Number(configured.weight * 100).toFixed(0) : 0}%</strong>
                </p>
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
                <Button asChild variant="link" className="mt-2 h-auto justify-start px-0">
                  <Link to={`/engine/components/${encodeURIComponent(component.id)}`}>
                    查看 {component.label.zh_CN} 结果
                  </Link>
                </Button>
              </article>
            );
          })}
        </div>
        <p className="text-muted-foreground">信任权重表示融合影响力，不代表胜率或资金分配。</p>
      </section>
      <section id="configuration" className="space-y-4">
        <h2 className="text-lg font-semibold">引擎配置</h2>
        <SectionTabs
          label="引擎配置分区"
          value={active}
          onValueChange={(value) => void navigate({ hash: `#${value}` }, { preventScrollReset: true })}
          items={[
            {
              id: 'market',
              label: '行情与上下文',
              dirty: config.isDirty('market'),
              content: <SettingsPage section="market" />,
            },
            {
              id: 'signals',
              label: '信号组件',
              dirty: config.isDirty('signals'),
              content: <SettingsPage section="signals" />,
            },
            {
              id: 'risk',
              label: '风控与审批',
              dirty: config.isDirty('risk'),
              content: <SettingsPage section="risk" />,
            },
            {
              id: 'automation',
              label: '自动运行',
              dirty: config.isDirty('scheduler'),
              content: (
                <>
                  <SettingsPage section="scheduler" />
                  <AutomationRules />
                </>
              ),
            },
          ]}
        />
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
    <div className="space-y-6 text-sm">
      <PageHeader
        title="引擎"
        subtitle="管理行情、信号组件、风控和自动运行。修改配置不会自动发起交易。"
        actions={
          <Button variant="outline" disabled={!readiness.data?.trading.ready} onClick={() => setTradingOpen(true)}>
            发起交易
          </Button>
        }
      />
      {tradingOpen ? (
        <RunDialog pairs={readiness.data?.execution_pairs ?? []} onClose={() => setTradingOpen(false)} />
      ) : null}
      <div className="grid items-start gap-4 xl:grid-cols-2">
        <ReadinessPanel />
        <section className="operational-panel" aria-label="独立分析">
          <h2 className="font-semibold">仅分析</h2>
          <p className="text-muted-foreground">使用已保存的引擎配置，不读取交易账户，不创建审批或订单。</p>
          <form
            className="grid gap-3"
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
            <TextField name="engine-analysis-pair" label="交易对" value={pair} onChange={setPair} required />
            <Button
              className="justify-self-start"
              type="submit"
              disabled={start.isPending || !readiness.data?.analysis.ready || !pair.trim()}
            >
              {start.isPending ? '正在提交…' : '仅分析，不交易'}
            </Button>
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
      </div>
      <ConfigurationEditorGate>
        <EngineConfiguration readiness={readiness} />
      </ConfigurationEditorGate>
    </div>
  );
}
