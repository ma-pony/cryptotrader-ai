import { AlertTriangle, CheckCircle2, Plus, Save, SlidersHorizontal } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { toRuntimeDocument, useRuntimeConfig } from '@/hooks/use-runtime-config';
import type { RuntimeDocument, RuntimeJsonObject } from '@/types/api';
import { ComponentWeightCard, type ComponentWeightDraft } from './components/component-weight-card';
import { DecisionSettingsCard, type DecisionSettingsDraft } from './components/decision-settings-card';

const ACCENTS = ['#f59e0b', '#38bdf8', '#a78bfa', '#34d399', '#fb7185'];
type Draft = DecisionSettingsDraft & {
  components: Array<ComponentWeightDraft & { parameters: RuntimeJsonObject }>;
  models: Record<string, string>;
};
const fromDocument = (document: RuntimeDocument): Draft => ({
  ...document.signals,
  components: document.signals.components.map((component) => ({ ...component })),
  models: { ...(document.llm.models as unknown as Record<string, string>) },
});

const StrategyPage = () => {
  const runtime = useRuntimeConfig();
  const [draft, setDraft] = useState<Draft>();
  const [customId, setCustomId] = useState('');
  const [saved, setSaved] = useState(false);
  const [parameterText, setParameterText] = useState<Record<string, string>>({});
  const [parameterErrors, setParameterErrors] = useState<Record<string, boolean>>({});
  // A query cache revision may change while this editor is open.  Only an empty
  // editor hydrates from it; explicit reload is the sole path that discards a draft.
  useEffect(() => {
    if (runtime.document && !draft) setDraft(fromDocument(runtime.document));
  }, [runtime.document, draft]);
  const enabled = draft?.components.filter((component) => component.enabled) ?? [];
  const total = enabled.reduce((sum, component) => sum + component.weight, 0);
  const valid =
    enabled.length > 0 &&
    Math.abs(total - 1) < 1e-9 &&
    !Object.values(parameterErrors).some(Boolean) && Boolean(
      draft &&
      draft.neutral_threshold >= 0 &&
      draft.neutral_threshold < 1 &&
      draft.max_target_ratio > 0 &&
      draft.max_target_ratio <= 1 &&
      draft.atr_stop_multiplier > 0 &&
      draft.reward_ratio > 0,
    );
  const updateComponent = (next: ComponentWeightDraft) =>
    setDraft(
      (current) =>
        current && {
          ...current,
          components: current.components.map((component) =>
            component.component_id === next.component_id ? { ...component, ...next } : component,
          ),
        },
    );
  const save = async () => {
    if (!runtime.document || !draft) return;
    setSaved(false);
    try {
      await runtime.replace({
        ...runtime.document,
        signals: {
          ...runtime.document.signals,
          components: draft.components,
          neutral_threshold: draft.neutral_threshold,
          max_target_ratio: draft.max_target_ratio,
          atr_stop_multiplier: draft.atr_stop_multiplier,
          reward_ratio: draft.reward_ratio,
          hitl_required: draft.hitl_required,
        },
        llm: { ...runtime.document.llm, models: { ...runtime.document.llm.models, ...draft.models } },
      });
      setSaved(true);
    } catch {
      setSaved(false);
    }
  };
  const reload = async () => {
    const result = await runtime.reload();
    if (result.isSuccess && !result.error && result.data) setDraft(fromDocument(toRuntimeDocument(result.data.document)));
  };
  const labels = useMemo(
    () => ({
      kronos: ['Kronos', '时序预测与市场结构信号'],
      llm_committee: ['LLM 四智能体委员会', '技术、链上、新闻、宏观内部辩论'],
    }),
    [],
  );
  return (
    <PageBoundary
      loading={runtime.isLoading}
      isError={runtime.isError}
      onRetry={() => void reload()}
      errorTitle="无法读取运行时配置"
      errorDescription="请检查配置服务后重试。"
    >
      {runtime.document && draft ? (
        <div className="space-y-6">
          <PageHeader
            eyebrow="SIGNAL FUSION"
            title="信号策略"
            subtitle="Kronos、LLM 四智能体委员会和自定义组件统一输出目标仓位。"
            actions={
              <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 font-mono text-xs text-amber-500">
                Revision {runtime.revision}
              </span>
            }
          />
          <p className="text-xs text-muted-foreground">最近配置更新时间：{runtime.updatedAt ?? '—'}</p>
          <section className="relative overflow-hidden rounded-2xl border border-border bg-card p-6">
            <div className="flex items-end justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[.18em] text-amber-500">
                  <SlidersHorizontal className="h-4 w-4" />
                  信任混合器
                </div>
                <p className="mt-3 text-sm text-muted-foreground">启用组件权重必须合计 100%，每个周期只融合一次。</p>
              </div>
              <div className="font-mono text-4xl font-semibold">{Math.round(total * 100)}%</div>
            </div>
            <div className="mt-5 flex h-3 overflow-hidden rounded-full border border-border bg-muted">
              {enabled.map((component, index) => (
                <span
                  key={component.component_id}
                  style={{ width: `${component.weight * 100}%`, backgroundColor: ACCENTS[index % ACCENTS.length] }}
                />
              ))}
            </div>
            {!valid ? (
              <p role="alert" className="mt-3 flex gap-2 text-xs text-amber-500">
                <AlertTriangle className="h-4 w-4" />
                至少启用一个组件，且权重必须合计 100%。
              </p>
            ) : null}
          </section>
          <section>
            <h2 className="mb-3 font-semibold">组件信任权重</h2>
            <div className="grid gap-4 xl:grid-cols-2">
              {draft.components.map((component, index) => {
                const label = labels[component.component_id as keyof typeof labels];
                return (<div key={component.component_id} className="rounded-xl border border-border p-3"><ComponentWeightCard
                    component={component}
                    displayName={label?.[0] ?? component.component_id}
                    description={label?.[1] ?? '自定义运行时信号组件'}
                    accent={ACCENTS[index % ACCENTS.length] ?? '#f59e0b'}
                    onChange={updateComponent}
                  />
                  <label className="block text-xs text-muted-foreground">参数 JSON<textarea aria-label={`${component.component_id} 参数`} value={parameterText[component.component_id] ?? JSON.stringify(component.parameters)} onChange={(event) => { const text = event.target.value; setParameterText((current) => ({ ...current, [component.component_id]: text })); try { const parameters = JSON.parse(text) as RuntimeJsonObject; if (!parameters || Array.isArray(parameters)) throw new Error(); setDraft((current) => current && ({ ...current, components: current.components.map((item) => item.component_id === component.component_id ? { ...item, parameters } : item) })); setParameterErrors((current) => ({ ...current, [component.component_id]: false })); } catch { setParameterErrors((current) => ({ ...current, [component.component_id]: true })); } }} className="mt-1 min-h-20 w-full rounded border bg-background p-2 font-mono" /></label>
                </div>);
              })}
            </div>
            <div className="mt-4 flex gap-2">
              <input
                aria-label="自定义 component ID"
                value={customId}
                onChange={(event) => setCustomId(event.target.value)}
                placeholder="自定义 component ID"
                className="h-10 flex-1 rounded border bg-background px-3"
              />
              <Button
                variant="outline"
                onClick={() => {
                  const id = customId.trim();
                  if (id && !draft.components.some((component) => component.component_id === id)) {
                    setDraft({
                      ...draft,
                      components: [
                        ...draft.components,
                        { component_id: id, enabled: false, weight: 0, parameters: {} },
                      ],
                    });
                    setCustomId('');
                  }
                }}
              >
                <Plus className="h-4 w-4" />
                添加
              </Button>
            </div>
          </section>
          <DecisionSettingsCard settings={draft} onChange={(settings) => setDraft({ ...draft, ...settings })} />
          <section className="rounded-2xl border border-border bg-card p-5">
            <h2 className="font-semibold">内部辩论模型</h2>
            <p className="mt-1 text-sm text-muted-foreground">四位委员会成员、辩论和汇总模型均在数据库配置。</p>
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              {['tech_agent', 'chain_agent', 'news_agent', 'macro_agent', 'debate', 'committee_summary'].map((key) => (
                <label key={key} className="text-xs text-muted-foreground">
                  {key}
                  <input
                    aria-label={key}
                    value={draft.models[key] ?? ''}
                    onChange={(event) => setDraft({ ...draft, models: { ...draft.models, [key]: event.target.value } })}
                    className="mt-1 h-10 w-full rounded border bg-background px-3 font-mono"
                  />
                </label>
              ))}
            </div>
          </section>
          {runtime.conflict ? (
            <div className="flex items-center gap-3">
              <p role="alert" className="text-sm text-trade-short">
                配置已被其他操作更新，请重新加载
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void reload()}
              >
                重新加载
              </Button>
            </div>
          ) : null}
          <div className="sticky bottom-4 flex justify-end rounded-xl border border-border bg-card/95 p-3">
            <Button size="lg" disabled={!valid || runtime.isSaving} onClick={() => void save()}>
              <Save className="h-4 w-4" />
              {runtime.isSaving ? '保存中' : '保存完整配置'}
            </Button>
          </div>
          {saved ? (
            <p role="status" className="flex items-center gap-2 text-xs text-trade-long">
              <CheckCircle2 className="h-4 w-4" />
              配置已保存。
            </p>
          ) : null}
          {!runtime.isSaving && !runtime.conflict && !saved ? (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <CheckCircle2 className="h-4 w-4" />
              编辑保留在草稿中，点击保存后才会写入。
            </p>
          ) : null}
        </div>
      ) : null}
    </PageBoundary>
  );
};
export default StrategyPage;
