import { AlertTriangle, CheckCircle2, Plus, Save, SlidersHorizontal } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { toRuntimeDocument, useRuntimeConfig } from '@/hooks/use-runtime-config';
import { useRuntimeSecrets } from '@/hooks/use-runtime-secrets';
import type { RuntimeDocument, RuntimeJsonObject } from '@/types/api';
import { ComponentWeightCard, type ComponentWeightDraft } from './components/component-weight-card';
import { DecisionSettingsCard, type DecisionSettingsDraft } from './components/decision-settings-card';
import { useSettingsStore } from '@/stores/use-settings-store';

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
  const { t } = useTranslation(['configuration', 'strategy']);
  const runtime = useRuntimeConfig();
  const secrets = useRuntimeSecrets();
  const setApiKey = useSettingsStore((state) => state.setApiKey);
  const [draft, setDraft] = useState<Draft>();
  const [customId, setCustomId] = useState('');
  const [saved, setSaved] = useState(false);
  const [parameterText, setParameterText] = useState<Record<string, string>>({});
  const [parameterErrors, setParameterErrors] = useState<Record<string, boolean>>({});
  const [gatewayToken, setGatewayToken] = useState('');
  const [accessToken, setAccessToken] = useState('');
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
  const rotateSecret = async (kind: 'llm' | 'api') => {
    const token = kind === 'llm' ? gatewayToken : accessToken;
    if (!token || runtime.revision === undefined || runtime.conflict) return;
    try {
      if (kind === 'llm') await secrets.writeLlmGateway(runtime.revision, token);
      else { await secrets.writeApiAccess(runtime.revision, token); setApiKey(token); }
    } finally {
      if (kind === 'llm') setGatewayToken(''); else setAccessToken('');
    }
  };
  const labels = useMemo(
    () => ({
      kronos: ['Kronos', t('strategy:runtime.kronos_description')],
      llm_committee: ['LLM', t('strategy:runtime.committee_description')],
    }),
    [t],
  );
  return (
    <PageBoundary
      loading={runtime.isLoading}
      isError={runtime.isError}
      onRetry={() => void reload()}
      errorTitle={t('configuration:strategyLoadError')}
      errorDescription={t('configuration:strategyLoadDescription')}
    >
      {runtime.document && draft ? (
        <div className="space-y-6">
          <PageHeader
            eyebrow="SIGNAL FUSION"
            title={t('strategy:title')}
            subtitle={t('strategy:subtitle')}
            actions={
              <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 font-mono text-xs text-amber-500">
                {t('strategy:runtime.revision', { revision: runtime.revision })}
              </span>
            }
          />
          <p className="text-xs text-muted-foreground">{t('strategy:runtime.updated_at', { time: runtime.updatedAt ?? '—' })}</p>
          {runtime.applyStatus !== 'applied' ? (
            <p role="alert" className="text-sm text-trade-short">
              {runtime.applyStatus === 'pending'
                ? t('configuration:apply.pending')
                : t('configuration:apply.failed', { error: t('configuration:apply.unknown') })}
            </p>
          ) : null}
          <section className="relative overflow-hidden rounded-2xl border border-border bg-card p-6">
            <div className="flex items-end justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[.18em] text-amber-500">
                  <SlidersHorizontal className="h-4 w-4" />
                  {t('strategy:mixer.label')}
                </div>
                <p className="mt-3 text-sm text-muted-foreground">{t('strategy:runtime.mixer_description')}</p>
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
                {t('strategy:runtime.invalid')}
              </p>
            ) : null}
          </section>
          <section>
            <h2 className="mb-3 font-semibold">{t('strategy:runtime.weights')}</h2>
            <div className="grid gap-4 xl:grid-cols-2">
              {draft.components.map((component, index) => {
                const label = labels[component.component_id as keyof typeof labels];
                return (<div key={component.component_id} className="rounded-xl border border-border p-3"><ComponentWeightCard
                    component={component}
                    displayName={label?.[0] ?? component.component_id}
                    description={label?.[1] ?? t('strategy:components.custom_description')}
                    accent={ACCENTS[index % ACCENTS.length] ?? '#f59e0b'}
                    onChange={updateComponent}
                  />
                  <label className="block text-xs text-muted-foreground">{t('strategy:runtime.parameter_json')}<textarea aria-label={t('strategy:runtime.parameter_label', { id: component.component_id })} value={parameterText[component.component_id] ?? JSON.stringify(component.parameters)} onChange={(event) => { const text = event.target.value; setParameterText((current) => ({ ...current, [component.component_id]: text })); try { const parameters = JSON.parse(text) as RuntimeJsonObject; if (!parameters || Array.isArray(parameters)) throw new Error(); setDraft((current) => current && ({ ...current, components: current.components.map((item) => item.component_id === component.component_id ? { ...item, parameters } : item) })); setParameterErrors((current) => ({ ...current, [component.component_id]: false })); } catch { setParameterErrors((current) => ({ ...current, [component.component_id]: true })); } }} className="mt-1 min-h-20 w-full rounded border bg-background p-2 font-mono" /></label>
                </div>);
              })}
            </div>
            <div className="mt-4 flex gap-2">
              <input
                aria-label={t('strategy:runtime.custom_id')}
                value={customId}
                onChange={(event) => setCustomId(event.target.value)}
                placeholder={t('strategy:runtime.custom_id')}
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
                {t('strategy:runtime.add')}
              </Button>
            </div>
          </section>
          <DecisionSettingsCard settings={draft} onChange={(settings) => setDraft({ ...draft, ...settings })} />
          <section className="rounded-2xl border border-border bg-card p-5">
            <h2 className="font-semibold">{t('strategy:runtime.committee_models')}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{t('strategy:runtime.committee_models_description')}</p>
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
          <section className="rounded-2xl border border-border bg-card p-5">
            <h2 className="font-semibold">{t('configuration:runtimeSecrets.title')}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{t('configuration:runtimeSecrets.hint')}</p>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <label className="text-xs text-muted-foreground">{t('configuration:runtimeSecrets.llm')}
                <input aria-label={t('configuration:runtimeSecrets.llm')} type="password" value={gatewayToken} onChange={(event) => setGatewayToken(event.target.value)} className="mt-1 h-10 w-full rounded border bg-background px-3" />
                <Button type="button" variant="outline" disabled={!gatewayToken || runtime.conflict} onClick={() => void rotateSecret('llm')}>{t('configuration:runtimeSecrets.rotateGateway')}</Button>
              </label>
              <label className="text-xs text-muted-foreground">{t('configuration:runtimeSecrets.api')}
                <input aria-label={t('configuration:runtimeSecrets.api')} type="password" value={accessToken} onChange={(event) => setAccessToken(event.target.value)} className="mt-1 h-10 w-full rounded border bg-background px-3" />
                <Button type="button" variant="outline" disabled={!accessToken || runtime.conflict} onClick={() => void rotateSecret('api')}>{t('configuration:runtimeSecrets.rotateApi')}</Button>
              </label>
            </div>
          </section>
          {runtime.conflict ? (
            <div className="flex items-center gap-3">
              <p role="alert" className="text-sm text-trade-short">
                {t('configuration:conflict')}
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => void reload()}
              >
                {t('configuration:reload')}
              </Button>
            </div>
          ) : null}
          <div className="sticky bottom-4 flex justify-end rounded-xl border border-border bg-card/95 p-3">
            <Button variant="outline" onClick={() => void reload()}>
              {t('configuration:reload')}
            </Button>
            <Button size="lg" disabled={!valid || runtime.isSaving || runtime.conflict} onClick={() => void save()}>
              <Save className="h-4 w-4" />
              {runtime.isSaving ? t('strategy:save.saving') : t('configuration:save')}
            </Button>
          </div>
          {saved ? (
            <p role="status" className="flex items-center gap-2 text-xs text-trade-long">
              <CheckCircle2 className="h-4 w-4" />
              {t('strategy:save.success')}
            </p>
          ) : null}
          {!runtime.isSaving && !runtime.conflict && !saved ? (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <CheckCircle2 className="h-4 w-4" />
              {t('strategy:runtime.draft_notice')}
            </p>
          ) : null}
        </div>
      ) : null}
    </PageBoundary>
  );
};
export default StrategyPage;
