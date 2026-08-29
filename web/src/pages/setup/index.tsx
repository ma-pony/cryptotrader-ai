import { ArrowRight, CheckCircle2, Plus } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { decodeEntries, useRuntimeConfig } from '@/hooks/use-runtime-config';
import { useRuntimeSecrets } from '@/hooks/use-runtime-secrets';
import type { RuntimeConfig, RuntimeDocument, RuntimeJsonObject } from '@/types/api';
import { BookForm, newBook, validateBooks } from '@/pages/settings/execution-books/book-form';
import { VenueForm } from '@/pages/settings/venues/venue-form';
import { useSettingsStore } from '@/stores/use-settings-store';

type DraftConnection = RuntimeDocument['execution']['connections'][number];
type ResponseConnection = RuntimeConfig['document']['execution']['connections'][number];

const hasExactFiniteNumbers = (value: unknown, fields: readonly string[]) =>
  typeof value === 'object' &&
  value !== null &&
  !Array.isArray(value) &&
  Object.keys(value).length === fields.length &&
  fields.every(
    (field) =>
      Object.prototype.hasOwnProperty.call(value, field) &&
      typeof (value as Record<string, unknown>)[field] === 'number' &&
      Number.isFinite((value as Record<string, number>)[field]),
  );

/** The risk editor accepts the whole strict runtime section, never an arbitrary JSON object. */
export const validateRiskSection = (value: unknown): RuntimeDocument['risk'] | undefined => {
  if (
    typeof value !== 'object' || value === null || Array.isArray(value) ||
    typeof (value as Record<string, unknown>).max_stop_loss_pct !== 'number' ||
    !Number.isFinite((value as Record<string, number>).max_stop_loss_pct) ||
    !hasExactFiniteNumbers((value as Record<string, unknown>).position, [
      'max_single_pct', 'max_total_exposure_pct', 'max_margin_used_pct', 'max_correlated_positions', 'max_same_direction_positions',
    ]) ||
    !hasExactFiniteNumbers((value as Record<string, unknown>).loss, [
      'max_daily_loss_pct', 'max_drawdown_pct', 'max_cvar_95', 'cvar_min_returns',
    ]) ||
    !hasExactFiniteNumbers((value as Record<string, unknown>).cooldown, ['same_pair_minutes', 'post_loss_minutes']) ||
    !hasExactFiniteNumbers((value as Record<string, unknown>).volatility, ['flash_crash_threshold', 'funding_rate_threshold', 'flash_crash_lookback']) ||
    !hasExactFiniteNumbers((value as Record<string, unknown>).exchange, ['max_api_latency_ms', 'health_check_interval_s']) ||
    !hasExactFiniteNumbers((value as Record<string, unknown>).rate_limit, ['max_trades_per_hour', 'max_trades_per_day']) ||
    Object.keys(value as Record<string, unknown>).length !== 7
  ) return undefined;
  return value as RuntimeDocument['risk'];
};

export const validateLlmAdvanced = (value: unknown): Pick<RuntimeDocument['llm'], 'streaming_models' | 'retry' | 'model_costs'> & { timeout_seconds: number } | undefined => {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined;
  const record = value as Record<string, unknown>;
  const timeoutSeconds: unknown = record.timeout_seconds;
  if (Object.keys(record).length !== 4 || !Array.isArray(record.streaming_models) || !record.streaming_models.every((item) => typeof item === 'string') || typeof timeoutSeconds !== 'number' || !Number.isInteger(timeoutSeconds)) return undefined;
  const retry = record.retry;
  if (typeof retry !== 'object' || retry === null || Array.isArray(retry)) return undefined;
  const retryRecord = retry as Record<string, unknown>;
  const maxAttempts: unknown = retryRecord.max_attempts;
  if (Object.keys(retry).length !== 4 || typeof maxAttempts !== 'number' || !Number.isInteger(maxAttempts) || !['retry_base_delay_s', 'retry_backoff_factor'].every((key) => { const candidate: unknown = retryRecord[key]; return typeof candidate === 'number' && Number.isFinite(candidate); }) || typeof retryRecord.retry_jitter !== 'boolean') return undefined;
  const costs: unknown = record.model_costs;
  if (!Array.isArray(costs) || !costs.every((cost: unknown) => { if (typeof cost !== 'object' || cost === null || Array.isArray(cost)) return false; const costRecord = cost as Record<string, unknown>; const input: unknown = costRecord.input_usd_per_mtok; const output: unknown = costRecord.output_usd_per_mtok; return Object.keys(cost).length === 3 && typeof costRecord.name === 'string' && typeof input === 'number' && Number.isFinite(input) && typeof output === 'number' && Number.isFinite(output); })) return undefined;
  return value as Pick<RuntimeDocument['llm'], 'streaming_models' | 'retry' | 'model_costs'> & { timeout_seconds: number };
};

export const testFingerprint = (connection: DraftConnection, credentialUpdatedAt?: string | null) =>
  JSON.stringify({
    id: connection.id,
    label: connection.label,
    adapter: connection.adapter_id,
    environment: connection.environment,
    enabled: connection.enabled,
    leverage: connection.leverage,
    margin: connection.margin_mode,
    parameters: connection.parameters,
    credentialUpdatedAt: credentialUpdatedAt ?? null,
  });

const asDraftConnection = (connection: ResponseConnection): DraftConnection => {
  const { credential_configured: _configured, credential_updated_at: _updatedAt, parameters, ...rest } = connection;
  return { ...rest, parameters: decodeEntries(parameters) } as DraftConnection;
};

const SetupEditor = ({
  initialDocument,
  onReload,
}: {
  initialDocument: RuntimeDocument;
  onReload: () => Promise<void>;
}) => {
  const { t } = useTranslation('configuration');
  const steps = t('steps', { returnObjects: true }) as string[];
  const runtime = useRuntimeConfig();
  const secrets = useRuntimeSecrets();
  const setApiKey = useSettingsStore((state) => state.setApiKey);
  const [step, setStep] = useState(0);
  const [draft, setDraft] = useState(initialDocument);
  const [tested, setTested] = useState<Record<string, string>>({});
  const [marketParameters, setMarketParameters] = useState(
    JSON.stringify(initialDocument.market_data.parameters, null, 2),
  );
  const [marketError, setMarketError] = useState('');
  const [marketDirty, setMarketDirty] = useState(false);
  const [customId, setCustomId] = useState('');
  const [customParameters, setCustomParameters] = useState('{}');
  const [signalError, setSignalError] = useState('');
  const [componentParameterText, setComponentParameterText] = useState<Record<string, string>>({});
  const [componentParameterErrors, setComponentParameterErrors] = useState<Record<string, boolean>>({});
  const [riskText, setRiskText] = useState(JSON.stringify(initialDocument.risk, null, 2));
  const [riskError, setRiskError] = useState('');
  const [riskDirty, setRiskDirty] = useState(false);
  const [llmAdvanced, setLlmAdvanced] = useState(JSON.stringify({ streaming_models: initialDocument.llm.streaming_models, retry: initialDocument.llm.retry, model_costs: initialDocument.llm.model_costs, timeout_seconds: initialDocument.llm.models.timeout_seconds }, null, 2));
  const [llmError, setLlmError] = useState('');
  const [llmAdvancedDirty, setLlmAdvancedDirty] = useState(false);
  const [activationError, setActivationError] = useState('');
  const [gatewayToken, setGatewayToken] = useState('');
  const [accessToken, setAccessToken] = useState('');

  const enabledConnections = draft.execution.connections.filter((connection) => connection.enabled);
  const bookErrors = validateBooks(draft.execution.books, draft.execution.connections);
  const enabled = draft.signals.components.filter((component) => component.enabled);
  const signalValid =
    enabled.length > 0 &&
    Math.abs(enabled.reduce((sum, component) => sum + component.weight, 0) - 1) < 1e-9 &&
    draft.signals.neutral_threshold >= 0 &&
    draft.signals.neutral_threshold < 1 &&
    draft.signals.max_target_ratio > 0 &&
    draft.signals.max_target_ratio <= 1 &&
    draft.signals.atr_stop_multiplier > 0 &&
    draft.signals.reward_ratio > 0;
  const connectionsTested =
    enabledConnections.length > 0 &&
    enabledConnections.every(
      (connection) =>
        tested[connection.id] === testFingerprint(connection, runtime.credentialStates[connection.id]?.updatedAt),
    );
  const llmCommitteeEnabled = draft.signals.components.some(
    (component) => component.enabled && component.component_id === 'llm_committee',
  );
  const credentialReady =
    (!draft.security.enabled || runtime.secretStates.apiAccess.configured) &&
    (!llmCommitteeEnabled || runtime.secretStates.llmGateway.configured);
  const ready =
    signalValid &&
    draft.execution.books.some((book) => book.enabled) &&
    bookErrors.length === 0 &&
    connectionsTested &&
    !marketDirty &&
    !marketError &&
    !riskDirty &&
    !riskError &&
    !llmAdvancedDirty &&
    !llmError &&
    !Object.values(componentParameterErrors).some(Boolean) &&
    credentialReady &&
    !runtime.conflict;

  const applyMarketParameters = () => {
    try {
      const parameters = JSON.parse(marketParameters) as RuntimeJsonObject;
      if (!parameters || Array.isArray(parameters) || typeof parameters !== 'object')
        throw new Error('object expected');
      setDraft((current) => ({ ...current, market_data: { ...current.market_data, parameters } }));
      setMarketError('');
      setMarketDirty(false);
    } catch {
      setMarketError(t('wizard.marketInvalid'));
    }
  };
  const activate = async () => {
    if (!ready) return;
    try {
      setActivationError('');
      await runtime.replace({ ...draft, system: { ...draft.system, active: true } });
    } catch {
      setActivationError(t('wizard.activationFailed'));
    }
  };
  const saveGatewayToken = async () => {
    if (!gatewayToken || runtime.revision === undefined || runtime.conflict) return;
    try {
      await secrets.writeLlmGateway(runtime.revision, gatewayToken);
    } catch {
      setActivationError(t('wizard.activationFailed'));
    } finally {
      setGatewayToken('');
    }
  };
  const saveAccessToken = async () => {
    if (!accessToken || runtime.revision === undefined || runtime.conflict) return;
    try {
      await secrets.writeApiAccess(runtime.revision, accessToken);
      setApiKey(accessToken);
    } catch {
      setActivationError(t('wizard.activationFailed'));
    } finally {
      setAccessToken('');
    }
  };
  const replaceConnection = (saved: ResponseConnection) =>
    setDraft((current) => ({
      ...current,
      execution: {
        ...current.execution,
        connections: current.execution.connections.some((connection) => connection.id === saved.id)
          ? current.execution.connections.map((connection) =>
              connection.id === saved.id ? asDraftConnection(saved) : connection,
            )
          : [...current.execution.connections, asDraftConnection(saved)],
      },
    }));

  const content = (() => {
    if (step === 0) {
      const keys = [
        'analysis',
        'fallback',
        'tech_agent',
        'chain_agent',
        'news_agent',
        'macro_agent',
        'debate',
        'committee_summary',
      ] as const;
      return (
        <div className="grid gap-3 md:grid-cols-2">
          {keys.map((key) => (
            <label key={key} className="text-sm text-muted-foreground">
              {key}
              <input
                aria-label={`LLM ${key}`}
                value={draft.llm.models[key]}
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    llm: { ...current.llm, models: { ...current.llm.models, [key]: event.target.value } },
                  }))
                }
                className="mt-1 h-10 w-full rounded border bg-background px-3 font-mono"
              />
            </label>
          ))}
          <label>Base URL<input aria-label="LLM base URL" value={draft.llm.base_url} onChange={(event) => setDraft((current) => ({ ...current, llm: { ...current.llm, base_url: event.target.value } }))} className="mt-1 h-10 w-full rounded border bg-background px-3" /></label>
          <label>{t('runtimeSecrets.llm')}<input aria-label={t('runtimeSecrets.llm')} type="password" value={gatewayToken} onChange={(event) => setGatewayToken(event.target.value)} className="mt-1 h-10 w-full rounded border bg-background px-3" /></label>
          <Button type="button" variant="outline" disabled={!gatewayToken || runtime.conflict} onClick={() => void saveGatewayToken()}>{runtime.secretStates.llmGateway.configured ? t('runtimeSecrets.rotateGateway') : t('runtimeSecrets.saveGateway')}</Button>
          <label>{t('wizard.defaultTemperature')}<input aria-label={t('wizard.defaultTemperature')} type="number" value={draft.llm.default_temperature} onChange={(event) => setDraft((current) => ({ ...current, llm: { ...current.llm, default_temperature: Number(event.target.value) } }))} className="mt-1 h-10 w-full rounded border bg-background px-3" /></label>
          <label>{t('wizard.timeout')}<input aria-label={t('wizard.timeout')} type="number" value={draft.llm.timeout} onChange={(event) => setDraft((current) => ({ ...current, llm: { ...current.llm, timeout: Number(event.target.value) } }))} className="mt-1 h-10 w-full rounded border bg-background px-3" /></label>
          <label className="flex items-center gap-2"><input aria-label="LLM prompt caching" type="checkbox" checked={draft.llm.prompt_caching} onChange={(event) => setDraft((current) => ({ ...current, llm: { ...current.llm, prompt_caching: event.target.checked } }))} />Prompt caching</label>
          <label className="md:col-span-2">{t('wizard.advanced')}<textarea aria-label={t('wizard.advanced')} value={llmAdvanced} onChange={(event) => { setLlmAdvanced(event.target.value); setLlmAdvancedDirty(true); }} className="mt-1 min-h-32 w-full rounded border bg-background p-3 font-mono text-xs" /></label>
          <Button type="button" variant="outline" onClick={() => { try { const value = validateLlmAdvanced(JSON.parse(llmAdvanced)); if (!value) throw new Error(); setDraft((current) => ({ ...current, llm: { ...current.llm, streaming_models: value.streaming_models, retry: value.retry, model_costs: value.model_costs, models: { ...current.llm.models, timeout_seconds: value.timeout_seconds } } })); setLlmError(''); setLlmAdvancedDirty(false); } catch { setLlmError(t('wizard.advancedInvalid')); setLlmAdvancedDirty(true); } }}>{t('wizard.applyAdvanced')}</Button>
          {llmError ? <p role="alert" className="text-sm text-trade-short">{llmError}</p> : null}
        </div>
      );
    }
    if (step === 1)
      return (
        <div className="space-y-3">
          <p className="text-sm text-muted-foreground">{t('wizard.signalHint')}</p>
          {draft.signals.components.map((component, index) => (
            <div key={component.component_id} className="grid grid-cols-[1fr_auto_auto] items-center gap-3">
              <span>{component.component_id}</span>
              <input
                aria-label={t('wizard.enabled', { name: component.component_id })}
                type="checkbox"
                checked={component.enabled}
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    signals: {
                      ...current.signals,
                      components: current.signals.components.map((item, itemIndex) =>
                        itemIndex === index ? { ...item, enabled: event.target.checked } : item,
                      ),
                    },
                  }))
                }
              />
              <label className="col-span-3 block text-xs text-muted-foreground">{t('wizard.parameters', { name: component.component_id })}<textarea aria-label={t('wizard.parameters', { name: component.component_id })} value={componentParameterText[component.component_id] ?? JSON.stringify(component.parameters)} onChange={(event) => { const text = event.target.value; setComponentParameterText((current) => ({ ...current, [component.component_id]: text })); try { const parameters = JSON.parse(text) as RuntimeJsonObject; if (!parameters || Array.isArray(parameters)) throw new Error(); setDraft((current) => ({ ...current, signals: { ...current.signals, components: current.signals.components.map((item) => item.component_id === component.component_id ? { ...item, parameters } : item) } })); setComponentParameterErrors((current) => ({ ...current, [component.component_id]: false })); } catch { setComponentParameterErrors((current) => ({ ...current, [component.component_id]: true })); } }} className="mt-1 min-h-20 w-full rounded border bg-background p-2 font-mono" /></label>
              <input
                aria-label={t('wizardWeight', { name: component.component_id })}
                className="h-9 w-24 rounded border bg-background px-2"
                type="number"
                value={component.weight * 100}
                onChange={(event) =>
                  setDraft((current) => ({
                    ...current,
                    signals: {
                      ...current.signals,
                      components: current.signals.components.map((item, itemIndex) =>
                        itemIndex === index ? { ...item, weight: Number(event.target.value) / 100 } : item,
                      ),
                    },
                  }))
                }
              />
            </div>
          ))}
          <div className="grid gap-2 md:grid-cols-[1fr_2fr_auto]">
            <input aria-label={t('wizard.customId')} value={customId} onChange={(event) => setCustomId(event.target.value)} placeholder={t('wizard.customId')} className="h-10 rounded border bg-background px-3" />
            <input aria-label={t('wizard.customParameters')} value={customParameters} onChange={(event) => setCustomParameters(event.target.value)} placeholder="{}" className="h-10 rounded border bg-background px-3 font-mono" />
            <Button type="button" variant="outline" onClick={() => {
              try {
                const id = customId.trim(); const parameters = JSON.parse(customParameters) as RuntimeJsonObject;
                if (!id || draft.signals.components.some((component) => component.component_id === id) || !parameters || Array.isArray(parameters)) throw new Error();
                setDraft((current) => ({ ...current, signals: { ...current.signals, components: [...current.signals.components, { component_id: id, enabled: false, weight: 0, parameters }] } }));
                setCustomId(''); setCustomParameters('{}'); setSignalError('');
              } catch { setSignalError(t('wizard.customInvalid')); }
            }}><Plus className="h-4 w-4" />{t('wizard.addCustom')}</Button>
          </div>
          {signalError ? <p role="alert" className="text-sm text-trade-short">{signalError}</p> : null}
          <label>
            {t('wizard.neutral')}
            <input
              aria-label={t('wizard.neutral')}
              type="number"
              value={draft.signals.neutral_threshold}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  signals: { ...current.signals, neutral_threshold: Number(event.target.value) },
                }))
              }
              className="ml-2 h-9 rounded border bg-background px-2"
            />
          </label>
          <label>{t('wizard.maxTarget')}<input aria-label={t('wizard.maxTarget')} type="number" value={draft.signals.max_target_ratio} onChange={(event) => setDraft((current) => ({ ...current, signals: { ...current.signals, max_target_ratio: Number(event.target.value) } }))} className="ml-2 h-9 rounded border bg-background px-2" /></label>
          <label>{t('wizard.atrStop')}<input aria-label={t('wizard.atrStop')} type="number" value={draft.signals.atr_stop_multiplier} onChange={(event) => setDraft((current) => ({ ...current, signals: { ...current.signals, atr_stop_multiplier: Number(event.target.value) } }))} className="ml-2 h-9 rounded border bg-background px-2" /></label>
          <label>{t('wizard.reward')}<input aria-label={t('wizard.reward')} type="number" value={draft.signals.reward_ratio} onChange={(event) => setDraft((current) => ({ ...current, signals: { ...current.signals, reward_ratio: Number(event.target.value) } }))} className="ml-2 h-9 rounded border bg-background px-2" /></label>
          <label className="flex items-center gap-2"><input aria-label={t('wizard.signalHitl')} type="checkbox" checked={draft.signals.hitl_required} onChange={(event) => setDraft((current) => ({ ...current, signals: { ...current.signals, hitl_required: event.target.checked } }))} />{t('wizard.signalHitl')}</label>
        </div>
      );
    if (step === 2)
      return (
        <div className="space-y-3">
          <label className="block text-sm">
            {t('wizard.marketSource')}
            <input
              aria-label={t('wizard.marketSource')}
              value={draft.market_data.source_id}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  market_data: { ...current.market_data, source_id: event.target.value },
                }))
              }
              className="mt-1 h-10 w-full rounded border bg-background px-3"
            />
          </label>
          <label className="block text-sm">
            {t('wizard.jsonParameters')}
            <textarea
              aria-label={t('wizard.jsonParameters')}
              value={marketParameters}
              onChange={(event) => { setMarketParameters(event.target.value); setMarketDirty(true); }}
              className="mt-1 min-h-32 w-full rounded border bg-background p-3 font-mono text-xs"
            />
          </label>
          <Button type="button" variant="outline" onClick={applyMarketParameters}>
            {t('wizard.applyMarket')}
          </Button>
          {marketError ? (
            <p role="alert" className="text-sm text-trade-short">
              {marketError}
            </p>
          ) : null}
        </div>
      );
    if (step === 3)
      return (
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">
            {t('wizard.connectionHint')}
          </p>
          {draft.execution.connections.map((connection) => (
            <VenueForm
              key={connection.id}
              revision={runtime.revision ?? 0}
              connection={connection}
              onSaved={replaceConnection}
              writeBlocked={runtime.conflict}
              tested={(id) => {
                const current = draft.execution.connections.find((item) => item.id === id);
                if (current)
                  setTested((old) => ({
                    ...old,
                    [id]: testFingerprint(current, runtime.credentialStates[id]?.updatedAt),
                  }));
              }}
            />
          ))}
          <VenueForm revision={runtime.revision ?? 0} onSaved={replaceConnection} writeBlocked={runtime.conflict} />
        </div>
      );
    if (step === 4)
      return (
        <div className="space-y-3">
          <section className="rounded border border-amber-500/50 bg-amber-500/5 p-3">
            <p className="text-sm text-muted-foreground">{t('liveWrite.warning')}</p>
            <label className="mt-2 flex items-center gap-2 text-sm font-medium"><input aria-label={t('liveWrite.enable')} type="checkbox" checked={draft.execution.live_order_execution_enabled} onChange={(event) => setDraft((current) => ({ ...current, execution: { ...current.execution, live_order_execution_enabled: event.target.checked } }))} />{t('liveWrite.enable')}</label>
          </section>
          {draft.execution.books.map((book, index) => (
            <BookForm
              key={`${book.id}-${index}`}
              book={book}
              connections={draft.execution.connections}
              onChange={(next) =>
                setDraft((current) => ({
                  ...current,
                  execution: {
                    ...current.execution,
                    books: current.execution.books.map((item, itemIndex) => (itemIndex === index ? next : item)),
                  },
                }))
              }
              onRemove={() =>
                setDraft((current) => ({
                  ...current,
                  execution: {
                    ...current.execution,
                    books: current.execution.books.filter((_, itemIndex) => itemIndex !== index),
                  },
                }))
              }
            />
          ))}
          <Button
            type="button"
            variant="outline"
            onClick={() =>
              setDraft((current) => ({
                ...current,
                execution: { ...current.execution, books: [...current.execution.books, newBook()] },
              }))
            }
          >
            {t('addBook')}
          </Button>
          {bookErrors.map((error, index) => (
            <p key={`${error.code}-${index}`} role="alert" className="text-sm text-trade-short">
              {t(`book.errors.${error.code}`, error.params ?? {})}
            </p>
          ))}
        </div>
      );
    if (step === 5)
      return (
        <div className="grid gap-3 md:grid-cols-2">
          <label>
            {t('wizard.riskStop')}
            <input
              aria-label={t('wizard.riskStop')}
              type="number"
              value={draft.risk.max_stop_loss_pct}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  risk: { ...current.risk, max_stop_loss_pct: Number(event.target.value) },
                }))
              }
              className="ml-2 h-9 rounded border bg-background px-2"
            />
          </label>
          <label>
            {t('wizard.approvalTtl')}
            <input
              aria-label={t('wizard.approvalTtl')}
              type="number"
              value={draft.hitl.approval_ttl_minutes}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  hitl: { ...current.hitl, approval_ttl_minutes: Number(event.target.value) },
                }))
              }
              className="ml-2 h-9 rounded border bg-background px-2"
            />
          </label>
          <label className="md:col-span-2">{t('wizard.riskJson')}<textarea aria-label={t('wizard.riskJson')} value={riskText} onChange={(event) => { setRiskText(event.target.value); setRiskDirty(true); }} className="mt-1 min-h-36 w-full rounded border bg-background p-3 font-mono text-xs" /></label>
          <Button type="button" variant="outline" onClick={() => { try { const risk = validateRiskSection(JSON.parse(riskText)); if (!risk) throw new Error(); setDraft((current) => ({ ...current, risk })); setRiskError(''); setRiskDirty(false); } catch { setRiskError(t('wizard.riskInvalid')); } }}>{t('wizard.applyRisk')}</Button>
          {riskError ? <p role="alert" className="text-sm text-trade-short">{riskError}</p> : null}
        </div>
      );
    if (step === 6)
      return (
        <div className="grid gap-3 md:grid-cols-2">
          <label className="flex items-center gap-2">
            <input
              aria-label={t('wizard.schedulerEnabled')}
              type="checkbox"
              checked={draft.scheduler.enabled}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  scheduler: { ...current.scheduler, enabled: event.target.checked },
                }))
              }
            />
            {t('wizard.schedulerEnabled')}
          </label>
          <label>
            {t('wizard.schedulerInterval')}
            <input
              aria-label={t('wizard.schedulerInterval')}
              type="number"
              value={draft.scheduler.interval_minutes}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  scheduler: { ...current.scheduler, interval_minutes: Number(event.target.value) },
                }))
              }
              className="ml-2 h-9 rounded border bg-background px-2"
            />
          </label>
          <label>{t('wizard.schedulerPairs')}<input aria-label={t('wizard.schedulerPairs')} value={draft.scheduler.pairs.join(',')} onChange={(event) => setDraft((current) => ({ ...current, scheduler: { ...current.scheduler, pairs: event.target.value.split(',').map((pair) => pair.trim()).filter(Boolean) } }))} className="ml-2 h-9 rounded border bg-background px-2" /></label>
          <label>{t('wizard.summaryHour')}<input aria-label={t('wizard.summaryHour')} type="number" value={draft.scheduler.daily_summary_hour} onChange={(event) => setDraft((current) => ({ ...current, scheduler: { ...current.scheduler, daily_summary_hour: Number(event.target.value) } }))} className="ml-2 h-9 rounded border bg-background px-2" /></label>
          <label className="md:col-span-2">{t('liveWrite.redis')}<input aria-label={t('liveWrite.redis')} value={draft.infrastructure.redis_url} onChange={(event) => setDraft((current) => ({ ...current, infrastructure: { ...current.infrastructure, redis_url: event.target.value } }))} className="ml-2 h-9 w-full rounded border bg-background px-2 font-mono" /></label>
        </div>
      );
    return (
      <div className="space-y-3">
        {runtime.applyStatus !== 'applied' ? (
          <p role="alert" className="text-sm text-trade-short">
            {runtime.applyStatus === 'pending'
              ? t('apply.pending')
              : t('apply.failed', { error: t('apply.unknown') })}
          </p>
        ) : null}
        <p className={ready ? 'text-trade-long' : 'text-trade-short'}>
          {ready
            ? t('wizard.ready')
            : t('wizard.notReady')}
        </p>
        <Button disabled={!ready || runtime.isSaving} onClick={() => void activate()}>
          <CheckCircle2 className="h-4 w-4" />
          {t('activate')}
        </Button>
        {!credentialReady ? <p role="alert" className="text-sm text-trade-short">{t('activationRequired')}</p> : null}
        <label className="block text-sm">{t('runtimeSecrets.api')}
          <input aria-label={t('runtimeSecrets.api')} type="password" value={accessToken} onChange={(event) => setAccessToken(event.target.value)} className="mt-1 h-10 w-full rounded border bg-background px-3" />
        </label>
        <label className="flex items-center gap-2"><input aria-label={t('runtimeSecrets.enable')} type="checkbox" checked={draft.security.enabled} onChange={(event) => setDraft((current) => ({ ...current, security: { enabled: event.target.checked } }))} />{t('runtimeSecrets.enable')}</label>
        <Button type="button" variant="outline" disabled={!accessToken || runtime.conflict} onClick={() => void saveAccessToken()}>{runtime.secretStates.apiAccess.configured ? t('runtimeSecrets.rotateApi') : t('runtimeSecrets.saveApi')}</Button>
        {activationError ? (
          <p role="alert" className="text-sm text-trade-short">
            {activationError}
          </p>
        ) : null}
      </div>
    );
  })();

  return (
    <main className="min-h-screen bg-background p-6 text-foreground">
      <div className="mx-auto max-w-5xl">
        <header className="border-b border-amber-500/30 pb-6">
          <p className="font-mono text-xs tracking-[.24em] text-amber-500">COMMISSIONING / REV {runtime.revision}</p>
          <h1 className="mt-3 text-3xl font-semibold">{t('commissioning')}</h1>
          <p className="mt-2 text-muted-foreground">{t('setupIntro')}</p>
        </header>
        <div className="mt-8 grid gap-6 lg:grid-cols-[230px_1fr]">
          <ol className="border-l border-amber-500/30">
            {steps.map((label, index) => (
              <li
                key={label}
                className={`relative py-3 pl-5 text-sm ${index === step ? 'font-semibold text-amber-500' : index < step ? 'text-trade-long' : 'text-muted-foreground'}`}
              >
                <span className="absolute -left-1.5 top-4 h-3 w-3 rounded-full bg-current" />
                {index + 1}. {label}
              </li>
            ))}
          </ol>
          <section className="rounded-2xl border border-border bg-card p-6">
            <p className="font-mono text-xs text-amber-500">{t('stage', { number: String(step + 1).padStart(2, '0') })}</p>
            <h2 className="mt-2 text-xl font-semibold">{steps[step]}</h2>
            <div className="mt-4">{content}</div>
            {step < 7 ? (
              <Button className="mt-6" onClick={() => setStep((current) => Math.min(current + 1, 7))}>
                {t('nextStage')} <ArrowRight className="h-4 w-4" />
              </Button>
            ) : null}
            {runtime.conflict ? (
              <div className="mt-4 flex gap-2">
                <p role="alert" className="text-sm text-trade-short">
                  {t('conflict')}
                </p>
                <Button size="sm" variant="outline" onClick={() => void onReload()}>
                  {t('reload')}
                </Button>
              </div>
            ) : null}
          </section>
        </div>
      </div>
    </main>
  );
};

const SetupPage = () => {
  const { t } = useTranslation('configuration');
  const runtime = useRuntimeConfig();
  const [editorVersion, setEditorVersion] = useState(0);
  const reloadEditor = async () => {
    const result = await runtime.reload();
    if (result.isSuccess && !result.error) setEditorVersion((version) => version + 1);
  };
  if (runtime.isLoading)
    return <div className="grid min-h-screen place-items-center text-amber-500">LOADING CONFIG…</div>;
  if (runtime.isError || !runtime.document || runtime.revision === undefined)
    return (
      <main className="grid min-h-screen place-items-center">
        <div>
          <h1>{t('loadError')}</h1>
          <Button onClick={() => void reloadEditor()}>{t('retry')}</Button>
        </div>
      </main>
    );
  return <SetupEditor key={editorVersion} initialDocument={runtime.document} onReload={reloadEditor} />;
};

export default SetupPage;
