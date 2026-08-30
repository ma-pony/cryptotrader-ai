import { useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import type {
  ConfigurationCatalog,
  ConfigurationDraft,
  ConfigurationField,
  RuntimeConfig,
  RuntimeDocument,
  RuntimeJsonObject,
} from '@/types/api';
import { ApiError } from '@/lib/api-client';
import { bookErrors } from '@/lib/configuration-readiness';
import { focusFirstError, type FieldErrors } from '@/components/configuration/field';
import { getParameter } from '@/components/configuration/parameter-fields';
import { decodeJsonValue, RUNTIME_CONFIG_QUERY_KEY, toRuntimeDocument, useRuntimeConfig } from './use-runtime-config';

export const CONFIGURATION_SECTION_KEYS = {
  models: ['llm'],
  signals: ['signals'],
  market: ['market_data'],
  risk: ['risk', 'hitl'],
  scheduler: ['scheduler', 'triggers'],
  system: ['security', 'infrastructure', 'notifications', 'observability'],
  books: ['execution'],
} as const;
export type ConfigurationSection = keyof typeof CONFIGURATION_SECTION_KEYS;
export type ConfigurationKey = (typeof CONFIGURATION_SECTION_KEYS)[ConfigurationSection][number];
export type SaveStatus = 'saving' | 'saved' | 'failed';
type DraftDocument = ConfigurationDraft<RuntimeDocument>;
type BookSettings = Pick<DraftDocument['execution'], 'books' | 'live_order_execution_enabled'>;
type DraftOverlay = Partial<Omit<Pick<DraftDocument, ConfigurationKey>, 'execution'> & { execution: BookSettings }>;
const bookSettings = (execution: DraftDocument['execution']): BookSettings => ({
  books: execution.books,
  live_order_execution_enabled: execution.live_order_execution_enabled,
});
const ownedValue = (key: ConfigurationKey, document: DraftDocument | undefined) =>
  key === 'execution' && document ? bookSettings(document.execution) : document?.[key];
const composeDraft = (baseline: RuntimeDocument, overlay: DraftOverlay): DraftDocument => ({
  ...baseline,
  ...overlay,
  execution: { ...baseline.execution, ...overlay.execution },
});

/** Validates only the selected business section. The server still owns final CAS/validation. */
export function validateConfigurationSection(
  section: ConfigurationSection,
  document: DraftDocument,
  catalog: ConfigurationCatalog | undefined,
  message: (key: string) => string,
): FieldErrors {
  const errors: FieldErrors = {};
  if (section === 'books')
    return bookErrors(
      document.execution.books,
      document.execution.connections as RuntimeDocument['execution']['connections'],
      message,
    );
  const number = (path: string, value: unknown, min = 0, max = Infinity, integer = false) => {
    if (
      typeof value !== 'number' ||
      !Number.isFinite(value) ||
      value < min ||
      value > max ||
      (integer && !Number.isInteger(value))
    )
      errors[path] = message(integer ? 'integerInvalid' : 'numberInvalid');
  };
  const required = (path: string, value: string) => {
    if (!value.trim()) errors[path] = message('required');
  };
  const url = (path: string, value: string, protocols: string[]) => {
    if (!value.trim()) return;
    try {
      if (!protocols.includes(new URL(value).protocol)) errors[path] = message('urlInvalid');
    } catch {
      errors[path] = message('urlInvalid');
    }
  };
  const parameters = (path: string, value: RuntimeJsonObject, fields: ConfigurationField[]) => {
    for (const field of fields) {
      const current = getParameter(value, field.key) ?? decodeJsonValue(field.default_value);
      if (current === null && getParameter(value, field.key) === undefined && !field.required) continue;
      const name = `${path}.${field.key}`;
      if (field.kind === 'number' || field.kind === 'integer') {
        if (current !== null || field.required)
          number(name, current, field.minimum ?? -Infinity, field.maximum ?? Infinity, field.kind === 'integer');
      } else if (field.kind === 'text' && field.required && (typeof current !== 'string' || !current.trim()))
        errors[name] = message('required');
      else if (field.kind === 'select' && !field.options.some((option) => option.value === current))
        errors[name] = message('selectInvalid');
      else if (
        field.kind === 'string_list' &&
        (!Array.isArray(current) || current.some((item) => typeof item !== 'string' || !item.trim()))
      )
        errors[name] = message('required');
    }
  };
  if (section === 'models') {
    const llm = document.llm;
    url('llm.base_url', llm.base_url, ['https:', 'http:']);
    for (const [role, value] of Object.entries(llm.models))
      if (role !== 'timeout_seconds') required(`llm.models.${role}`, String(value));
    number('llm.default_temperature', llm.default_temperature, 0, 2);
    number('llm.timeout', llm.timeout, 1, Infinity, true);
    number('llm.models.timeout_seconds', llm.models.timeout_seconds, 1, Infinity, true);
    number('llm.retry.max_attempts', llm.retry.max_attempts, 1, Infinity, true);
    number('llm.retry.retry_base_delay_s', llm.retry.retry_base_delay_s);
    number('llm.retry.retry_backoff_factor', llm.retry.retry_backoff_factor, 1);
    const names = new Set<string>();
    llm.model_costs.forEach((cost, index) => {
      const path = `llm.model_costs.${index}`;
      required(`${path}.name`, cost.name);
      if (names.has(cost.name)) errors[`${path}.name`] = message('duplicate');
      names.add(cost.name);
      number(`${path}.input_usd_per_mtok`, cost.input_usd_per_mtok);
      number(`${path}.output_usd_per_mtok`, cost.output_usd_per_mtok);
    });
  }
  if (section === 'signals') {
    const signal = document.signals;
    number('signals.neutral_threshold', signal.neutral_threshold, 0, 1);
    number('signals.max_target_ratio', signal.max_target_ratio, 0, 1);
    number('signals.atr_stop_multiplier', signal.atr_stop_multiplier, Number.MIN_VALUE);
    number('signals.reward_ratio', signal.reward_ratio, Number.MIN_VALUE);
    let sum = 0;
    signal.components.forEach((component, index) => {
      number(`signals.components.${index}.weight`, component.weight, 0, 1);
      if (component.enabled && typeof component.weight === 'number') sum += component.weight;
      const plugin = catalog?.components.find((item) => item.id === component.component_id);
      if (plugin) parameters(`signals.components.${index}.parameters`, component.parameters, plugin.fields);
      else errors[`signals.components.${index}.enabled`] = message('catalogRequired');
    });
    if (Math.abs(sum - 1) > 1e-9) {
      const firstEnabled = signal.components.findIndex((component) => component.enabled);
      const path = firstEnabled < 0 ? 'signals.components' : `signals.components.${firstEnabled}.weight`;
      errors[path] ??= message('weightInvalid');
      errors['signals.components'] = message('weightInvalid');
    }
  }
  if (section === 'market') {
    const source = catalog?.market_sources.find((item) => item.id === document.market_data.source_id);
    if (!source) errors['market_data.source_id'] = message('catalogRequired');
    else parameters('market_data.parameters', document.market_data.parameters, source.fields);
  }
  if (section === 'risk') {
    for (const key of ['max_total_exposure_pct', 'max_single_pct', 'max_margin_used_pct'] as const)
      number(`risk.position.${key}`, document.risk.position[key], 0, 1);
    number('risk.loss.max_drawdown_pct', document.risk.loss.max_drawdown_pct, 0, 1);
    number('hitl.approval_ttl_minutes', document.hitl.approval_ttl_minutes, 1, Infinity, true);
  }
  if (section === 'scheduler') {
    number('scheduler.interval_minutes', document.scheduler.interval_minutes, 1, Infinity, true);
    number('scheduler.daily_summary_hour', document.scheduler.daily_summary_hour, 0, 23, true);
    for (const key of ['max_rules', 'ws_reconnect_max_s', 'funding_rate_poll_interval_minutes'] as const)
      number(`triggers.${key}`, document.triggers[key], 1, Infinity, true);
    if (document.scheduler.enabled && !document.scheduler.pairs.length) errors['scheduler.pairs'] = message('required');
    if (document.scheduler.pairs.some((pair) => !/^[^/\s]+\/[^/\s]+(?::[^/\s]+)?$/.test(pair)))
      errors['scheduler.pairs'] = message('pairInvalid');
  }
  if (section === 'system') {
    number('notifications.webhook_timeout', document.notifications.webhook_timeout, 1, Infinity, true);
    url('infrastructure.redis_url', document.infrastructure.redis_url, ['redis:', 'rediss:']);
    url('notifications.webhook_url', document.notifications.webhook_url, ['https:', 'http:']);
    url('observability.otlp_endpoint', document.observability.otlp_endpoint, ['https:', 'http:']);
  }
  return errors;
}

/** Keep this one owner mounted above section navigation. Secrets never enter this state. */
export function useConfigurationDraft(catalog?: ConfigurationCatalog) {
  const runtime = useRuntimeConfig();
  const client = useQueryClient();
  const { t } = useTranslation('configuration');
  const [overlay, setOverlay] = useState<DraftOverlay>({});
  const savingKeys = useRef<readonly ConfigurationKey[]>([]);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [status, setStatus] = useState<Partial<Record<ConfigurationSection, SaveStatus>>>({});
  const [failure, setFailure] = useState<string>();
  const [isReloading, setReloading] = useState(false);
  const document = runtime.document ? composeDraft(runtime.document, overlay) : undefined;
  const isDirty = (section: ConfigurationSection) =>
    CONFIGURATION_SECTION_KEYS[section].some(
      (key) =>
        overlay[key] !== undefined &&
        JSON.stringify(overlay[key]) !== JSON.stringify(ownedValue(key, runtime.document)),
    );
  const update = <K extends ConfigurationKey>(key: K, value: DraftDocument[K]) => {
    // An in-flight write may replace this baseline, so retain a restore until it settles.
    const owned = key === 'execution' ? bookSettings(value as DraftDocument['execution']) : value;
    const restored =
      !savingKeys.current.includes(key) && JSON.stringify(owned) === JSON.stringify(ownedValue(key, runtime.document));
    setOverlay((current) => {
      const next = { ...current, [key]: owned };
      if (restored) delete next[key];
      return next;
    });
    const section = (Object.keys(CONFIGURATION_SECTION_KEYS) as ConfigurationSection[]).find((candidate) =>
      (CONFIGURATION_SECTION_KEYS[candidate] as readonly string[]).includes(key),
    );
    if (section)
      setStatus((current) => {
        const next = { ...current };
        delete next[section];
        return next;
      });
    setFailure(undefined);
    setErrors((current) =>
      Object.fromEntries(Object.entries(current).filter(([path]) => path !== key && !path.startsWith(`${key}.`))),
    );
  };
  const discard = (section?: ConfigurationSection) => {
    setOverlay((current) => {
      if (!section) return {};
      const next = { ...current };
      for (const key of CONFIGURATION_SECTION_KEYS[section]) delete next[key];
      return next;
    });
    setErrors({});
    setFailure(undefined);
    setStatus((current) => {
      if (!section) return {};
      const next = { ...current };
      delete next[section];
      return next;
    });
  };
  const save = async (section: ConfigurationSection, form?: HTMLFormElement): Promise<boolean> => {
    if (!document || !runtime.document || runtime.isSaving || runtime.conflict) return false;
    const latest = client.getQueryData<RuntimeConfig>(RUNTIME_CONFIG_QUERY_KEY);
    if (!latest) return false;
    const baseline = toRuntimeDocument(latest.document);
    const currentDocument = composeDraft(baseline, overlay);
    const sectionErrors = validateConfigurationSection(section, currentDocument, catalog, (key) =>
      t(`forms.validation.${key}`),
    );
    setErrors(sectionErrors);
    setFailure(undefined);
    if (Object.keys(sectionErrors).length) {
      focusFirstError(sectionErrors, form);
      return false;
    }
    const keys = CONFIGURATION_SECTION_KEYS[section];
    // Only selected document keys are overlaid. Other drafts are deliberately not sent.
    const patch = Object.fromEntries(keys.map((key) => [key, currentDocument[key]]));
    const submitted = { ...baseline, ...patch } as RuntimeDocument;
    savingKeys.current = keys;
    setStatus((current) => ({ ...current, [section]: 'saving' }));
    try {
      await runtime.replace(submitted, latest.revision);
      setOverlay((current) => {
        const next = { ...current };
        for (const key of keys) if (current[key] === overlay[key]) delete next[key];
        return next;
      });
      setStatus((current) => (current[section] === 'saving' ? { ...current, [section]: 'saved' } : current));
      return true;
    } catch (error) {
      setStatus((current) => ({ ...current, [section]: 'failed' }));
      setFailure(t(error instanceof ApiError && error.status === 409 ? 'forms.conflictHelp' : 'forms.saveFailed'));
      if (error instanceof ApiError && error.details?.fieldErrors && typeof error.details.fieldErrors === 'object') {
        const serverErrors = Object.fromEntries(
          Object.keys(error.details.fieldErrors).map((path) => [path, t('forms.validation.serverInvalid')]),
        );
        setErrors(serverErrors);
        focusFirstError(serverErrors, form);
      }
      return false;
    } finally {
      savingKeys.current = [];
      const latest = client.getQueryData<RuntimeConfig>(RUNTIME_CONFIG_QUERY_KEY);
      const settledBaseline = latest ? toRuntimeDocument(latest.document) : undefined;
      setOverlay((current) => {
        const next = { ...current };
        for (const key of keys)
          if (JSON.stringify(current[key]) === JSON.stringify(ownedValue(key, settledBaseline))) delete next[key];
        return next;
      });
    }
  };
  const reload = async () => {
    setReloading(true);
    setFailure(undefined);
    try {
      const result = await runtime.reload();
      if (result.error) setFailure(t('forms.reloadFailed'));
      return result;
    } finally {
      setReloading(false);
    }
  };
  return {
    ...runtime,
    document,
    baseline: runtime.document,
    update,
    discard,
    save,
    reload,
    isReloading,
    isDirty,
    dirty: (Object.keys(CONFIGURATION_SECTION_KEYS) as ConfigurationSection[]).some(isDirty),
    errors,
    status,
    failure,
  };
}
