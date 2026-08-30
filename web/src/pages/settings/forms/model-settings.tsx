import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { RuntimeDocument } from '@/types/api';
import {
  BooleanField,
  NumberField,
  StringListField,
  TextField,
  RuntimeSecretField,
  type DomainFormProps,
  type SecretFieldState,
} from '@/components/configuration/field';
import { AdvancedSection, Section } from '@/components/configuration/section';

const roles = [
  'analysis',
  'tech_agent',
  'chain_agent',
  'news_agent',
  'macro_agent',
  'debate',
  'committee_summary',
  'fallback',
] as const;
export function ModelSettings({
  value,
  onChange,
  errors = {},
  secrets,
}: DomainFormProps<RuntimeDocument['llm']> & { secrets?: SecretFieldState }) {
  const { t } = useTranslation('configuration');
  // Row identity is independent of an editable model name; removal preserves siblings.
  const [rowKeys, setRowKeys] = useState<string[]>(() => value.model_costs.map(() => crypto.randomUUID()));
  const costRows = value.model_costs.map((cost, index) => ({ cost, key: rowKeys[index] ?? `loaded-${index}` }));
  return (
    <div className="configuration-form">
      <Section title={t('forms.modelsTitle')} description={t('forms.modelsHelp')}>
        <TextField
          name="llm.base_url"
          label={t('forms.gatewayUrl')}
          value={value.base_url}
          onChange={(base_url) => onChange({ ...value, base_url })}
          error={errors['llm.base_url']}
          type="url"
        />
        {secrets ? <RuntimeSecretField kind="llm-gateway" state={secrets} /> : null}
        {roles.map((role) => (
          <TextField
            key={role}
            name={`llm.models.${role}`}
            label={t(`forms.roles.${role}`)}
            value={value.models[role]}
            onChange={(model) => onChange({ ...value, models: { ...value.models, [role]: model } })}
            error={errors[`llm.models.${role}`]}
            required
          />
        ))}
        <NumberField
          name="llm.default_temperature"
          label={t('forms.temperature')}
          help={t('forms.temperatureHelp')}
          value={value.default_temperature}
          min={0}
          max={2}
          step={0.1}
          onChange={(default_temperature) => onChange({ ...value, default_temperature })}
          error={errors['llm.default_temperature']}
        />
        <BooleanField
          name="llm.prompt_caching"
          label={t('forms.promptCaching')}
          value={value.prompt_caching}
          onChange={(prompt_caching) => onChange({ ...value, prompt_caching })}
        />
      </Section>
      <AdvancedSection title={t('forms.modelAdvanced')}>
        <NumberField
          name="llm.timeout"
          label={t('forms.requestTimeout')}
          value={value.timeout}
          min={1}
          step={1}
          onChange={(timeout) => onChange({ ...value, timeout })}
          error={errors['llm.timeout']}
        />
        <NumberField
          name="llm.models.timeout_seconds"
          label={t('forms.debateTimeout')}
          value={value.models.timeout_seconds}
          min={1}
          step={1}
          onChange={(timeout_seconds) => onChange({ ...value, models: { ...value.models, timeout_seconds } })}
          error={errors['llm.models.timeout_seconds']}
        />
        {(['max_attempts', 'retry_base_delay_s', 'retry_backoff_factor'] as const).map((key) => (
          <NumberField
            key={key}
            name={`llm.retry.${key}`}
            label={t(`forms.${key}`)}
            value={value.retry[key]}
            min={key === 'retry_base_delay_s' ? 0 : 1}
            {...(key === 'max_attempts' ? { step: 1 } : {})}
            onChange={(next) => onChange({ ...value, retry: { ...value.retry, [key]: next } })}
            error={errors[`llm.retry.${key}`]}
          />
        ))}
        <BooleanField
          name="llm.retry.retry_jitter"
          label={t('forms.retryJitter')}
          value={value.retry.retry_jitter}
          onChange={(retry_jitter) => onChange({ ...value, retry: { ...value.retry, retry_jitter } })}
        />
        <StringListField
          name="llm.streaming_models"
          label={t('forms.streamingModels')}
          help={t('forms.streamingHelp')}
          value={value.streaming_models}
          onChange={(streaming_models) => onChange({ ...value, streaming_models })}
        />
        <div className="configuration-wide">
          <h3>{t('forms.modelCosts')}</h3>
          <p className="configuration-help">{t('forms.costHelp')}</p>
          {costRows.length === 0 ? <p className="configuration-help">{t('forms.noCosts')}</p> : null}
          {costRows.map(({ cost, key }, index) => (
            <div className="configuration-cost-row" key={key}>
              <TextField
                name={`llm.model_costs.${index}.name`}
                label={t('forms.costName', { number: index + 1 })}
                required
                value={cost.name}
                onChange={(name) =>
                  onChange({
                    ...value,
                    model_costs: value.model_costs.map((row, i) => (i === index ? { ...row, name } : row)),
                  })
                }
                error={errors[`llm.model_costs.${index}.name`]}
              />
              {(['input_usd_per_mtok', 'output_usd_per_mtok'] as const).map((field) => (
                <NumberField
                  key={field}
                  name={`llm.model_costs.${index}.${field}`}
                  label={t(`forms.${field}`, { number: index + 1 })}
                  value={cost[field]}
                  min={0}
                  onChange={(next) =>
                    onChange({
                      ...value,
                      model_costs: value.model_costs.map((row, i) => (i === index ? { ...row, [field]: next } : row)),
                    })
                  }
                  error={errors[`llm.model_costs.${index}.${field}`]}
                />
              ))}
              <button
                type="button"
                className="configuration-button"
                aria-label={t('forms.removeCost', { number: index + 1 })}
                onClick={() => {
                  setRowKeys(costRows.filter((_, i) => i !== index).map((row) => row.key));
                  onChange({ ...value, model_costs: value.model_costs.filter((_, i) => i !== index) });
                }}
              >
                {t('forms.remove')}
              </button>
            </div>
          ))}
          <button
            type="button"
            className="configuration-button"
            onClick={() => {
              setRowKeys([...costRows.map((row) => row.key), crypto.randomUUID()]);
              onChange({
                ...value,
                model_costs: [...value.model_costs, { name: '', input_usd_per_mtok: 0, output_usd_per_mtok: 0 }],
              });
            }}
          >
            {t('forms.addCost')}
          </button>
        </div>
      </AdvancedSection>
    </div>
  );
}
