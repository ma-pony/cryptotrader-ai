import { useTranslation } from 'react-i18next';
import type { ConfigurationCatalog, RuntimeDocument } from '@/types/api';
import {
  BooleanField,
  ChoiceField,
  NumberField,
  TextField,
  type DomainFormProps,
} from '@/components/configuration/field';
import { ParameterFields, parameterDefaults } from '@/components/configuration/parameter-fields';
import { Section } from '@/components/configuration/section';

export function SignalSettings({
  value,
  onChange,
  errors = {},
  catalog,
  componentId,
}: DomainFormProps<RuntimeDocument['signals']> & { catalog: ConfigurationCatalog; componentId?: string }) {
  const { t, i18n } = useTranslation('configuration');
  const locale = i18n.language.startsWith('zh') ? 'zh_CN' : 'en_US';
  const total = value.components
    .filter((component) => component.enabled)
    .reduce((sum, component) => sum + (component.weight === '' ? 0 : component.weight * 100), 0);
  const available = catalog.components.filter(
    (plugin) => !value.components.some((component) => component.component_id === plugin.id),
  );
  return (
    <div className="configuration-form">
      <Section title={t('forms.signalsTitle')} description={t('forms.signalHelp')}>
        <div className="configuration-wide">
          <ChoiceField
            name="signals.components"
            label={t('forms.addSignal')}
            value=""
            placeholder={t('forms.select')}
            error={value.components.some((component) => component.enabled) ? undefined : errors['signals.components']}
            options={available.map((plugin) => ({ value: plugin.id, label: plugin.label[locale] }))}
            onChange={(id) => {
              const plugin = available.find((item) => item.id === id);
              if (plugin)
                onChange({
                  ...value,
                  components: [
                    ...value.components,
                    { component_id: id, enabled: true, weight: 0, parameters: parameterDefaults(plugin.fields) },
                  ],
                });
            }}
          />
          <p role="status" className={Math.abs(total - 100) > 1e-7 ? 'configuration-help' : 'configuration-total'}>
            {t('forms.weightTotal', { total: Number(total.toFixed(6)), remaining: Number((100 - total).toFixed(6)) })}
          </p>
          {errors['signals.components'] ? (
            <p role="alert" className="configuration-error">
              {errors['signals.components']}
            </p>
          ) : null}
          {value.components.length === 0 ? <p className="configuration-help">{t('forms.noSignals')}</p> : null}
        </div>
        {value.components.map((component, index) => {
          if (componentId && component.component_id !== componentId) return null;
          const plugin = catalog.components.find((item) => item.id === component.component_id);
          const label = plugin?.label[locale] ?? component.component_id;
          const update = (patch: Partial<typeof component>) =>
            onChange({
              ...value,
              components: value.components.map((item, i) => (i === index ? { ...item, ...patch } : item)),
            });
          return (
            <div className="configuration-plugin configuration-wide" key={component.component_id}>
              <h3>{label}</h3>
              <p className="configuration-help">{plugin?.description[locale] ?? t('forms.pluginUnavailable')}</p>
              <div className="configuration-grid">
                <BooleanField
                  name={`signals.components.${index}.enabled`}
                  label={t('forms.enabledName', { name: label })}
                  value={component.enabled}
                  onChange={(enabled) => update({ enabled })}
                  error={errors[`signals.components.${index}.enabled`]}
                />
                <NumberField
                  name={`signals.components.${index}.weight`}
                  label={t('forms.weightName', { name: label })}
                  value={component.weight}
                  min={0}
                  max={100}
                  percent
                  onChange={(weight) => update({ weight })}
                  error={errors[`signals.components.${index}.weight`]}
                />
              </div>
              {plugin ? (
                <ParameterFields
                  fields={plugin.fields}
                  value={component.parameters}
                  onChange={(parameters) => update({ parameters })}
                  errors={errors}
                  idPrefix={`signals.components.${index}.parameters`}
                />
              ) : null}
              <button
                type="button"
                className="configuration-button"
                aria-label={t('forms.removeItem', { name: label })}
                onClick={() => onChange({ ...value, components: value.components.filter((_, i) => i !== index) })}
              >
                {t('forms.remove')}
              </button>
            </div>
          );
        })}
        <TextField
          name="signals.evaluation_interval"
          label="信号评估周期"
          help="留空跟随全局参考周期，例如 2h。修改仅影响后续运行。"
          value={value.evaluation_interval ?? ''}
          onChange={(interval) => onChange({ ...value, evaluation_interval: interval || null })}
          error={errors['signals.evaluation_interval']}
        />
        <NumberField
          name="signals.neutral_threshold"
          label={t('forms.neutral')}
          help={t('forms.neutralHelp')}
          value={value.neutral_threshold}
          min={0}
          max={1}
          onChange={(neutral_threshold) => onChange({ ...value, neutral_threshold })}
          error={errors['signals.neutral_threshold']}
        />
        <NumberField
          name="signals.max_target_ratio"
          label={t('forms.maxTarget')}
          value={value.max_target_ratio}
          min={0}
          max={100}
          percent
          onChange={(max_target_ratio) => onChange({ ...value, max_target_ratio })}
          error={errors['signals.max_target_ratio']}
        />
        <NumberField
          name="signals.atr_stop_multiplier"
          label={t('forms.atrStop')}
          value={value.atr_stop_multiplier}
          min={0.01}
          onChange={(atr_stop_multiplier) => onChange({ ...value, atr_stop_multiplier })}
          error={errors['signals.atr_stop_multiplier']}
        />
        <NumberField
          name="signals.reward_ratio"
          label={t('forms.reward')}
          value={value.reward_ratio}
          min={0.01}
          onChange={(reward_ratio) => onChange({ ...value, reward_ratio })}
          error={errors['signals.reward_ratio']}
        />
      </Section>
    </div>
  );
}
