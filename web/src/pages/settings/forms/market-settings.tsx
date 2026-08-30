import { useTranslation } from 'react-i18next';
import type { ConfigurationCatalog, RuntimeDocument } from '@/types/api';
import { ChoiceField, type DomainFormProps } from '@/components/configuration/field';
import { ParameterFields, parameterDefaults } from '@/components/configuration/parameter-fields';
import { Section } from '@/components/configuration/section';

export function MarketSettings({
  value,
  onChange,
  errors = {},
  catalog,
}: DomainFormProps<RuntimeDocument['market_data']> & { catalog: ConfigurationCatalog }) {
  const { t, i18n } = useTranslation('configuration');
  const locale = i18n.language.startsWith('zh') ? 'zh_CN' : 'en_US';
  const selected = catalog.market_sources.find((source) => source.id === value.source_id);
  return (
    <div className="configuration-form">
      <Section title={t('forms.marketTitle')}>
        <ChoiceField
          name="market_data.source_id"
          label={t('forms.marketSource')}
          value={value.source_id}
          options={catalog.market_sources.map((source) => ({ value: source.id, label: source.label[locale] }))}
          placeholder={t('forms.select')}
          error={errors['market_data.source_id']}
          onChange={(source_id) => {
            if (source_id === value.source_id) return;
            const source = catalog.market_sources.find((item) => item.id === source_id);
            if (source) onChange({ source_id, parameters: parameterDefaults(source.fields) });
          }}
        />
        {selected ? (
          <div className="configuration-wide">
            <p className="configuration-help">{selected.description[locale]}</p>
            <ParameterFields
              fields={selected.fields}
              value={value.parameters}
              onChange={(parameters) => onChange({ ...value, parameters })}
              errors={errors}
              idPrefix="market_data.parameters"
            />
          </div>
        ) : (
          <p className="configuration-help">{t('forms.pluginUnavailable')}</p>
        )}
      </Section>
    </div>
  );
}
