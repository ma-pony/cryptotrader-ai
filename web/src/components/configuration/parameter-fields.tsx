import { useTranslation } from 'react-i18next';
import { decodeJsonValue } from '@/hooks/use-runtime-config';
import type { ConfigurationField, RuntimeJsonObject, RuntimeJsonValue } from '@/types/api';
import { BooleanField, ChoiceField, NumberField, StringListField, TextField, type FieldErrors } from './field';
import { AdvancedSection } from './section';

export function isValidParameterNumber(value: unknown, field: ConfigurationField): boolean {
  return typeof value === 'number' && Number.isFinite(value) &&
    (field.minimum === null || value >= field.minimum) &&
    (field.maximum === null || value <= field.maximum) &&
    (field.exclusive_minimum === null || value > field.exclusive_minimum) &&
    (field.exclusive_maximum === null || value < field.exclusive_maximum) &&
    (field.kind !== 'integer' || Number.isInteger(value));
}

export function getParameter(value: RuntimeJsonObject, path: string): RuntimeJsonValue | undefined {
  let current: RuntimeJsonValue | undefined = value;
  for (const key of path.split('.'))
    current = current && typeof current === 'object' && !Array.isArray(current) ? current[key] : undefined;
  return current;
}
export function setParameter(value: RuntimeJsonObject, path: string, next: RuntimeJsonValue): RuntimeJsonObject {
  const [key, ...rest] = path.split('.');
  if (!key) return value;
  const child = value[key];
  return {
    ...value,
    [key]: rest.length
      ? setParameter(child && typeof child === 'object' && !Array.isArray(child) ? child : {}, rest.join('.'), next)
      : next,
  };
}
export function parameterDefaults(fields: ConfigurationField[]): RuntimeJsonObject {
  return fields.reduce((value, field) => {
    const fallback = decodeJsonValue(field.default_value);
    return fallback === null ? value : setParameter(value, field.key, fallback);
  }, {});
}
export function ParameterFields({
  fields,
  value,
  onChange,
  errors = {},
  idPrefix,
}: {
  fields: ConfigurationField[];
  value: RuntimeJsonObject;
  onChange: (value: RuntimeJsonObject) => void;
  errors?: FieldErrors;
  idPrefix: string;
}) {
  const { t, i18n } = useTranslation('configuration');
  const locale = i18n.language.startsWith('zh') ? 'zh_CN' : 'en_US';
  const renderField = (field: ConfigurationField) => {
    const name = `${idPrefix}.${field.key}`;
    const raw = getParameter(value, field.key) ?? decodeJsonValue(field.default_value);
    const percent = field.unit === 'ratio';
    const scale = percent ? 100 : 1;
    const unit = percent ? '%' : field.unit ? t(`forms.units.${field.unit}`, { defaultValue: field.unit }) : '';
    const common = {
      name,
      label: field.label[locale] + (unit ? (locale === 'zh_CN' ? `（${unit}）` : ` (${unit})`) : ''),
      help: field.description[locale],
      error: errors[name] ?? errors[field.key],
      required: field.required,
    };
    const change = (next: RuntimeJsonValue) => onChange(setParameter(value, field.key, next));
    switch (field.kind) {
      case 'number':
      case 'integer':
        return (
          <NumberField
            key={name}
            {...common}
            required={field.required || raw !== null}
            value={typeof raw === 'number' ? raw : ''}
            percent={percent}
            onChange={change}
            {...(field.minimum === null ? {} : { min: field.minimum * scale })}
            {...(field.maximum === null ? {} : { max: field.maximum * scale })}
            {...(field.step === null
              ? field.kind === 'integer'
                ? { step: scale }
                : {}
              : { step: field.step * scale })}
          />
        );
      case 'boolean':
        return <BooleanField key={name} {...common} value={raw === true} onChange={change} />;
      case 'select':
        return (
          <ChoiceField
            key={name}
            {...common}
            value={typeof raw === 'string' ? raw : ''}
            onChange={change}
            options={field.options.map((option) => ({ value: option.value, label: option.label[locale] }))}
            placeholder={t('forms.select')}
          />
        );
      case 'string_list':
        return (
          <StringListField
            key={name}
            {...common}
            value={Array.isArray(raw) ? raw.filter((item): item is string => typeof item === 'string') : []}
            onChange={change}
          />
        );
      case 'text':
        return <TextField key={name} {...common} value={typeof raw === 'string' ? raw : ''} onChange={change} />;
    }
  };
  const advanced = fields.filter((field) => field.advanced);
  return (
    <div className="configuration-parameters">
      <div className="configuration-grid">{fields.filter((field) => !field.advanced).map(renderField)}</div>
      {advanced.length ? (
        <AdvancedSection title={t('forms.advanced')}>{advanced.map(renderField)}</AdvancedSection>
      ) : null}
    </div>
  );
}
