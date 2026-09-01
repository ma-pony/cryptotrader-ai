import { useTranslation } from 'react-i18next';
import type { ConfigurationDraft, RuntimeDocument } from '@/types/api';
import { BooleanField, NumberField, type DomainFormProps } from '@/components/configuration/field';
import { AdvancedSection, Section } from '@/components/configuration/section';

export function SchedulerSettings({
  value,
  onChange,
  errors = {},
  triggers,
  onTriggersChange,
}: DomainFormProps<RuntimeDocument['scheduler']> & {
  triggers: ConfigurationDraft<RuntimeDocument['triggers']>;
  onTriggersChange: (value: ConfigurationDraft<RuntimeDocument['triggers']>) => void;
}) {
  const { t } = useTranslation('configuration');
  return (
    <div className="configuration-form">
      <Section title={t('forms.schedulerTitle')}>
        <BooleanField
          name="scheduler.enabled"
          label={t('forms.schedulerEnabled')}
          value={value.enabled}
          onChange={(enabled) => onChange({ ...value, enabled })}
        />
        <NumberField
          name="scheduler.interval_minutes"
          label={t('forms.interval')}
          min={1}
          step={1}
          value={value.interval_minutes}
          onChange={(interval_minutes) => onChange({ ...value, interval_minutes })}
          error={errors['scheduler.interval_minutes']}
        />
        <NumberField
          name="scheduler.daily_summary_hour"
          label={t('forms.summaryHour')}
          help={t('forms.summaryHelp')}
          min={0}
          max={23}
          step={1}
          value={value.daily_summary_hour}
          onChange={(daily_summary_hour) => onChange({ ...value, daily_summary_hour })}
          error={errors['scheduler.daily_summary_hour']}
        />
        <BooleanField
          name="triggers.enabled"
          label={t('forms.triggersEnabled')}
          value={triggers.enabled}
          onChange={(enabled) => onTriggersChange({ ...triggers, enabled })}
        />
      </Section>
      <AdvancedSection title={t('forms.triggerAdvanced')}>
        {(['max_rules', 'ws_reconnect_max_s', 'funding_rate_poll_interval_minutes'] as const).map((key) => (
          <NumberField
            key={key}
            name={`triggers.${key}`}
            label={t(`forms.${key}`)}
            min={1}
            step={1}
            value={triggers[key]}
            onChange={(next) => onTriggersChange({ ...triggers, [key]: next })}
            error={errors[`triggers.${key}`]}
          />
        ))}
      </AdvancedSection>
    </div>
  );
}
