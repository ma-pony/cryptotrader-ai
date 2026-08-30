import { useTranslation } from 'react-i18next';
import type { ConfigurationDraft, RuntimeDocument } from '@/types/api';
import { NumberField, type DomainFormProps } from '@/components/configuration/field';
import { Section } from '@/components/configuration/section';

export function RiskSettings({
  value,
  onChange,
  errors = {},
  hitl,
  onHitlChange,
}: DomainFormProps<RuntimeDocument['risk']> & {
  hitl: ConfigurationDraft<RuntimeDocument['hitl']>;
  onHitlChange: (value: ConfigurationDraft<RuntimeDocument['hitl']>) => void;
}) {
  const { t } = useTranslation('configuration');
  return (
    <div className="configuration-form">
      <Section title={t('forms.riskTitle')} description={t('forms.riskHelp')}>
        {(['max_total_exposure_pct', 'max_single_pct', 'max_margin_used_pct'] as const).map((key) => (
          <NumberField
            key={key}
            name={`risk.position.${key}`}
            label={t(`forms.${key}`)}
            help={t(`forms.${key}Help`)}
            value={value.position[key]}
            min={0}
            max={100}
            percent
            onChange={(next) => onChange({ ...value, position: { ...value.position, [key]: next } })}
            error={errors[`risk.position.${key}`]}
          />
        ))}
        <NumberField
          name="risk.loss.max_drawdown_pct"
          label={t('forms.drawdown')}
          help={t('forms.drawdownHelp')}
          value={value.loss.max_drawdown_pct}
          min={0}
          max={100}
          percent
          onChange={(max_drawdown_pct) => onChange({ ...value, loss: { ...value.loss, max_drawdown_pct } })}
          error={errors['risk.loss.max_drawdown_pct']}
        />
        <NumberField
          name="hitl.approval_ttl_minutes"
          label={t('forms.approvalTtl')}
          help={t('forms.approvalHelp')}
          value={hitl.approval_ttl_minutes}
          min={1}
          step={1}
          onChange={(approval_ttl_minutes) => onHitlChange({ approval_ttl_minutes })}
          error={errors['hitl.approval_ttl_minutes']}
        />
      </Section>
    </div>
  );
}
