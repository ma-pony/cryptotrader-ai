import { useTranslation } from 'react-i18next';
import type { ConfigurationDraft, RuntimeDocument } from '@/types/api';
import {
  BooleanField,
  NumberField,
  TextField,
  RuntimeSecretField,
  type FieldErrors,
  type SecretFieldState,
} from '@/components/configuration/field';
import { Section } from '@/components/configuration/section';

export type SystemSettingsValue = Pick<
  RuntimeDocument,
  'security' | 'accounts' | 'infrastructure' | 'observability'
>;
export function SystemSettings({
  value,
  onChange,
  errors = {},
  secrets,
}: {
  value: ConfigurationDraft<SystemSettingsValue>;
  onChange: (patch: Partial<ConfigurationDraft<SystemSettingsValue>>) => void;
  errors?: FieldErrors;
  secrets?: SecretFieldState;
}) {
  const { t } = useTranslation('configuration');
  return (
    <div className="configuration-form">
      <Section title={t('forms.securityTitle')}>
        <BooleanField
          name="security.enabled"
          label={t('runtimeSecrets.enable')}
          value={value.security.enabled}
          onChange={(enabled) => onChange({ security: { enabled } })}
        />
        {secrets ? <RuntimeSecretField kind="api-access" state={secrets} /> : null}
      </Section>
      <Section title={t('forms.infrastructureTitle')}>
        <NumberField
          name="accounts.sync_interval_seconds"
          label={t('forms.accountSyncInterval')}
          help={t('forms.accountSyncIntervalHelp')}
          min={1}
          max={86400}
          step={1}
          value={value.accounts.sync_interval_seconds}
          onChange={(sync_interval_seconds) => onChange({ accounts: { sync_interval_seconds } })}
          error={errors['accounts.sync_interval_seconds']}
        />
        <TextField
          name="infrastructure.redis_url"
          label={t('forms.redisUrl')}
          help={t('forms.redisHelp')}
          value={value.infrastructure.redis_url}
          onChange={(redis_url) => onChange({ infrastructure: { redis_url } })}
          error={errors['infrastructure.redis_url']}
        />
        <TextField
          name="observability.otlp_endpoint"
          label={t('forms.otlpEndpoint')}
          help={t('forms.otlpHelp')}
          value={value.observability.otlp_endpoint}
          onChange={(otlp_endpoint) => onChange({ observability: { otlp_endpoint } })}
          error={errors['observability.otlp_endpoint']}
        />
      </Section>
    </div>
  );
}
