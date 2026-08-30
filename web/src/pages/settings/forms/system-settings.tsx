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
  'security' | 'infrastructure' | 'notifications' | 'observability'
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
      <Section title={t('forms.notificationsTitle')} description={t('forms.notificationsHelp')}>
        <BooleanField
          name="notifications.enabled"
          label={t('forms.notificationsEnabled')}
          value={value.notifications.enabled}
          onChange={(enabled) =>
            onChange({ notifications: { ...value.notifications, enabled, events: enabled ? ['daily_summary'] : [] } })
          }
        />
        <TextField
          name="notifications.webhook_url"
          label={t('forms.webhookUrl')}
          type="url"
          value={value.notifications.webhook_url}
          onChange={(webhook_url) => onChange({ notifications: { ...value.notifications, webhook_url } })}
          error={errors['notifications.webhook_url']}
        />
        <NumberField
          name="notifications.webhook_timeout"
          label={t('forms.webhookTimeout')}
          min={1}
          step={1}
          value={value.notifications.webhook_timeout}
          onChange={(webhook_timeout) => onChange({ notifications: { ...value.notifications, webhook_timeout } })}
          error={errors['notifications.webhook_timeout']}
        />
      </Section>
    </div>
  );
}
