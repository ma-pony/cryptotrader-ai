import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { formatDateTime } from '@/lib/format';
import {
  BooleanField,
  ChoiceField,
  NumberField,
  TextField,
  focusFirstError,
  type FieldErrors,
} from '@/components/configuration/field';
import { AdvancedSection } from '@/components/configuration/section';
import { ParameterFields, parameterDefaults, getParameter, isValidParameterNumber } from '@/components/configuration/parameter-fields';
import { useVenueConnections } from '@/hooks/use-venue-connections';
import { decodeJsonValue } from '@/hooks/use-runtime-config';
import { ApiError } from '@/lib/api-client';
import { connectionFingerprint, type Connection, type ConnectionCheck } from '@/lib/configuration-readiness';
import type { ConfigurationCatalog, ConfigurationDraft } from '@/types/api';

export type VenueDraft = ConfigurationDraft<Connection>;
export const newVenue = (catalog: ConfigurationCatalog): VenueDraft => {
  const adapter = catalog.venues.find((item) => item.id === 'paper') ?? catalog.venues[0];
  return {
    id: 'venue-' + crypto.randomUUID(),
    label: '',
    adapter_id: adapter?.id ?? '',
    environment: (adapter?.environments[0] ?? 'paper') as Connection['environment'],
    enabled: true,
    leverage: 1,
    margin_mode: adapter?.margin_modes[0] ?? 'cross',
    canary_only: false,
    parameters: parameterDefaults(adapter?.fields ?? []),
  };
};
export function VenueForm({
  value,
  connection,
  catalog,
  revision,
  credentialState,
  check,
  onChange,
  onSaved,
  onChecked,
  onCancel,
  writeBlocked = false,
}: {
  value: VenueDraft;
  connection?: Connection | undefined;
  catalog: ConfigurationCatalog;
  revision: number;
  credentialState?: { configured: boolean; updatedAt: string | null } | undefined;
  check?: ConnectionCheck | undefined;
  onChange: (value: VenueDraft) => void;
  onSaved: () => void;
  onChecked: (check: ConnectionCheck | undefined) => void;
  onCancel?: (() => void) | undefined;
  writeBlocked?: boolean;
}) {
  const { t, i18n } = useTranslation('configuration');
  const form = useRef<HTMLFormElement>(null);
  const venues = useVenueConnections();
  const [credentials, setCredentials] = useState<Record<string, string>>({});
  const [credentialSaving, setCredentialSaving] = useState(false);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [failure, setFailure] = useState('');
  const [savedNeedsReload, setSavedNeedsReload] = useState(false);
  const plugin = catalog.venues.find((item) => item.id === value.adapter_id);
  const locale = i18n.language.startsWith('zh') ? 'zh_CN' : 'en_US';
  const prefix = 'venue.' + (connection?.id ?? 'new');
  const dirty = !connection || JSON.stringify(value) !== JSON.stringify(connection);
  const pending = venues.create.isPending || venues.update.isPending || credentialSaving;
  const blocked = pending || writeBlocked;
  const fingerprint = connection ? connectionFingerprint(connection, credentialState?.updatedAt) : undefined;
  const currentCheck =
    !failure &&
    !venues.test.isPending &&
    !dirty &&
    !Object.values(credentials).some(Boolean) &&
    check?.fingerprint === fingerprint
      ? check
      : undefined;
  const credentialLabels: Record<string, string> = {
    api_key: 'API Key', // pragma: allowlist secret -- public field label
    secret: 'API Secret', // pragma: allowlist secret -- public field label
    passphrase: 'OKX Passphrase', // pragma: allowlist secret -- public field label
  };
  const credentialHelpKeys: Record<string, string> = {
    api_key: 'identity', // pragma: allowlist secret -- translation key
    secret: 'signing', // pragma: allowlist secret -- translation key
    passphrase: 'phrase', // pragma: allowlist secret -- translation key
  };
  const change = (next: VenueDraft) => {
    setErrors({});
    setFailure('');
    onChange(next);
  };
  const showFailure = (error: unknown, kind: 'save' | 'check' | 'access') => {
    const code = error instanceof ApiError ? error.code : '';
    setFailure(
      t(
        error instanceof ApiError && error.status === 409
          ? 'forms.conflictHelp'
          : kind === 'check'
            ? code === 'authentication_failed'
              ? 'connection.authFailed'
              : code === 'credentials_missing'
                ? 'connection.missingAccess'
                : 'connection.testFailed'
            : kind === 'access'
              ? 'connection.accessSaveFailed'
              : 'connection.saveFailed',
      ),
    );
    if (error instanceof ApiError && error.details?.fieldErrors) {
      const local = Object.fromEntries(
        Object.keys(error.details.fieldErrors).map((path) => [
          prefix + '.' + path.replace(/^connection\./, ''),
          t('forms.validation.serverInvalid'),
        ]),
      );
      setErrors(local);
      focusFirstError(local, form.current ?? undefined);
    }
  };
  const save = async () => {
    if (blocked) return;
    const invalid: FieldErrors = {};
    if (!value.id.trim()) invalid[prefix + '.id'] = t('forms.validation.required');
    if (!value.label.trim()) invalid[prefix + '.label'] = t('forms.validation.required');
    if (!plugin || !plugin.environments.includes(value.environment))
      invalid[prefix + '.adapter_id'] = t('forms.validation.catalogRequired');
    if (typeof value.leverage !== 'number' || !Number.isInteger(value.leverage) || value.leverage < 1)
      invalid[prefix + '.leverage'] = t('forms.validation.integerInvalid');
    for (const field of plugin?.fields ?? []) {
      const candidate = getParameter(value.parameters, field.key) ?? decodeJsonValue(field.default_value);
      const name = prefix + '.parameters.' + field.key;
      if (field.kind === 'number' || field.kind === 'integer') {
        if (!isValidParameterNumber(candidate, field))
          invalid[name] = t('forms.validation.numberInvalid');
      } else if (field.required && (candidate === null || candidate === ''))
        invalid[name] = t('forms.validation.required');
      else if (field.kind === 'select' && !field.options.some((item) => item.value === candidate))
        invalid[name] = t('forms.validation.selectInvalid');
    }
    setErrors(invalid);
    if (Object.keys(invalid).length) {
      focusFirstError(invalid, form.current ?? undefined);
      return;
    }
    try {
      setFailure('');
      const body = {
        ...value,
        margin_mode: plugin?.margin_modes.length === 1 ? plugin.margin_modes[0]! : value.margin_mode,
        leverage: value.leverage as number,
        expected_revision: revision,
      };
      const saved = connection
        ? await venues.update.mutateAsync({ id: connection.id, body })
        : await venues.create.mutateAsync(body);
      setSavedNeedsReload(saved.savedNeedsReload);
      onSaved();
    } catch (error) {
      showFailure(error, 'save');
    }
  };
  const saveCredentials = async () => {
    if (!connection || blocked || dirty || !plugin?.credential_fields.every((key) => credentials[key]?.trim())) return;
    setCredentialSaving(true);
    setFailure('');
    try {
      const saved = await venues.putCredentials({
        id: connection.id,
        expectedRevision: revision,
        credentials: {
          api_key: credentials.api_key!,
          secret: credentials.secret!,
          ...(credentials.passphrase ? { passphrase: credentials.passphrase } : {}),
        },
      });
      setSavedNeedsReload(saved.savedNeedsReload);
    } catch (error) {
      showFailure(error, 'access');
    } finally {
      setCredentials({});
      setCredentialSaving(false);
    }
  };
  const test = async () => {
    if (!connection || dirty || blocked) return;
    setFailure('');
    onChecked(undefined);
    try {
      const health = await venues.test.mutateAsync(connection.id);
      onChecked({ fingerprint: fingerprint!, health });
    } catch (error) {
      showFailure(error, 'check');
    }
  };
  return (
    <form
      ref={form}
      aria-label={connection?.label ?? t('addVenue')}
      noValidate
      className="configuration-venue-form"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      <fieldset disabled={pending} className="space-y-4">
        <div className="configuration-grid">
          <TextField
            name={prefix + '.label'}
            label={t('connection.name')}
            required
            value={value.label}
            onChange={(label) => change({ ...value, label })}
            error={errors[prefix + '.label']}
          />
          <ChoiceField
            name={prefix + '.adapter_id'}
            label={t('connection.adapter')}
            value={value.adapter_id}
            disabled={Boolean(connection)}
            options={catalog.venues.map((item) => ({ value: item.id, label: item.label[locale] }))}
            onChange={(adapter_id) => {
              const adapter = catalog.venues.find((item) => item.id === adapter_id)!;
              setCredentials({});
              change({
                ...value,
                adapter_id,
                environment: adapter.environments[0] as Connection['environment'],
                margin_mode: adapter.margin_modes[0] ?? 'cross',
                parameters: parameterDefaults(adapter.fields),
              });
            }}
            error={errors[prefix + '.adapter_id']}
          />
          <ChoiceField
            name={prefix + '.environment'}
            label={t('connection.environment')}
            disabled={Boolean(connection)}
            value={value.environment}
            options={(plugin?.environments ?? []).map((environment) => ({
              value: environment,
              label: t('connection.environments.' + environment, { defaultValue: environment }),
            }))}
            onChange={(environment) => change({ ...value, environment: environment as Connection['environment'] })}
          />
          <BooleanField
            name={prefix + '.enabled'}
            label={t('connection.enabled')}
            value={value.enabled}
            onChange={(enabled) => change({ ...value, enabled })}
          />
        </div>
        <ParameterFields
          fields={plugin?.fields ?? []}
          value={value.parameters}
          idPrefix={prefix + '.parameters'}
          onChange={(parameters) => change({ ...value, parameters })}
          errors={errors}
        />
        <AdvancedSection title={t('connection.advanced')}>
          <TextField
            name={prefix + '.id'}
            label={t('connection.id')}
            help={t('connection.idHelp')}
            disabled={Boolean(connection)}
            required
            value={value.id}
            onChange={(id) => change({ ...value, id })}
            error={errors[prefix + '.id']}
          />
          <NumberField
            name={prefix + '.leverage'}
            label={t('connection.leverage')}
            value={value.leverage}
            min={1}
            step={1}
            onChange={(leverage) => change({ ...value, leverage })}
            error={errors[prefix + '.leverage']}
          />
          {plugin && plugin.margin_modes.length > 1 ? <ChoiceField
            name={prefix + '.margin_mode'}
            label={t('connection.marginMode')}
            value={value.margin_mode}
            options={plugin.margin_modes.map((mode) => ({ value: mode, label: t('connection.modes.' + mode) }))}
            onChange={(margin_mode) => change({ ...value, margin_mode: margin_mode as Connection['margin_mode'] })}
          /> : plugin?.margin_modes.length === 1 ? (
            <p className="configuration-help">
              {t('connection.fixedMargin', { mode: t('connection.modes.' + plugin.margin_modes[0]) })}
              {' '}{t('connection.marginHelp.' + plugin.margin_modes[0])}
            </p>
          ) : null}
          <BooleanField
            name={prefix + '.canary_only'}
            label={t('connection.canaryOnly')}
            help={t('connection.canaryWarning')}
            value={value.canary_only}
            onChange={(canary_only) => change({ ...value, canary_only })}
          />
        </AdvancedSection>
        <div className="configuration-actions">
          <button className="configuration-button configuration-primary" type="submit" disabled={blocked || !dirty}>
            {t(connection ? 'connection.save' : 'connection.create')}
          </button>
          {onCancel ? (
            <button className="configuration-button" type="button" onClick={onCancel}>
              {t('forms.discard')}
            </button>
          ) : null}
        </div>
        {plugin?.credential_fields.length ? (
          <section className="configuration-secret">
            <p className="configuration-help">{t('connection.accessHelp')}</p>
            <div className="configuration-grid">
              {plugin.credential_fields.map((key) => (
                <TextField
                  key={key}
                  name={prefix + '.credentials.' + key}
                  label={credentialLabels[key] ?? key}
                  help={t('connection.credentialHelp.' + credentialHelpKeys[key], {
                    defaultValue: t('runtimeSecrets.hint'),
                  })}
                  type="password"
                  required
                  value={credentials[key] ?? ''}
                  onChange={(text) => setCredentials((current) => ({ ...current, [key]: text }))}
                />
              ))}
            </div>
            <button
              type="button"
              className="configuration-button"
              disabled={
                !connection || dirty || blocked || !plugin.credential_fields.every((key) => credentials[key]?.trim())
              }
              onClick={() => void saveCredentials()}
            >
              {t('connection.saveAccess')}
            </button>
            <p className="configuration-help">
              {t(credentialState?.configured ? 'forms.accessConfigured' : 'forms.accessMissing')}
              {credentialState?.updatedAt ? ' · ' + credentialState.updatedAt : ''}
            </p>
          </section>
        ) : (
          <p className="configuration-help">{t('connection.paperNoAccess')}</p>
        )}
      </fieldset>
      {connection ? (
        <div className="configuration-check">
          <p className="configuration-help">{t('connection.checkHelp')}</p>
          <button
            type="button"
            className="configuration-button"
            disabled={dirty || blocked || venues.test.isPending || Object.values(credentials).some(Boolean)}
            onClick={() => void test()}
          >
            {t(venues.test.isPending ? 'connection.checking' : 'connection.test')}
          </button>
          {dirty ? <p className="configuration-help">{t('connection.saveBeforeCheck')}</p> : null}
          {currentCheck?.health.healthy ? (
            <p role="status">
              {t('connection.verified')} ·{' '}
              <time dateTime={currentCheck.health.checked_at}>{formatDateTime(currentCheck.health.checked_at)}</time>
            </p>
          ) : null}
        </div>
      ) : null}
      {failure ? (
        <p role="alert" className="configuration-error">
          {failure}
        </p>
      ) : null}
      {savedNeedsReload && writeBlocked ? (
        <p role="status" className="configuration-error">
          {t('connection.savedNeedsReload')}
        </p>
      ) : null}
    </form>
  );
}
