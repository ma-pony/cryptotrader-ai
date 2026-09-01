import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { formatDateTime } from '@/lib/format';
import { ApiError } from '@/lib/api-client';
import {
  BooleanField,
  ChoiceField,
  focusFirstError,
  NumberField,
  TextField,
  type FieldErrors,
} from '@/components/configuration/field';
import { CredentialPanel } from '@/components/configuration/credential-panel';
import {
  getParameter,
  isValidParameterNumber,
  ParameterFields,
  parameterDefaults,
  setParameter,
} from '@/components/configuration/parameter-fields';
import { useVenueEnvironmentDefinition } from '@/hooks/use-configuration-catalog';
import { useVenueConnections } from '@/hooks/use-venue-connections';
import { decodeJsonValue } from '@/hooks/use-runtime-config';
import { useConfiguration } from '@/pages/settings/configuration-context';
import { connectionFingerprint, type Connection } from '@/lib/configuration-readiness';

type Draft = Omit<Connection, 'credential_configured' | 'credential_updated_at'>;
const equalDraft = (left: Draft, right: Connection) =>
  JSON.stringify(left) ===
  JSON.stringify({
    id: right.id,
    label: right.label,
    adapter_id: right.adapter_id,
    environment: right.environment,
    enabled: right.enabled,
    leverage: right.leverage,
    margin_mode: right.margin_mode,
    canary_only: right.canary_only,
    parameters: right.parameters,
  });

export function ConnectionForm({ connectionId, onSaved }: { connectionId?: string; onSaved: (id: string) => void }) {
  const runtime = useConfiguration();
  const { t, i18n } = useTranslation('configuration');
  const locale = i18n.language.startsWith('zh') ? 'zh_CN' : 'en_US';
  const catalog = runtime.catalog.data;
  const fieldPrefix = `${connectionId ?? 'new'}.connection`;
  const connection = runtime.baseline?.execution.connections.find((item) => item.id === connectionId);
  const firstVenue = catalog?.venues[0];
  const sharedDraft = connectionId ? runtime.venueDrafts[connectionId] : undefined;
  const draft = connectionId ? (sharedDraft ?? connection ?? null) : runtime.newVenueDraft;
  const venue = catalog?.venues.find((item) => item.id === draft?.adapter_id);
  const definition = useVenueEnvironmentDefinition(draft?.adapter_id ?? '', draft?.environment ?? '');
  const credentials = runtime.credentialStates[connectionId ?? ''];
  const [values, setValues] = useState<Record<string, string>>({});
  const [editingCredentials, setEditingCredentials] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [missing, setMissing] = useState<string[]>([]);
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const formRef = useRef<HTMLFormElement>(null);
  const definitionMatchesDraft = Boolean(
    draft && definition.data?.id === draft.adapter_id && definition.data.environment.id === draft.environment,
  );
  const definitionReady = definitionMatchesDraft;
  const fields = definitionReady ? definition.data!.fields : [];
  const credentialFields = definitionReady ? definition.data!.credential_fields : [];
  const margins = definitionReady ? definition.data!.margin_modes : [];
  useEffect(() => {
    if (!connectionId && !runtime.newVenueDraft && firstVenue) {
      runtime.setNewVenueDraft({
        id: `venue-${crypto.randomUUID()}`,
        label: '',
        adapter_id: firstVenue.id,
        environment: firstVenue.environments[0]?.id ?? '',
        enabled: true,
        leverage: 1,
        margin_mode: firstVenue.margin_modes[0] ?? 'cross',
        canary_only: false,
        parameters: {},
      });
    }
  }, [connectionId, firstVenue, runtime]);
  const venueApi = useVenueConnections();
  const dirty = Boolean(draft && (!connection || !equalDraft(draft, connection)));
  const configured = credentials?.configured ?? false;
  const openCredentials =
    credentialFields.length > 0 && (!configured || editingCredentials || Object.keys(values).length > 0);
  const savedCheck = connectionId ? runtime.checks[connectionId] : undefined;
  const currentHealth =
    savedCheck &&
    !dirty &&
    !editingCredentials &&
    Object.keys(values).length === 0 &&
    !runtime.checking[connectionId ?? ''] &&
    !runtime.checkErrors[connectionId ?? ''] &&
    draft &&
    savedCheck.fingerprint === connectionFingerprint(draft, credentials?.updatedAt)
      ? savedCheck.health
      : undefined;
  const requiredMissing = credentialFields
    .filter((field) => field.required && !values[field.key]?.trim())
    .map((field) => field.label[locale]);
  if (!draft || !catalog) return null;
  const leverageMinimum = definition.data?.leverage_minimum ?? 1;
  const leverageMaximum = definition.data?.leverage_maximum;
  const effectiveLeverage = Math.max(
    leverageMinimum,
    Math.min(draft.leverage, leverageMaximum ?? draft.leverage),
  );
  const effectiveMarginMode = margins.includes(draft.margin_mode)
    ? draft.margin_mode
    : (margins[0] ?? draft.margin_mode);
  const effectiveParameters = fields.reduce((parameters, field) => {
    const explicit = getParameter(draft.parameters, field.key);
    return explicit === undefined ? parameters : setParameter(parameters, field.key, explicit);
  }, parameterDefaults(fields));
  const change = (next: Draft) => {
    if (connectionId) {
      runtime.setVenueDrafts((current) => {
        if (connection && equalDraft(next, connection)) {
          const restored = { ...current };
          delete restored[connectionId];
          return restored;
        }
        return { ...current, [connectionId]: next };
      });
    } else runtime.setNewVenueDraft(next);
    setError('');
    setMissing([]);
    setFieldErrors({});
  };
  const validateDraft = () => {
    const errors: FieldErrors = {};
    if (!draft.label.trim()) errors[`${fieldPrefix}.label`] = t('forms.validation.required');
    if (
      !Number.isFinite(draft.leverage) ||
      draft.leverage < leverageMinimum ||
      (leverageMaximum !== null && leverageMaximum !== undefined && draft.leverage > leverageMaximum)
    ) {
      errors[`${fieldPrefix}.leverage`] = t('forms.validation.numberInvalid');
    }
    fields.forEach((field) => {
      const value = getParameter(draft.parameters, field.key) ?? decodeJsonValue(field.default_value);
      const name = `${fieldPrefix}.parameters.${field.key}`;
      const empty =
        value === undefined || value === null || value === '' || (Array.isArray(value) && value.length === 0);
      if (field.required && empty) errors[name] = t('forms.validation.required');
      else if ((field.kind === 'number' || field.kind === 'integer') && !isValidParameterNumber(value, field)) {
        errors[name] = t(`forms.validation.${field.kind === 'integer' ? 'integerInvalid' : 'numberInvalid'}`);
      } else if (
        field.kind === 'select' &&
        typeof value === 'string' &&
        !field.options.some((option) => option.value === value)
      ) {
        errors[name] = t('forms.validation.required');
      }
    });
    setFieldErrors(errors);
    if (Object.keys(errors).length) focusFirstError(errors, formRef.current ?? undefined);
    return Object.keys(errors).length === 0;
  };
  const saveAndCheck = async () => {
    if (saving || !venue || !definitionReady || definition.isError || runtime.conflict) return;
    if (!validateDraft()) return;
    const stoppingBooks =
      !draft.enabled && connection?.enabled
        ? (runtime.baseline?.execution.books.filter(
            (b) => b.enabled && b.allocations.some((a) => a.enabled && a.connection_id === connection.id),
          ) ?? [])
        : [];
    if (
      stoppingBooks.length &&
      !window.confirm(t('connection.stopOwnedBooks', { names: stoppingBooks.map((b) => b.label).join('、') }))
    )
      return;
    setSaving(true);
    setError('');
    setMissing([]);
    let connectionSaved = Boolean(connection && !dirty);
    let stage: 'connection' | 'credentials' | 'check' = 'connection';
    try {
      let saved = connection;
      let revision = runtime.revision ?? 0;
      if (dirty) {
        const body = {
          expected_revision: revision,
          label: draft.label,
          adapter_id: draft.adapter_id,
          environment: draft.environment,
          enabled: draft.enabled,
          leverage: effectiveLeverage,
          margin_mode: effectiveMarginMode,
          canary_only: draft.canary_only,
          parameters: effectiveParameters,
          ...(stoppingBooks.length ? { confirm_stop: true } : {}),
        };
        const result = connection
          ? await venueApi.update.mutateAsync({ id: connection.id, body })
          : await venueApi.create.mutateAsync({ ...body, id: draft.id });
        saved = {
          id: result.connection.id,
          label: result.connection.label,
          adapter_id: result.connection.adapter_id,
          environment: result.connection.environment,
          enabled: result.connection.enabled,
          leverage: result.connection.leverage,
          margin_mode: result.connection.margin_mode,
          canary_only: result.connection.canary_only,
          parameters: Object.fromEntries(
            result.connection.parameters.map((entry) => [entry.key, decodeJsonValue(entry.value)]),
          ),
        };
        revision = result.revision;
        connectionSaved = true;
        if (connectionId)
          runtime.setVenueDrafts((current) => {
            const next = { ...current };
            delete next[connectionId];
            return next;
          });
        if (!connection) {
          runtime.setNewVenueDraft(null);
          onSaved(saved.id);
        }
        if (result.savedNeedsReload) {
          setError(t('connection.savedNeedsReload'));
          return;
        }
      }
      if (!saved) return;
      const credentialValues = Object.fromEntries(
        credentialFields.filter((field) => values[field.key]?.trim()).map((field) => [field.key, values[field.key]!]),
      );
      const needsCredentials =
        credentialFields.length > 0 && (!configured || editingCredentials || Object.keys(credentialValues).length > 0);
      if (needsCredentials && requiredMissing.length) {
        setMissing(requiredMissing);
        return;
      }
      let checkedAt = credentials?.updatedAt ?? null;
      if (needsCredentials) {
        stage = 'credentials';
        const result = await venueApi.putCredentials({
          id: saved.id,
          expectedRevision: revision,
          values: credentialValues,
        });
        revision = result.revision;
        checkedAt = result.credential.updated_at;
        setValues({});
        setEditingCredentials(false);
        if (result.savedNeedsReload) {
          setError(t('connection.credentialSavedNeedsReload'));
          return;
        }
      }
      stage = 'check';
      await runtime.checkConnection(saved.id, saved, checkedAt);
      onSaved(saved.id);
    } catch (cause) {
      setError(
        cause instanceof ApiError && cause.status === 409
          ? t('forms.conflictHelp')
          : t(
              stage === 'credentials'
                ? 'connection.credentialSaveFailed'
                : stage === 'check'
                  ? 'connection.checkFailed'
                  : connectionSaved
                    ? 'connection.savedNeedsReload'
                    : 'connection.saveFailed',
            ),
      );
    } finally {
      setSaving(false);
    }
  };
  const deleteCredentials = async () => {
    if (!connection || runtime.conflict || !window.confirm(t('connection.deleteConfirm'))) return;
    setSaving(true);
    setError('');
    try {
      const result = await venueApi.deleteCredentials({ id: connection.id, expectedRevision: runtime.revision ?? 0 });
      setValues({});
      setEditingCredentials(false);
      if (result.savedNeedsReload) setError(t('connection.credentialSavedNeedsReload'));
    } catch (cause) {
      setError(
        cause instanceof ApiError && cause.status === 409 ? t('forms.conflictHelp') : t('connection.deleteFailed'),
      );
    } finally {
      setSaving(false);
    }
  };
  return (
    <form
      ref={formRef}
      noValidate
      aria-label={connection?.label ?? t('addVenue')}
      className="configuration-venue-form"
      onSubmit={(event) => {
        event.preventDefault();
        void saveAndCheck();
      }}
    >
      <div className="configuration-grid">
        <TextField
          name={`${fieldPrefix}.label`}
          label={t('connection.name')}
          required
          error={fieldErrors[`${fieldPrefix}.label`]}
          value={draft.label}
          onChange={(label) => change({ ...draft, label })}
        />
        <ChoiceField
          name={`${fieldPrefix}.adapter`}
          label={t('connection.adapter')}
          disabled={Boolean(connection)}
          value={draft.adapter_id}
          options={catalog.venues.map((item) => ({ value: item.id, label: item.label[locale] }))}
          onChange={(adapterId) => {
            const selected = catalog.venues.find((item) => item.id === adapterId)!;
            setValues({});
            change({
              ...draft,
              adapter_id: adapterId,
              environment: selected.environments[0]?.id ?? '',
              margin_mode: selected.margin_modes[0] ?? 'cross',
              parameters: {},
            });
          }}
        />
        <ChoiceField
          name={`${fieldPrefix}.environment`}
          label={t('connection.environment')}
          disabled={Boolean(connection)}
          value={draft.environment}
          options={(venue?.environments ?? []).map((environment) => ({
            value: environment.id,
            label: environment.label[locale],
          }))}
          onChange={(environment) => {
            setValues({});
            change({
              ...draft,
              environment,
              margin_mode: venue?.margin_modes[0] ?? draft.margin_mode,
              parameters: {},
            });
          }}
        />
        <BooleanField
          name={`${fieldPrefix}.enabled`}
          label={t('connection.enabled')}
          value={draft.enabled}
          onChange={(enabled) => change({ ...draft, enabled })}
        />
      </div>
      {!definitionReady && !definition.isError ? (
        <p role="status" className="configuration-help">
          {t('connection.definitionLoading')}
        </p>
      ) : null}
      <ParameterFields
        fields={fields}
        value={draft.parameters}
        idPrefix={`${fieldPrefix}.parameters`}
        onChange={(parameters) => change({ ...draft, parameters })}
        errors={fieldErrors}
      />
      <div className="configuration-grid">
        <NumberField
          name={`${fieldPrefix}.leverage`}
          label={t('connection.leverage')}
          error={fieldErrors[`${fieldPrefix}.leverage`]}
          min={definition.data?.leverage_minimum ?? 1}
          {...(definition.data?.leverage_maximum === null || definition.data?.leverage_maximum === undefined
            ? {}
            : { max: definition.data.leverage_maximum })}
          value={draft.leverage}
          onChange={(leverage) => change({ ...draft, leverage: typeof leverage === 'number' ? leverage : 1 })}
        />
        {margins.length > 1 ? (
          <ChoiceField
            name={`${fieldPrefix}.margin_mode`}
            label={t('connection.marginMode')}
            value={effectiveMarginMode}
            options={margins.map((mode) => ({
              value: mode,
              label: t(`connection.modes.${mode}`, { defaultValue: mode }),
            }))}
            onChange={(margin_mode) => change({ ...draft, margin_mode })}
          />
        ) : (
          <p className="configuration-help">
            {t('connection.fixedMargin', {
              mode: t(`connection.modes.${effectiveMarginMode}`, { defaultValue: effectiveMarginMode }),
            })}
          </p>
        )}
        <BooleanField
          name={`${fieldPrefix}.canary_only`}
          label={t('connection.canaryOnly')}
          help={t('connection.canaryWarning')}
          value={draft.canary_only}
          onChange={(canary_only) => change({ ...draft, canary_only })}
        />
      </div>
      {credentialFields.length ? (
        <>
          <CredentialPanel
            title={t('credentials.venueTitle')}
            configured={configured}
            updatedAt={credentials?.updatedAt}
            editing={editingCredentials || !configured}
            pending={saving}
            disabled={saving || dirty}
            onEdit={() => {
              setValues({});
              setEditingCredentials(true);
            }}
            onCancel={() => {
              setValues({});
              setEditingCredentials(false);
            }}
          >
            <div className="configuration-grid">
              {openCredentials
                ? credentialFields.map((field) => (
                    <TextField
                      key={field.key}
                      name={`${fieldPrefix}.credentials.${field.key}`}
                      label={field.label[locale]}
                      help={field.description[locale]}
                      type="password"
                      required={field.required}
                      value={values[field.key] ?? ''}
                      onChange={(value) => setValues((current) => ({ ...current, [field.key]: value }))}
                    />
                  ))
                : null}
            </div>
          </CredentialPanel>
          {configured ? (
            <button
              className="configuration-button"
              type="button"
              disabled={saving || runtime.conflict}
              onClick={() => void deleteCredentials()}
            >
              {t('connection.deleteCredentials')}
            </button>
          ) : null}
        </>
      ) : (
        <p className="configuration-help">{t('connection.noCredentials')}</p>
      )}
      {missing.length ? (
        <p role="status" className="configuration-help">
          {t('connection.credentialsMissing', { names: missing.join('、') })}
        </p>
      ) : null}
      <div className="configuration-actions">
        <button
          className="configuration-button configuration-primary"
          type="submit"
          disabled={saving || !definitionReady || definition.isError || runtime.conflict}
        >
          {t(saving ? 'connection.checking' : 'connection.saveAndCheck')}
        </button>
      </div>
      {definition.isError ? (
        <p role="alert" className="configuration-error">
          {t('connection.definitionLoadFailed')}
        </p>
      ) : null}
      {currentHealth?.healthy ? (
        <p role="status">
          {t('connection.verified')} ·{' '}
          <time dateTime={currentHealth.checked_at}>{formatDateTime(currentHealth.checked_at)}</time>
        </p>
      ) : null}
      {currentHealth && !currentHealth.healthy ? (
        <p role="alert" className="configuration-error">
          {t('connection.testFailed')}
        </p>
      ) : null}
      {error ? (
        <p role="alert" className="configuration-error">
          {error}
        </p>
      ) : null}
    </form>
  );
}
