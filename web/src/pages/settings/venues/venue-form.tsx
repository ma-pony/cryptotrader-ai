import { FlaskConical, Save } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { useVenueConnections } from '@/hooks/use-venue-connections';
import type { RuntimeConfig, RuntimeDocument, RuntimeJsonObject } from '@/types/api';

type Environment = 'paper' | 'demo' | 'testnet' | 'live';
export type VenueDraft = {
  id: string;
  label: string;
  adapter_id: string;
  environment: Environment;
  enabled: boolean;
  leverage: number;
  margin_mode: string;
  parameters: RuntimeJsonObject;
};
const emptyDraft = (): VenueDraft => ({
  id: '',
  label: '',
  adapter_id: 'paper',
  environment: 'paper',
  enabled: true,
  leverage: 1,
  margin_mode: 'cross',
  parameters: {},
});

export const VenueForm = ({
  revision,
  connection,
  onSaved,
  tested,
}: {
  revision: number;
  connection?: RuntimeDocument['execution']['connections'][number];
  onSaved?: ((connection: RuntimeConfig['document']['execution']['connections'][number]) => void) | undefined;
  tested?: ((id: string) => void) | undefined;
}) => {
  const { t } = useTranslation('configuration');
  const venues = useVenueConnections();
  const [draft, setDraft] = useState<VenueDraft>(() => (connection ? { ...connection } : emptyDraft()));
  const [apiKey, setApiKey] = useState('');
  const [secret, setSecret] = useState('');
  const [passphrase, setPassphrase] = useState('');
  const [credentialSaving, setCredentialSaving] = useState(false);
  const [error, setError] = useState('');
  const [savedNeedsReload, setSavedNeedsReload] = useState(false);
  const hydratedConnectionId = useRef(connection?.id);
  useEffect(() => {
    if (hydratedConnectionId.current === connection?.id) return;
    hydratedConnectionId.current = connection?.id;
    setDraft(connection ? { ...connection } : emptyDraft());
    setApiKey('');
    setSecret('');
    setPassphrase('');
  }, [connection]);
  const existing = Boolean(connection);
  const isPaper = connection?.environment === 'paper';
  const save = async () => {
    try {
      setError('');
      if (!draft.id.trim() || !draft.label.trim() || !draft.adapter_id.trim()) return;
      const saved = existing
        ? await venues.update.mutateAsync({ id: draft.id, body: { ...draft, expected_revision: revision } })
        : await venues.create.mutateAsync({ ...draft, expected_revision: revision });
      onSaved?.(saved.connection);
      setSavedNeedsReload(saved.savedNeedsReload);
    } catch {
      setError(t('connection.saveFailed'));
    }
  };
  const saveCredentials = async () => {
    if (!connection || !apiKey || !secret) return;
    try {
      setError('');
      setCredentialSaving(true);
      await venues.putCredentials({
        id: connection.id,
        expectedRevision: revision,
        credentials: { api_key: apiKey, secret, ...(passphrase ? { passphrase } : {}) },
      });
      setApiKey('');
      setSecret('');
      setPassphrase('');
    } catch {
      setError(t('connection.accessSaveFailed'));
    } finally {
      setCredentialSaving(false);
    }
  };
  const test = async () => {
    try {
      setError('');
      const health = await venues.test.mutateAsync(draft.id);
      if (health.healthy) tested?.(draft.id);
    } catch {
      setError(t('connection.testFailed'));
    }
  };
  return (
    <form
      className="grid gap-3 rounded-xl border border-border bg-muted/10 p-4"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      <div className="grid gap-3 md:grid-cols-3">
        <label className="text-xs text-muted-foreground">
          {t('connection.id')}
          <input
            aria-label={t('connection.id')}
            disabled={existing}
            value={draft.id}
            onChange={(event) => setDraft({ ...draft, id: event.target.value })}
            className="mt-1 h-10 w-full rounded border bg-background px-3"
          />
        </label>
        <label className="text-xs text-muted-foreground">
          {t('connection.name')}
          <input
            aria-label={t('connection.name')}
            value={draft.label}
            onChange={(event) => setDraft({ ...draft, label: event.target.value })}
            className="mt-1 h-10 w-full rounded border bg-background px-3"
          />
        </label>
        <label className="text-xs text-muted-foreground">
          {t('connection.adapter')}
          <input
            aria-label={t('connection.adapter')}
            list="venue-adapters"
            value={draft.adapter_id}
            onChange={(event) => setDraft({ ...draft, adapter_id: event.target.value })}
            className="mt-1 h-10 w-full rounded border bg-background px-3"
          />
        </label>
        <datalist id="venue-adapters">
          <option value="paper" />
          <option value="okx" />
          <option value="bybit" />
        </datalist>
        <label className="text-xs text-muted-foreground">
          {t('connection.environment')}
          <select
            aria-label={t('connection.environment')}
            disabled={existing}
            value={draft.environment}
            onChange={(event) => setDraft({ ...draft, environment: event.target.value as Environment })}
            className="mt-1 h-10 w-full rounded border bg-background px-3"
          >
            <option value="paper">paper</option>
            <option value="demo">demo</option>
            <option value="testnet">testnet</option>
            <option value="live">live</option>
          </select>
        </label>
        <label className="text-xs text-muted-foreground">
          {t('connection.leverage')}
          <input
            aria-label={t('connection.leverage')}
            type="number"
            min="1"
            value={draft.leverage}
            onChange={(event) => setDraft({ ...draft, leverage: Number(event.target.value) })}
            className="mt-1 h-10 w-full rounded border bg-background px-3"
          />
        </label>
        <label className="flex items-center gap-2 pt-5 text-sm">
          <input
            type="checkbox"
            checked={draft.enabled}
            onChange={(event) => setDraft({ ...draft, enabled: event.target.checked })}
          />
          {t('connection.enabled')}
        </label>
      </div>
      {existing && !isPaper ? (
        <div className="grid gap-3 border-t border-border pt-3 md:grid-cols-3">
          <label className="text-xs text-muted-foreground">
            {t('connection.apiAccessId')}
            <input
              aria-label={t('connection.apiAccessId')}
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              autoComplete="off"
              className="mt-1 h-10 w-full rounded border bg-background px-3"
            />
          </label>
          <label className="text-xs text-muted-foreground">
            {t('connection.apiSigningPhrase')}
            <input
              aria-label={t('connection.apiSigningPhrase')}
              type="password"
              value={secret}
              onChange={(event) => setSecret(event.target.value)}
              autoComplete="new-password"
              className="mt-1 h-10 w-full rounded border bg-background px-3"
            />
          </label>
          <label className="text-xs text-muted-foreground">
            {t('connection.accessPhrase')}
            <input
              aria-label="Passphrase"
              type="password"
              value={passphrase}
              onChange={(event) => setPassphrase(event.target.value)}
              autoComplete="new-password"
              className="mt-1 h-10 w-full rounded border bg-background px-3"
            />
          </label>
          <div className="flex gap-2 md:col-span-3">
            <Button
              type="button"
              variant="outline"
              disabled={!apiKey || !secret || credentialSaving}
              onClick={() => void saveCredentials()}
            >
              <Save className="h-4 w-4" />
              {t('connection.saveAccess')}
            </Button>
            <Button type="button" variant="outline" disabled={venues.test.isPending} onClick={() => void test()}>
              <FlaskConical className="h-4 w-4" />
              {t('connection.test')}
            </Button>
          </div>
        </div>
      ) : null}
      {existing && isPaper ? (
        <div className="flex items-center gap-2 border-t border-border pt-3 text-sm text-muted-foreground">
          <span>{t('connection.paperNoAccess')}</span>
          <Button type="button" variant="outline" disabled={venues.test.isPending} onClick={() => void test()}>
            <FlaskConical className="h-4 w-4" />
            {t('connection.test')}
          </Button>
        </div>
      ) : null}
      {error ? (
        <p role="alert" className="text-sm text-trade-short">
          {error}
        </p>
      ) : null}
      {savedNeedsReload ? <p role="status" className="text-sm text-amber-500">{t('connection.savedNeedsReload')}</p> : null}
      <Button type="submit" disabled={venues.create.isPending || venues.update.isPending || savedNeedsReload}>
        <Save className="h-4 w-4" />
        {existing ? t('connection.save') : t('connection.create')}
      </Button>
    </form>
  );
};
