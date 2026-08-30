import { useTranslation } from 'react-i18next';
import { useConfiguration } from '../configuration-context';
import { VenueForm, newVenue, type VenueDraft } from './venue-form';
export default function VenuesPage() {
  const runtime = useConfiguration();
  const { t } = useTranslation('configuration');
  const catalog = runtime.catalog.data;
  if (!runtime.baseline || !catalog) return null;
  const clear = (key: string) =>
    runtime.setVenueDrafts((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
  const connections = runtime.baseline.execution.connections;
  const update = (key: string, value: VenueDraft) => {
    const baseline = connections.find((item) => item.id === key);
    if (baseline && JSON.stringify(baseline) === JSON.stringify(value)) clear(key);
    else runtime.setVenueDrafts((current) => ({ ...current, [key]: value }));
  };
  return (
    <section className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold">{t('venues')}</h1>
        <p className="configuration-help">{t('venuesSubtitle')}</p>
      </header>
      {!connections.length ? <p className="configuration-help">{t('connection.empty')}</p> : null}
      {connections.map((connection) => (
        <section key={connection.id} className="configuration-section">
          <h2 className="text-base font-semibold">{connection.label}</h2>
          <VenueForm
            connection={connection}
            value={runtime.venueDrafts[connection.id] ?? connection}
            catalog={catalog}
            revision={runtime.revision ?? 0}
            credentialState={runtime.credentialStates[connection.id]}
            check={runtime.checks[connection.id]}
            onChange={(value) => update(connection.id, value)}
            onSaved={() => clear(connection.id)}
            onChecked={(check) =>
              runtime.setChecks((current) => {
                const next = { ...current };
                if (check) next[connection.id] = check;
                else delete next[connection.id];
                return next;
              })
            }
            writeBlocked={runtime.conflict}
            onCancel={
              runtime.venueDrafts[connection.id]
                ? () => {
                    if (window.confirm(t('forms.discardConfirm'))) clear(connection.id);
                  }
                : undefined
            }
          />
        </section>
      ))}
      {runtime.venueDrafts.new ? (
        <section className="configuration-section">
          <h2 className="text-base font-semibold">{t('addVenue')}</h2>
          <VenueForm
            value={runtime.venueDrafts.new}
            catalog={catalog}
            revision={runtime.revision ?? 0}
            onChange={(value) => update('new', value)}
            onSaved={() => clear('new')}
            onChecked={() => {}}
            writeBlocked={runtime.conflict}
            onCancel={() => {
              if (window.confirm(t('forms.discardConfirm'))) clear('new');
            }}
          />
        </section>
      ) : (
        <button
          className="configuration-button"
          disabled={!catalog.venues.length}
          onClick={() => update('new', newVenue(catalog))}
        >
          {t('addConnection')}
        </button>
      )}
      <button className="configuration-button" disabled={runtime.isReloading} onClick={() => void runtime.reload()}>
        {t('reload')}
      </button>
      {runtime.failure || runtime.conflict ? (
        <p role="alert" className="configuration-error">
          {runtime.failure ?? t('forms.conflictHelp')}
        </p>
      ) : null}
    </section>
  );
}
