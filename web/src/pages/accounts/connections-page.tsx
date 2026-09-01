import { useState } from 'react';
import { Link } from 'react-router';
import { useTranslation } from 'react-i18next';
import { ConnectionForm } from './connection-form';
import { useConfiguration } from '@/pages/settings/configuration-context';

export default function ConnectionsPage() {
  const { t } = useTranslation('configuration');
  const runtime = useConfiguration();
  const [adding, setAdding] = useState<true | string | false>(false);
  const addingForm = adding || runtime.newVenueDraft;
  const connections = runtime.baseline?.execution.connections ?? [];
  return (
    <section className="space-y-6">
      <header id="execution.connections">
        <h1 className="text-xl font-semibold">{t('venues')}</h1>
        <p className="configuration-help">{t('venuesSubtitle')}</p>
        <Link className="configuration-button" to="/accounts">
          查看账户与收益
        </Link>
        <button
          className="configuration-button"
          type="button"
          disabled={runtime.isReloading}
          onClick={() => void runtime.reload()}
        >
          {t('reload')}
        </button>
      </header>
      {runtime.conflict ? (
        <p role="alert" className="configuration-error">
          {t('forms.conflictHelp')}
        </p>
      ) : null}
      {runtime.failure ? (
        <p role="alert" className="configuration-error">
          {runtime.failure}
        </p>
      ) : null}
      {connections
        .filter((connection) => connection.id !== adding)
        .map((connection) => (
          <section id={`execution.connections.${connection.id}`} key={connection.id} className="configuration-section">
            <h2 className="text-base font-semibold">{connection.label}</h2>
            <Link className="text-primary underline" to={`/accounts/connections/${encodeURIComponent(connection.id)}`}>
              查看账户详情
            </Link>
            {runtime.checks[connection.id]?.health.healthy &&
            !runtime.baseline?.execution.books.some(
              (book) =>
                book.enabled &&
                book.allocations.some((allocation) => allocation.enabled && allocation.connection_id === connection.id),
            ) ? (
              <p className="configuration-help">已连接，未参与执行</p>
            ) : null}
            <ConnectionForm connectionId={connection.id} onSaved={() => undefined} />
          </section>
        ))}
      {addingForm ? (
        <section className="configuration-section">
          <h2 className="text-base font-semibold">{t('addVenue')}</h2>
          <ConnectionForm
            {...(typeof adding === 'string' ? { connectionId: adding } : {})}
            onSaved={(id) => setAdding((current) => (current === id ? false : id))}
          />
        </section>
      ) : (
        <button className="configuration-button" type="button" onClick={() => setAdding(true)}>
          {t('addConnection')}
        </button>
      )}
    </section>
  );
}
