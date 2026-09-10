import { useState } from 'react';
import { Link } from 'react-router';
import { useTranslation } from 'react-i18next';
import { PageHeader } from '@/components/ui/page-header';
import { Button } from '@/components/ui/button';
import { EmptyState } from '@/components/ui/empty-state';
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
      <PageHeader
        id="execution.connections"
        title={t('venues')}
        subtitle={t('venuesSubtitle')}
        actions={
          <>
            <Button asChild variant="outline">
              <Link to="/accounts">查看账户与收益</Link>
            </Button>
            <Button variant="ghost" disabled={runtime.isReloading} onClick={() => void runtime.reload()}>
              {t('reload')}
            </Button>
          </>
        }
      />
      {!connections.length && !addingForm ? (
        <EmptyState
          title="尚未添加平台连接"
          description="选择已注册的平台与环境，再填写连接信息。可以从本地模拟器开始。"
        />
      ) : null}
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
        <Button onClick={() => setAdding(true)}>{t('addConnection')}</Button>
      )}
    </section>
  );
}
