import { useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, NavLink, Outlet, useLocation } from 'react-router';
import { SaveBar } from '@/components/configuration/save-bar';
import type { ConfigurationSection } from '@/hooks/use-configuration-draft';
import { ModelSettings } from './forms/model-settings';
import { SignalSettings } from './forms/signal-settings';
import { MarketSettings } from './forms/market-settings';
import { RiskSettings } from './forms/risk-settings';
import { SchedulerSettings } from './forms/scheduler-settings';
import { SystemSettings } from './forms/system-settings';
import { useConfiguration } from './configuration-context';
import { SETTINGS_SECTIONS } from './navigation';

export function ConfigurationLayout() {
  const runtime = useConfiguration();
  const { t } = useTranslation('configuration');
  const { pathname } = useLocation();
  return (
    <div className="configuration-workbench">
      <header className="configuration-center-header">
        <span>
          {t('center.title')} · {t('revisionValue', { revision: runtime.revision })}
        </span>
        {pathname !== '/setup' ? <Link to="/setup">{t('center.checklist')}</Link> : null}
      </header>
      {pathname !== '/setup' ? (
        <nav className="configuration-navigation" aria-label={t('center.title')}>
          {SETTINGS_SECTIONS.map((section) => (
            <NavLink key={section.id} to={section.path}>
              {t(section.label)}
            </NavLink>
          ))}
        </nav>
      ) : null}
      {runtime.catalog.isPending ? (
        <p role="status">{t('center.catalogLoading')}</p>
      ) : runtime.catalog.isError ? (
        <div role="alert">
          <p>{t('center.catalogFailed')}</p>
          <button className="configuration-button" onClick={() => void runtime.catalog.refetch()}>
            {t('retry')}
          </button>
        </div>
      ) : (
        <Outlet />
      )}
    </div>
  );
}

export function ConfigurationSaveBar({
  section,
  form,
}: {
  section: ConfigurationSection;
  form?: HTMLFormElement | null;
}) {
  const runtime = useConfiguration();
  return (
    <SaveBar
      dirty={runtime.isDirty(section)}
      status={runtime.status[section]}
      failure={runtime.failure}
      applyStatus={runtime.applyStatus}
      conflict={runtime.conflict}
      loading={runtime.isSaving || runtime.isReloading}
      onSave={() => void runtime.save(section, form ?? undefined)}
      onDiscard={() => runtime.discard(section)}
      onReload={() => void runtime.reload()}
    />
  );
}

export default function SettingsPage({ section }: { section: Exclude<ConfigurationSection, 'books'> }) {
  const runtime = useConfiguration();
  const form = useRef<HTMLFormElement>(null);
  const { t } = useTranslation('configuration');
  const document = runtime.document;
  const catalog = runtime.catalog.data;
  if (!document || !catalog) return null;
  const errors = runtime.errors;
  const secret = (kind: 'llmGateway' | 'apiAccess') => ({
    ...runtime.secretStates[kind],
    revision: runtime.revision ?? 0,
  });
  return (
    <form
      ref={form}
      noValidate
      onSubmit={(event) => {
        event.preventDefault();
        void runtime.save(section, form.current ?? undefined);
      }}
    >
      {section === 'models' ? (
        <ModelSettings
          value={document.llm}
          onChange={(value) => runtime.update('llm', value)}
          errors={errors}
          secrets={secret('llmGateway')}
        />
      ) : null}
      {section === 'signals' ? (
        <SignalSettings
          value={document.signals}
          onChange={(value) => runtime.update('signals', value)}
          errors={errors}
          catalog={catalog}
        />
      ) : null}
      {section === 'market' ? (
        <MarketSettings
          value={document.market_data}
          onChange={(value) => runtime.update('market_data', value)}
          errors={errors}
          catalog={catalog}
        />
      ) : null}
      {section === 'risk' ? (
        <RiskSettings
          value={document.risk}
          onChange={(value) => runtime.update('risk', value)}
          hitl={document.hitl}
          onHitlChange={(value) => runtime.update('hitl', value)}
          errors={errors}
        />
      ) : null}
      {section === 'scheduler' ? (
        <SchedulerSettings
          value={document.scheduler}
          onChange={(value) => runtime.update('scheduler', value)}
          triggers={document.triggers}
          onTriggersChange={(value) => runtime.update('triggers', value)}
          errors={errors}
        />
      ) : null}
      {section === 'system' ? (
        <>
          <h1 className="text-xl font-semibold">{t('center.systemTitle')}</h1>
          <SystemSettings
            value={document}
            onChange={(patch) => {
              if (patch.security) runtime.update('security', patch.security);
              if (patch.infrastructure) runtime.update('infrastructure', patch.infrastructure);
              if (patch.notifications) runtime.update('notifications', patch.notifications);
              if (patch.observability) runtime.update('observability', patch.observability);
            }}
            errors={errors}
            secrets={secret('apiAccess')}
          />
        </>
      ) : null}
      <ConfigurationSaveBar section={section} form={form.current} />
    </form>
  );
}
