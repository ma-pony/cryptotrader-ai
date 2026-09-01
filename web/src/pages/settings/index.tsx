import { useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router';
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
  const navigate = useNavigate();
  return (
    <div className="configuration-workbench">
      <header className="configuration-center-header">
        <span>{t('center.systemHub')} · {runtime.revision ? t('revisionValue', { revision: runtime.revision }) : '配置版本未知'}</span>
      </header>
      <div className="configuration-mobile-navigation configuration-field">
        <label htmlFor="configuration-section">{t('center.section')}</label>
        <select
          id="configuration-section"
          className="configuration-control"
          value={pathname}
          onChange={(event) => void navigate(event.target.value)}
        >
          {SETTINGS_SECTIONS.map((section) => (
            <option key={section.id} value={section.path}>
              {t(section.label)}
            </option>
          ))}
        </select>
      </div>
      <nav className="configuration-navigation" aria-label={t('center.systemHub')}>
        {SETTINGS_SECTIONS.map((section) => (
          <NavLink key={section.id} to={section.path}>
            {t(section.label)}
          </NavLink>
        ))}
      </nav>
      <Outlet />
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
      submit
      onSave={() => void runtime.save(section, form ?? undefined)}
      onDiscard={() => runtime.discard(section)}
      onReload={() => void runtime.reload()}
    />
  );
}

export default function SettingsPage({ section, componentId }: { section: Exclude<ConfigurationSection, 'books' | 'notifications'>; componentId?: string }) {
  const runtime = useConfiguration();
  const form = useRef<HTMLFormElement>(null);
  const { t } = useTranslation('configuration');
  const document = runtime.document;
  const catalog = runtime.catalog.data;
  if (!document || !catalog) return null;
  const errors = runtime.errors;
  const secret = (kind: 'llmGateway' | 'apiAccess' | 'newsProvider') => ({
    ...runtime.secretStates[kind],
    revision: runtime.revision ?? 0,
  });
  return (
    <form
      id={section === 'models' ? 'models' : section === 'system' ? 'system-settings' : undefined}
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
          {...(componentId ? { componentId } : {})}
        />
      ) : null}
      {section === 'market' ? (
        <MarketSettings
          value={document.market_data}
          secrets={secret('newsProvider')}
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
              if (patch.accounts) runtime.update('accounts', patch.accounts);
              if (patch.infrastructure) runtime.update('infrastructure', patch.infrastructure);
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
