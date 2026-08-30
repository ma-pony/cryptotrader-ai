import { useState } from 'react';
import { Link } from 'react-router';
import { useTranslation } from 'react-i18next';
import { useQueryClient } from '@tanstack/react-query';
import {
  CONFIGURATION_SECTION_KEYS,
  validateConfigurationSection,
  type ConfigurationSection,
} from '@/hooks/use-configuration-draft';
import { RUNTIME_CONFIG_QUERY_KEY, toRuntimeDocument } from '@/hooks/use-runtime-config';
import { connectionChecksReady } from '@/lib/configuration-readiness';
import { useConfiguration } from '@/pages/settings/configuration-context';
import { SETTINGS_SECTIONS } from '@/pages/settings/navigation';
import type { RuntimeConfig } from '@/types/api';

export default function SetupPage() {
  const runtime = useConfiguration();
  const client = useQueryClient();
  const { t } = useTranslation('configuration');
  const [failure, setFailure] = useState(false);
  const document = runtime.baseline;
  if (!document) return null;
  const confirmedActive =
    !runtime.conflict && document.system.active && runtime.applyStatus === 'applied' && runtime.appliedRevision === runtime.revision;
  // Readiness validates the intended active state; inactive section saves remain partial.
  const activationDocument = { ...document, system: { ...document.system, active: true } };
  const errors = (section: ConfigurationSection) =>
    validateConfigurationSection(section, activationDocument, runtime.catalog.data, (key) => t('forms.validation.' + key));
  const valid = (section: ConfigurationSection) =>
    !Object.keys(errors(section)).length;
  const checked = connectionChecksReady(document, runtime.checks, runtime.credentialStates);
  const accessReady =
    (!document.security.enabled || runtime.secretStates.apiAccess.configured) &&
    (!document.signals.components.some((item) => item.enabled && item.component_id === 'llm_committee') ||
      runtime.secretStates.llmGateway.configured);
  const ready =
    checked &&
    accessReady &&
    document.execution.books.some((book) => book.enabled) &&
    (Object.keys(CONFIGURATION_SECTION_KEYS) as ConfigurationSection[]).every(valid) &&
    !runtime.hasUnsaved &&
    !runtime.conflict;
  const activate = async () => {
    if (!ready || runtime.isSaving || runtime.isReloading || failure) return;
    const latest = client.getQueryData<RuntimeConfig>(RUNTIME_CONFIG_QUERY_KEY);
    if (!latest || latest.revision !== runtime.revision) return;
    const saved = toRuntimeDocument(latest.document);
    try {
      setFailure(false);
      await runtime.replace({ ...saved, system: { ...saved.system, active: true } }, latest.revision);
    } catch {
      setFailure(true);
    }
  };
  const reload = async () => {
    const result = await runtime.reload();
    if (!result.error) setFailure(false);
  };
  return (
    <section className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">{t('setup')}</h1>
        <p className="configuration-help">{t('center.setupHelp')}</p>
      </header>
      <nav aria-label={t('center.checklist')}>
        <ol className="configuration-checklist">
          {SETTINGS_SECTIONS.map((section) => {
            const sectionReady = section.id === 'venues' ? checked : valid(section.id) && !runtime.isDirty(section.id);
            return (
              <li key={section.id}>
                <Link to={section.path}>{t(section.label)}</Link>
                <span className="configuration-help">
                  {t(sectionReady ? 'center.savedReady' : 'center.reviewRequired')}
                </span>
                {section.id === 'system' && errors('system')['infrastructure.redis_url'] ? (
                  <p className="configuration-error">{t('center.redisRequired')}</p>
                ) : null}
              </li>
            );
          })}
        </ol>
      </nav>
      <p className="configuration-help">{t('center.activationHelp')}</p>
      {!accessReady ? <p className="configuration-error">{t('center.accessRequired')}</p> : null}
      {runtime.hasUnsaved ? <p className="configuration-error">{t('center.saveBeforeActivation')}</p> : null}
      {runtime.conflict || failure ? (
        <p role="alert" className="configuration-error">
          {t(runtime.conflict ? 'forms.conflictHelp' : 'center.activationFailed')}
        </p>
      ) : null}
      {confirmedActive ? (
        <p role="status">{t('center.active')}</p>
      ) : (
        <div className="space-y-3">
          {document.system.active ? (
            <p role="status" className="configuration-help">
              {t(runtime.conflict ? 'center.activationUnconfirmed' : runtime.applyStatus === 'failed' ? 'center.activationApplyFailed' : 'center.activationPending')}
            </p>
          ) : null}
          <div className="configuration-actions">
            <button
              className="configuration-button configuration-primary"
              disabled={!ready || runtime.isSaving || runtime.isReloading || failure}
              onClick={() => void activate()}
            >
              {t(document.system.active ? 'center.retryActivation' : 'center.activate')}
            </button>
            <button
              className="configuration-button"
              disabled={runtime.isSaving || runtime.isReloading}
              onClick={() => void reload()}
            >
              {t('reload')}
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
