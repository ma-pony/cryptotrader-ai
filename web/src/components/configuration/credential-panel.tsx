import { useId, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { formatDateTime } from '@/lib/format';

/** Displays public credential metadata only; secret inputs belong to the caller. */
export function CredentialPanel({
  title,
  configured,
  updatedAt,
  editing,
  pending,
  disabled,
  onEdit,
  onCancel,
  children,
}: {
  title: string;
  configured: boolean;
  updatedAt?: string | null | undefined;
  editing: boolean;
  pending: boolean;
  disabled: boolean;
  onEdit: () => void;
  onCancel: () => void;
  children: ReactNode;
}) {
  const { t } = useTranslation('configuration');
  const id = useId();
  const open = !configured || editing;
  return (
    <section aria-label={t('credentials.region', { name: title })} className="configuration-secret">
      <div className="configuration-secret-header">
        <div className="space-y-2">
          <h3>{title}</h3>
          <p role="status" className={configured ? 'configuration-secret-configured' : 'configuration-help'}>
            {t(configured ? 'forms.accessConfigured' : 'forms.accessMissing')}
          </p>
        </div>
        {configured ? (
          <button
            type="button"
            className="configuration-button"
            aria-expanded={open}
            aria-controls={`${id}-editor`}
            disabled={editing ? pending : disabled}
            onClick={editing ? onCancel : onEdit}
          >
            {t(editing ? 'credentials.cancel' : 'credentials.replace')}
          </button>
        ) : null}
      </div>
      {configured ? (
        <div className="space-y-1">
          <p className="configuration-help">{t('credentials.storedHelp')}</p>
          {updatedAt ? (
            <p className="configuration-help">
              {t('credentials.updatedAt')} <time dateTime={updatedAt}>{formatDateTime(updatedAt)}</time>
            </p>
          ) : null}
        </div>
      ) : null}
      <div id={`${id}-editor`} hidden={!open} className="configuration-secret-editor">
        {open ? (
          <>
            {configured ? <p className="configuration-help">{t('credentials.replaceHelp')}</p> : null}
            {children}
          </>
        ) : null}
      </div>
    </section>
  );
}
