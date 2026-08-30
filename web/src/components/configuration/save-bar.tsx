import { useTranslation } from 'react-i18next';
import type { SaveStatus } from '@/hooks/use-configuration-draft';

export type SaveBarProps = {
  dirty: boolean;
  status?: SaveStatus | undefined;
  failure?: string | undefined;
  applyStatus?: 'pending' | 'applied' | 'failed' | undefined;
  conflict?: boolean;
  loading?: boolean;
  onSave: () => void;
  onDiscard: () => void;
  onReload?: () => void;
};
export function SaveBar({
  dirty,
  status,
  failure,
  applyStatus,
  conflict,
  loading,
  onSave,
  onDiscard,
  onReload,
}: SaveBarProps) {
  const { t } = useTranslation('configuration');
  const pending = status === 'saving' || loading;
  const message =
    failure ||
    (conflict
      ? t('forms.conflictHelp')
      : pending
        ? t('forms.saving')
        : dirty
          ? t('forms.unsaved')
          : status === 'saved'
            ? t('forms.saved')
            : t('forms.noChanges'));
  return (
    <div className="configuration-save-bar">
      <div role="status" aria-live="polite">
        <p className={failure || conflict ? 'configuration-error' : ''}>{message}</p>
        {applyStatus === 'pending' ? (
          <p className="configuration-help">{t('apply.pending')}</p>
        ) : applyStatus === 'failed' ? (
          <p className="configuration-error">{t('forms.applyFailed')}</p>
        ) : status === 'saved' && !dirty ? (
          <p className="configuration-help">{t('forms.savedNotActivated')}</p>
        ) : null}
      </div>
      <div className="configuration-actions">
        {onReload ? (
          <button type="button" className="configuration-button" disabled={pending} onClick={onReload}>
            {t('reload')}
          </button>
        ) : null}
        <button
          type="button"
          className="configuration-button"
          disabled={pending || !dirty}
          onClick={() => {
            if (!dirty || window.confirm(t('forms.discardConfirm'))) onDiscard();
          }}
        >
          {t('forms.discard')}
        </button>
        <button
          type="button"
          className="configuration-button configuration-primary"
          disabled={pending || conflict || !dirty}
          onClick={onSave}
        >
          {pending ? t('forms.saving') : t('forms.saveSection')}
        </button>
      </div>
    </div>
  );
}
