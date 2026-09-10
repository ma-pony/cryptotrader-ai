import { useTranslation } from 'react-i18next';
import type { SaveStatus } from '@/hooks/use-configuration-draft';
import { Button } from '@/components/ui/button';
import { LoaderCircle, Save } from 'lucide-react';

export type SaveBarProps = {
  dirty: boolean;
  status?: SaveStatus | undefined;
  failure?: string | undefined;
  applyStatus?: 'pending' | 'applied' | 'failed' | undefined;
  conflict?: boolean;
  loading?: boolean;
  submit?: boolean;
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
  submit = false,
  onSave,
  onDiscard,
  onReload,
}: SaveBarProps) {
  const { t } = useTranslation('configuration');
  const pending = status === 'saving' || loading;
  const message = pending
    ? t('forms.saving')
    : failure ||
      (conflict
        ? t('forms.conflictHelp')
        : dirty
          ? t('forms.unsaved')
          : status === 'saved'
            ? t('forms.saved')
            : t('forms.noChanges'));
  return (
    <div className="configuration-save-bar">
      <div role="status" aria-live="polite">
        <p className={!pending && (failure || conflict) ? 'configuration-error' : 'font-medium'}>{message}</p>
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
          <Button variant="ghost" disabled={pending} onClick={onReload}>
            {t('reload')}
          </Button>
        ) : null}
        <Button
          type="button"
          variant="outline"
          disabled={pending || !dirty}
          onClick={() => {
            if (!dirty || window.confirm(t('forms.discardConfirm'))) onDiscard();
          }}
        >
          {t('forms.discard')}
        </Button>
        <Button
          type={submit ? 'submit' : 'button'}
          disabled={pending || conflict || !dirty}
          onClick={submit ? undefined : onSave}
        >
          {pending ? (
            <LoaderCircle className="h-4 w-4 motion-safe:animate-spin" aria-hidden="true" />
          ) : (
            <Save className="h-4 w-4" aria-hidden="true" />
          )}
          {pending ? t('forms.saving') : t('forms.saveSection')}
        </Button>
      </div>
    </div>
  );
}
