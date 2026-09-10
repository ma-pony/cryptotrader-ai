import { useState } from 'react';
import { PageHeader } from '@/components/ui/page-header';
import { Button } from '@/components/ui/button';
import { useTranslation } from 'react-i18next';
import { TextField } from '@/components/configuration/field';
import { RouteSkeleton } from '@/components/route-skeleton';
import { useSettingsStore } from '@/stores/use-settings-store';
import type { useRuntimeConfig } from '@/hooks/use-runtime-config';

export function ConfigurationAccess({
  runtime,
  embedded = false,
}: {
  runtime: Pick<ReturnType<typeof useRuntimeConfig>, 'isLoading' | 'authenticationRequired' | 'reload'>;
  embedded?: boolean;
}) {
  const { t } = useTranslation('configuration');
  const [key, setKey] = useState('');
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState(false);
  const showKey = runtime.authenticationRequired || pending || failed;
  const unlock = async () => {
    if (!key.trim() || pending) return;
    useSettingsStore.getState().setApiKey(key);
    setKey('');
    setPending(true);
    setFailed(false);
    try {
      const result = await runtime.reload();
      if (result.isError) {
        useSettingsStore.getState().setApiKey('');
        setFailed(true);
      }
    } finally {
      setPending(false);
    }
  };
  if (runtime.isLoading && !showKey) return <RouteSkeleton />;
  const content = (
    <div
      className={
        embedded
          ? 'grid min-h-48 place-items-center rounded-lg border border-border p-6'
          : 'grid min-h-screen place-items-center p-6'
      }
    >
      {showKey ? (
        <form
          className="w-full max-w-md space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            void unlock();
          }}
        >
          <PageHeader title={t('access.title')} />
          <TextField
            name="existing-api-access"
            label={t('access.existing')}
            help={t('access.help')}
            type="password"
            value={key}
            required
            disabled={pending}
            onChange={setKey}
          />
          {failed ? (
            <p role="alert" className="configuration-error">
              {t('access.failed')}
            </p>
          ) : null}
          <Button type="submit" disabled={pending || !key.trim()}>
            {t(pending ? 'access.loading' : 'access.unlock')}
          </Button>
        </form>
      ) : (
        <Button variant="outline" onClick={() => void runtime.reload()}>
          {t('loadError')} · {t('retry')}
        </Button>
      )}
    </div>
  );
  return embedded ? content : <main>{content}</main>;
}
