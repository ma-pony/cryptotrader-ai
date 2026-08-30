import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { TextField } from '@/components/configuration/field';
import { RouteSkeleton } from '@/components/route-skeleton';
import { useSettingsStore } from '@/stores/use-settings-store';
import type { useRuntimeConfig } from '@/hooks/use-runtime-config';

export function ConfigurationAccess({
  runtime,
}: {
  runtime: Pick<ReturnType<typeof useRuntimeConfig>, 'isLoading' | 'authenticationRequired' | 'reload'>;
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
  return (
    <main className="grid min-h-screen place-items-center p-6">
      {showKey ? (
        <form
          className="w-full max-w-md space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            void unlock();
          }}
        >
          <h1 className="text-xl font-semibold">{t('access.title')}</h1>
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
          <button
            type="submit"
            className="configuration-button configuration-primary"
            disabled={pending || !key.trim()}
          >
            {t(pending ? 'access.loading' : 'access.unlock')}
          </button>
        </form>
      ) : (
        <button className="configuration-button" onClick={() => void runtime.reload()}>
          {t('loadError')} · {t('retry')}
        </button>
      )}
    </main>
  );
}
