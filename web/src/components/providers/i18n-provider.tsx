import { useEffect, type ReactNode } from 'react';
import { I18nextProvider } from 'react-i18next';

import i18n, { setLocaleAndPersist } from '@/lib/i18n';
import { useUIStore } from '@/stores/use-ui-store';

export const I18nProvider = ({ children }: { children: ReactNode }) => {
  const locale = useUIStore((s) => s.locale);

  useEffect(() => {
    if (i18n.language !== locale) setLocaleAndPersist(locale);
    document.documentElement.lang = locale;
  }, [locale]);

  return <I18nextProvider i18n={i18n}>{children}</I18nextProvider>;
};
