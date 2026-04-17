import { useEffect, type ReactNode } from 'react';

import { useUIStore, type Theme } from '@/stores/use-ui-store';

const STORAGE_KEY = 'cryptotrader-ui';

const resolveSystemPref = (): 'light' | 'dark' => {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

const applyTheme = (theme: Theme) => {
  if (typeof document === 'undefined') return;
  const effective = theme === 'system' ? resolveSystemPref() : theme;
  const root = document.documentElement;
  root.dataset.theme = effective;
  root.style.colorScheme = effective;
};

export const themeBootstrapScript = `(function() {
  try {
    var raw = localStorage.getItem('${STORAGE_KEY}');
    var theme = 'system';
    if (raw) {
      var parsed = JSON.parse(raw);
      if (parsed && parsed.state && parsed.state.theme) theme = parsed.state.theme;
    }
    var effective = theme;
    if (effective === 'system') {
      effective = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    document.documentElement.dataset.theme = effective;
    document.documentElement.style.colorScheme = effective;
  } catch (e) {}
})();`;

export const ThemeProvider = ({ children }: { children: ReactNode }) => {
  const theme = useUIStore((s) => s.theme);

  useEffect(() => {
    applyTheme(theme);
    if (theme !== 'system') return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => applyTheme('system');
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, [theme]);

  return <>{children}</>;
};
