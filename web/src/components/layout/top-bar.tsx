import { Check, Globe, Languages, Menu, Moon, Sun, SunMoon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { env } from '@/lib/env';
import { useSettingsStore } from '@/stores/use-settings-store';
import { useUIStore, type Locale, type Theme } from '@/stores/use-ui-store';

const THEME_OPTIONS: { value: Theme; icon: typeof Sun; labelKey: 'theme.light' | 'theme.dark' | 'theme.system' }[] = [
  { value: 'light', icon: Sun, labelKey: 'theme.light' },
  { value: 'dark', icon: Moon, labelKey: 'theme.dark' },
  { value: 'system', icon: SunMoon, labelKey: 'theme.system' },
];

const LOCALE_OPTIONS: { value: Locale; labelKey: 'locale.zh-CN' | 'locale.en-US' }[] = [
  { value: 'zh-CN', labelKey: 'locale.zh-CN' },
  { value: 'en-US', labelKey: 'locale.en-US' },
];

const ApiKeyBadge = () => {
  const apiKey = useSettingsStore((s) => s.apiKey);
  const present = (apiKey && apiKey.length > 0) || env.VITE_API_KEY.length > 0;
  return (
    <span
      className={
        present
          ? 'rounded-full bg-success/10 px-2 py-0.5 text-[10px] font-medium text-success'
          : 'rounded-full bg-warning/10 px-2 py-0.5 text-[10px] font-medium text-warning'
      }
      title="X-API-Key"
    >
      {present ? 'API ✓' : 'API ✗'}
    </span>
  );
};

export const TopBar = () => {
  const { t } = useTranslation();
  const theme = useUIStore((s) => s.theme);
  const setTheme = useUIStore((s) => s.setTheme);
  const locale = useUIStore((s) => s.locale);
  const setLocale = useUIStore((s) => s.setLocale);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-card px-4">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="icon" onClick={toggleSidebar} aria-label="toggle sidebar">
          <Menu className="h-4 w-4" />
        </Button>
        <ApiKeyBadge />
      </div>
      <div className="flex items-center gap-1">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" aria-label={t('theme.system')}>
              <SunMoon className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>
              <Globe className="mr-2 inline h-3.5 w-3.5" /> Theme
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            {THEME_OPTIONS.map((opt) => (
              <DropdownMenuItem key={opt.value} onSelect={() => setTheme(opt.value)}>
                <opt.icon className="mr-2 h-4 w-4" />
                <span className="flex-1">{t(opt.labelKey)}</span>
                {theme === opt.value ? <Check className="ml-2 h-3.5 w-3.5" /> : null}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" aria-label={t('locale.zh-CN')}>
              <Languages className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Language</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {LOCALE_OPTIONS.map((opt) => (
              <DropdownMenuItem key={opt.value} onSelect={() => setLocale(opt.value)}>
                <span className="flex-1">{t(opt.labelKey)}</span>
                {locale === opt.value ? <Check className="ml-2 h-3.5 w-3.5" /> : null}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
};
