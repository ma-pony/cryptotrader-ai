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
import { useMarketDataWS } from '@/hooks/use-market-data-ws';
import { useSettingsStore } from '@/stores/use-settings-store';
import { useUIStore, type Locale, type Theme } from '@/stores/use-ui-store';
import { Breadcrumb } from './breadcrumb';

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
  const { t } = useTranslation();
  const apiKey = useSettingsStore((s) => s.apiKey);
  const present = apiKey.length > 0;
  return (
    <span
      className={
        present
          ? 'rounded-full border border-trade-long/40 bg-trade-long-soft px-2 py-1 text-sm font-medium text-trade-long'
          : 'rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-sm font-medium text-amber-500'
      }
      title={t('header.accessCredential')}
    >
      {t(present ? 'header.keyPresent' : 'header.keyMissing')}
    </span>
  );
};

const BtcPriceDisplay = () => {
  const { tickerData } = useMarketDataWS('BTCUSDT');
  if (!tickerData) return null;
  const price = tickerData.price;
  const changePct = tickerData.priceChangePercent;
  if (!Number.isFinite(price)) return null;
  return (
    <div className="hidden items-center gap-1.5 md:flex">
      <span className="text-sm text-muted-foreground">BTC</span>
      <span className="font-mono text-sm font-semibold tabular-nums">
        ${price.toLocaleString('en-US', { maximumFractionDigits: 0 })}
      </span>
      {Number.isFinite(changePct) ? (
        <span className={changePct >= 0 ? 'font-mono text-sm text-trade-long' : 'font-mono text-sm text-trade-short'}>
          {changePct >= 0 ? '+' : ''}
          {changePct.toFixed(2)}%
        </span>
      ) : null}
    </div>
  );
};

export const TopBar = () => {
  const { t } = useTranslation();
  const { connectionStatus } = useMarketDataWS();
  const theme = useUIStore((s) => s.theme);
  const setTheme = useUIStore((s) => s.setTheme);
  const locale = useUIStore((s) => s.locale);
  const setLocale = useUIStore((s) => s.setLocale);
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);
  const setMobileNavOpen = useUIStore((s) => s.setMobileNavOpen);

  // Below md the sidebar is hidden; the menu button opens the mobile drawer.
  // At md+ the menu button toggles the desktop sidebar's collapsed state.
  const onMenuClick = () => {
    if (typeof window !== 'undefined' && window.matchMedia('(max-width: 767px)').matches) {
      setMobileNavOpen(true);
    } else {
      toggleSidebar();
    }
  };

  return (
    <header className="app-top-bar flex min-h-14 items-center justify-between gap-2 border-b border-border bg-card px-3 md:px-4">
      <div className="flex min-w-0 items-center gap-2 md:gap-3">
        <Button variant="ghost" size="icon" onClick={onMenuClick} aria-label={t('header.toggleSidebar')}>
          <Menu className="h-4 w-4" />
        </Button>
        <Breadcrumb />
      </div>

      <div className="flex shrink-0 items-center gap-1 md:gap-2">
        <BtcPriceDisplay />
        <span className="hidden h-3.5 w-px bg-border md:block" />
        <div className="hidden sm:block">
          <ApiKeyBadge />
        </div>
        <span className="hidden text-sm text-muted-foreground lg:inline">
          行情流：
          {connectionStatus === 'connected' ? '已连接' : connectionStatus === 'connecting' ? '连接中' : '未连接'}
        </span>

        <span className="mx-1 h-5 w-px bg-border" />

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" aria-label={`${t('theme.title')}：${t(`theme.${theme}`)}`}>
              <SunMoon className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>
              <Globe className="mr-2 inline h-3.5 w-3.5" /> {t('theme.title')}
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
            <Button variant="ghost" size="icon" aria-label={`${t('locale.title')}：${t(`locale.${locale}`)}`}>
              <Languages className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>{t('locale.title')}</DropdownMenuLabel>
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
