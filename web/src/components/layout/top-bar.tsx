import { Check, ChevronRight, Globe, Languages, Menu, Moon, Sun, SunMoon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useLocation } from 'react-router';

import { StatusPill } from '@/components/ui/status-pill';
import { WSStatusIndicator } from '@/components/ws-status-indicator';
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

const THEME_OPTIONS: { value: Theme; icon: typeof Sun; labelKey: 'theme.light' | 'theme.dark' | 'theme.system' }[] = [
  { value: 'light', icon: Sun, labelKey: 'theme.light' },
  { value: 'dark', icon: Moon, labelKey: 'theme.dark' },
  { value: 'system', icon: SunMoon, labelKey: 'theme.system' },
];

const LOCALE_OPTIONS: { value: Locale; labelKey: 'locale.zh-CN' | 'locale.en-US' }[] = [
  { value: 'zh-CN', labelKey: 'locale.zh-CN' },
  { value: 'en-US', labelKey: 'locale.en-US' },
];

const PATH_LABELS: Record<string, 'nav.dashboard' | 'nav.decisions' | 'nav.debate' | 'nav.backtest' | 'nav.risk' | 'nav.metrics' | 'nav.chat' | 'nav.market' | 'nav.scheduler'> = {
  '/': 'nav.dashboard',
  '/decisions': 'nav.decisions',
  '/debate': 'nav.debate',
  '/backtest': 'nav.backtest',
  '/risk': 'nav.risk',
  '/metrics': 'nav.metrics',
  '/chat': 'nav.chat',
  '/market': 'nav.market',
  '/scheduler': 'nav.scheduler',
};

const ApiKeyBadge = () => {
  const apiKey = useSettingsStore((s) => s.apiKey);
  // SEC-I3: presence reflects only the in-memory store; VITE_API_KEY no longer
  // contributes to runtime auth (forbidden in production builds).
  const present = apiKey.length > 0;
  return (
    <span
      className={
        present
          ? 'rounded-full border border-trade-long/40 bg-trade-long-soft px-2 py-0.5 text-[10px] font-medium text-trade-long'
          : 'rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-500'
      }
      title="X-API-Key"
    >
      {present ? 'API ✓' : 'API ✗'}
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
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">BTC</span>
      <span className="font-mono text-xs font-semibold tabular-nums">
        ${price.toLocaleString('en-US', { maximumFractionDigits: 0 })}
      </span>
      {Number.isFinite(changePct) ? (
        <span
          className={
            changePct >= 0
              ? 'font-mono text-[10px] text-trade-long'
              : 'font-mono text-[10px] text-trade-short'
          }
        >
          {changePct >= 0 ? '+' : ''}
          {changePct.toFixed(2)}%
        </span>
      ) : null}
    </div>
  );
};

const Breadcrumb = () => {
  const { t } = useTranslation();
  const { pathname } = useLocation();
  const segments = pathname.split('/').filter(Boolean);
  const topSegment = `/${segments[0] ?? ''}`.replace(/\/$/, '') || '/';
  const labelKey = PATH_LABELS[topSegment];
  const pageLabel = labelKey ? t(labelKey) : segments[0] ?? '';

  return (
    <nav
      className="hidden items-center gap-1.5 text-[11px] font-medium md:flex"
      aria-label="breadcrumb"
    >
      <span className="text-muted-foreground">{t('app.name')}</span>
      <ChevronRight className="h-3 w-3 text-muted-foreground" strokeWidth={2} />
      <span className="text-foreground">{pageLabel || t('nav.dashboard')}</span>
      {segments.length > 1 ? (
        <>
          <ChevronRight className="h-3 w-3 text-muted-foreground" strokeWidth={2} />
          <span className="font-mono text-muted-foreground">{segments.slice(1).join('/')}</span>
        </>
      ) : null}
    </nav>
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

  const isOnline = connectionStatus === 'connected';

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-card px-4">
      <div className="flex items-center gap-3">
        <Button variant="ghost" size="icon" onClick={toggleSidebar} aria-label="toggle sidebar">
          <Menu className="h-4 w-4" />
        </Button>
        <Breadcrumb />
      </div>

      <div className="flex items-center gap-2">
        <BtcPriceDisplay />
        <span className="hidden h-3.5 w-px bg-border md:block" />
        <ApiKeyBadge />
        <WSStatusIndicator status={connectionStatus} />
        <StatusPill tone={isOnline ? 'success' : 'default'} live={isOnline}>
          {isOnline ? t('ws.online', { defaultValue: '在线' }) : t('ws.offline', { defaultValue: '离线' })}
        </StatusPill>

        <span className="mx-1 h-5 w-px bg-border" />

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
