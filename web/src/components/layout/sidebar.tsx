import { Activity, BarChart3, Briefcase, Gauge, MessageSquare, ScrollText, ShieldCheck, TrendingUp, type LucideIcon } from 'lucide-react';
import { type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { NavLink } from 'react-router';

import { cn } from '@/lib/cn';
import { useUIStore } from '@/stores/use-ui-store';

interface NavItem {
  to: string;
  labelKey: 'nav.dashboard' | 'nav.decisions' | 'nav.backtest' | 'nav.risk' | 'nav.metrics' | 'nav.chat' | 'nav.market';
  icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  { to: '/', labelKey: 'nav.dashboard', icon: Briefcase },
  { to: '/decisions', labelKey: 'nav.decisions', icon: ScrollText },
  { to: '/backtest', labelKey: 'nav.backtest', icon: BarChart3 },
  { to: '/risk', labelKey: 'nav.risk', icon: ShieldCheck },
  { to: '/metrics', labelKey: 'nav.metrics', icon: Gauge },
  { to: '/chat', labelKey: 'nav.chat', icon: MessageSquare },
  { to: '/market', labelKey: 'nav.market', icon: TrendingUp },
];

export const Sidebar = () => {
  const { t } = useTranslation();
  const collapsed = useUIStore((s) => s.sidebarCollapsed);

  return (
    <aside
      className={cn(
        'flex h-screen flex-col border-r border-border bg-card transition-[width] duration-200',
        collapsed ? 'w-16' : 'w-60',
      )}
      aria-label="primary"
    >
      <div className="flex h-14 items-center gap-2 border-b border-border px-4">
        <Activity className="h-5 w-5 text-primary" aria-hidden="true" />
        {!collapsed ? (
          <div className="flex flex-col leading-tight">
            <span className="text-sm font-semibold text-foreground">{t('app.name')}</span>
            <span className="text-[10px] text-muted-foreground">{t('app.tagline')}</span>
          </div>
        ) : null}
      </div>
      <nav className="flex-1 space-y-0.5 px-2 py-3">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground',
              )
            }
          >
            <item.icon className="h-4 w-4 shrink-0" aria-hidden="true" />
            {!collapsed ? <span className="truncate">{t(item.labelKey)}</span> : null}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
};

export const SidebarFooterSlot = ({ children }: { children: ReactNode }) => (
  <div className="border-t border-border px-3 py-2">{children}</div>
);
