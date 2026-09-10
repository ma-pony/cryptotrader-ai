import { BarChart3, Briefcase, Cog, Landmark, ScrollText, SlidersHorizontal, type LucideIcon } from 'lucide-react';
import { type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { NavLink } from 'react-router';

import { useRuntimeStatus } from '@/hooks/use-runtime-status';
import { useSchedulerStatus } from '@/hooks/use-scheduler-status';
import { cn } from '@/lib/cn';
import { useUIStore } from '@/stores/use-ui-store';

type NavLabelKey = 'nav.workbench' | 'nav.decisions' | 'nav.engine' | 'nav.accounts' | 'nav.research' | 'nav.settings';

interface NavItem {
  to: string;
  labelKey: NavLabelKey;
  icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  { to: '/', labelKey: 'nav.workbench', icon: Briefcase },
  { to: '/decisions', labelKey: 'nav.decisions', icon: ScrollText },
  { to: '/engine', labelKey: 'nav.engine', icon: SlidersHorizontal },
  { to: '/accounts', labelKey: 'nav.accounts', icon: Landmark },
  { to: '/research', labelKey: 'nav.research', icon: BarChart3 },
  { to: '/settings', labelKey: 'nav.settings', icon: Cog },
];

const SidebarFooter = () => {
  const runtime = useRuntimeStatus();
  const scheduler = useSchedulerStatus();
  const automation = runtime.data?.automation_enabled;
  const runtimeLabel =
    runtime.isPending || (!runtime.isError && !runtime.data)
      ? '正在读取运行状态'
      : runtime.isError
        ? '运行状态未知'
        : automation
          ? '自动运行已开启'
          : '自动运行已暂停';
  const schedulerLabel =
    scheduler.isPending || (!scheduler.isError && !scheduler.data)
      ? '正在读取定时来源状态'
      : scheduler.isError
        ? '定时来源未知'
        : scheduler.data.enabled
          ? '定时来源已启用'
          : '定时来源未启用';

  return (
    <div className="flex flex-col gap-2 border-t border-border p-3">
      <div className="flex items-center gap-2 rounded-lg border border-border bg-muted/40 px-2.5 py-2">
        <span className={cn('h-2 w-2 shrink-0 rounded-full', automation ? 'bg-amber-500' : 'bg-muted-foreground')} />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-foreground">{runtimeLabel}</div>
          <div className="text-sm text-muted-foreground">{schedulerLabel}</div>
        </div>
      </div>
    </div>
  );
};

/** Brand mark + name. Shared by the desktop sidebar and the mobile drawer. */
const SidebarBrand = ({ collapsed }: { collapsed: boolean }) => {
  const { t } = useTranslation();
  return (
    <div className="flex h-14 items-center gap-2.5 border-b border-border px-4">
      <span
        className="flex h-8 w-8 items-center justify-center rounded-lg font-semibold text-[15px] shadow-glow-amber"
        style={{
          background: 'var(--amber-500)',
          color: 'hsl(var(--primary-foreground))',
        }}
        aria-hidden
      >
        ₵
      </span>
      {!collapsed ? (
        <div className="flex flex-col leading-tight">
          <span className="text-sm font-semibold text-foreground">{t('app.name')}</span>
          <span className="text-sm text-muted-foreground">{t('app.version', { version: '2.4' })}</span>
        </div>
      ) : null}
    </div>
  );
};

interface SidebarNavProps {
  collapsed: boolean;
  /** Called after a nav link is activated (used to close the mobile drawer). */
  onNavigate?: (() => void) | undefined;
}

const SidebarNav = ({ collapsed, onNavigate }: SidebarNavProps) => {
  const { t } = useTranslation();
  return (
    <nav className="flex-1 overflow-y-auto px-2 py-3">
      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.to === '/'}
          onClick={onNavigate}
          aria-label={t(item.labelKey)}
          title={collapsed ? t(item.labelKey) : undefined}
          className={({ isActive }) =>
            cn(
              'flex min-h-10 items-center gap-3 whitespace-nowrap rounded-md border-l-2 px-3 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 active:bg-muted max-md:min-h-11',
              isActive
                ? 'border-l-amber-500 bg-muted pl-[10px] font-medium text-foreground'
                : 'border-l-transparent text-muted-foreground hover:bg-muted/50 hover:text-foreground',
            )
          }
        >
          <item.icon className="h-4 w-4 shrink-0" aria-hidden="true" />
          {!collapsed ? <span className="truncate">{t(item.labelKey)}</span> : null}
        </NavLink>
      ))}
    </nav>
  );
};

/** Drawer body for the mobile sidebar (rendered inside <Sheet>). */
export const SidebarDrawerBody = ({ onNavigate }: { onNavigate?: () => void }) => (
  <>
    <SidebarBrand collapsed={false} />
    <SidebarNav collapsed={false} onNavigate={onNavigate} />
    <SidebarFooter />
  </>
);

export const Sidebar = () => {
  const { t } = useTranslation();
  const collapsed = useUIStore((s) => s.sidebarCollapsed);

  return (
    // Hidden below md — the TopBar menu button opens the SidebarDrawer
    // (rendered by AppShell) instead. Above md the sidebar is always
    // present and its width depends on the collapsed flag.
    <aside
      className={cn(
        'sticky top-0 hidden h-dvh shrink-0 flex-col border-r border-border bg-card md:flex',
        collapsed ? 'w-16' : 'w-60',
      )}
      aria-label={t('header.primaryNavigation')}
    >
      <SidebarBrand collapsed={collapsed} />
      <SidebarNav collapsed={collapsed} />
      {!collapsed ? <SidebarFooter /> : null}
    </aside>
  );
};

export const SidebarFooterSlot = ({ children }: { children: ReactNode }) => (
  <div className="border-t border-border px-3 py-2">{children}</div>
);
