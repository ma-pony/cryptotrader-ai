import {
  BarChart3,
  Briefcase,
  CalendarClock,
  Gauge,
  GitBranch,
  MessageSquare,
  ScrollText,
  ShieldCheck,
  TrendingUp,
  type LucideIcon,
} from 'lucide-react';
import { type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { NavLink } from 'react-router';

import { useCountdown } from '@/hooks/use-countdown';
import { useSchedulerStatus } from '@/hooks/use-scheduler-status';
import { cn } from '@/lib/cn';
import { useUIStore } from '@/stores/use-ui-store';

type NavLabelKey =
  | 'nav.dashboard'
  | 'nav.decisions'
  | 'nav.debate'
  | 'nav.backtest'
  | 'nav.risk'
  | 'nav.metrics'
  | 'nav.chat'
  | 'nav.market'
  | 'nav.scheduler';

interface NavItem {
  to: string;
  labelKey: NavLabelKey;
  icon: LucideIcon;
}

interface NavSection {
  /** Sidebar group label (i18n key with sensible defaultValue). */
  titleKey: 'nav.section.trading' | 'nav.section.analysis' | 'nav.section.operations';
  defaultTitle: string;
  items: NavItem[];
}

// Grouping rationale (deep review FE-2026-05-06):
//   trading      = the live decision loop (overview / decisions / debate)
//   analysis     = retrospective + market context (backtest / market / metrics)
//   operations   = control plane (risk / chat / scheduler)
const NAV_SECTIONS: NavSection[] = [
  {
    titleKey: 'nav.section.trading',
    defaultTitle: '交易',
    items: [
      { to: '/', labelKey: 'nav.dashboard', icon: Briefcase },
      { to: '/decisions', labelKey: 'nav.decisions', icon: ScrollText },
      { to: '/debate', labelKey: 'nav.debate', icon: GitBranch },
    ],
  },
  {
    titleKey: 'nav.section.analysis',
    defaultTitle: '分析',
    items: [
      { to: '/market', labelKey: 'nav.market', icon: TrendingUp },
      { to: '/backtest', labelKey: 'nav.backtest', icon: BarChart3 },
      { to: '/metrics', labelKey: 'nav.metrics', icon: Gauge },
    ],
  },
  {
    titleKey: 'nav.section.operations',
    defaultTitle: '运维',
    items: [
      { to: '/risk', labelKey: 'nav.risk', icon: ShieldCheck },
      { to: '/scheduler', labelKey: 'nav.scheduler', icon: CalendarClock },
      { to: '/chat', labelKey: 'nav.chat', icon: MessageSquare },
    ],
  },
];

const SidebarFooter = () => {
  const { t } = useTranslation();
  const { data } = useSchedulerStatus();
  // Footer precision: 5 seconds is plenty for sidebar display and cuts re-renders
  // from 60/min to 12/min versus the old per-second tick.
  const { formatted: countdown } = useCountdown(data?.next_run_at, 5_000);

  const running = data?.enabled ?? false;

  return (
    <div className="flex flex-col gap-2 border-t border-border p-3">
      <div
        className="flex items-center gap-2 rounded-lg border border-border bg-muted/40 px-2.5 py-2"
      >
        <span
          className={cn(
            'h-2 w-2 rounded-full shrink-0',
            running ? 'bg-trade-long animate-ct-pulse' : 'bg-muted-foreground',
          )}
        />
        <div className="flex-1 min-w-0">
          <div className="text-[11px] font-medium text-foreground">
            {running
              ? t('scheduler.running', { defaultValue: '模拟交易 · 运行中' })
              : t('scheduler.paused', { defaultValue: '调度器 · 已暂停' })}
          </div>
          <div className="font-mono text-[10px] text-muted-foreground">
            {running ? `${t('scheduler.next_run', { defaultValue: '下次分析' })} ${countdown}` : '—'}
          </div>
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
          background: 'linear-gradient(135deg, var(--amber-500), var(--amber-600))',
          color: 'hsl(var(--primary-foreground))',
          fontFamily: "'Space Grotesk', system-ui, sans-serif",
        }}
        aria-hidden
      >
        ₵
      </span>
      {!collapsed ? (
        <div className="flex flex-col leading-tight">
          <span className="text-sm font-semibold text-foreground">{t('app.name')}</span>
          <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            AI · v2.4
          </span>
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
      {NAV_SECTIONS.map((section, sectionIdx) => (
        <div
          key={section.titleKey}
          className={cn('space-y-0.5', sectionIdx > 0 && 'mt-4')}
        >
          {!collapsed ? (
            <div className="px-3 pb-1 pt-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70">
              {t(section.titleKey, { defaultValue: section.defaultTitle })}
            </div>
          ) : sectionIdx > 0 ? (
            // collapsed mode: thin divider between groups
            <div className="mx-2 mb-1 h-px bg-border" />
          ) : null}
          {section.items.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              onClick={onNavigate}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors border-l-2',
                  isActive
                    ? 'bg-muted text-foreground font-medium border-l-amber-500 pl-[10px]'
                    : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground border-l-transparent',
                )
              }
            >
              <item.icon className="h-4 w-4 shrink-0" aria-hidden="true" />
              {!collapsed ? <span className="truncate">{t(item.labelKey)}</span> : null}
            </NavLink>
          ))}
        </div>
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
  const collapsed = useUIStore((s) => s.sidebarCollapsed);

  return (
    // Hidden below md — the TopBar menu button opens the SidebarDrawer
    // (rendered by AppShell) instead. Above md the sidebar is always
    // present and its width depends on the collapsed flag.
    <aside
      className={cn(
        'hidden h-screen flex-col border-r border-border bg-card transition-[width] duration-200 md:flex',
        collapsed ? 'w-16' : 'w-60',
      )}
      aria-label="primary"
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
