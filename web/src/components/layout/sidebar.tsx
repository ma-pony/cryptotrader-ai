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

interface NavItem {
  to: string;
  labelKey:
    | 'nav.dashboard'
    | 'nav.decisions'
    | 'nav.debate'
    | 'nav.backtest'
    | 'nav.risk'
    | 'nav.metrics'
    | 'nav.chat'
    | 'nav.market'
    | 'nav.scheduler';
  icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  { to: '/', labelKey: 'nav.dashboard', icon: Briefcase },
  { to: '/decisions', labelKey: 'nav.decisions', icon: ScrollText },
  { to: '/debate', labelKey: 'nav.debate', icon: GitBranch },
  { to: '/backtest', labelKey: 'nav.backtest', icon: BarChart3 },
  { to: '/risk', labelKey: 'nav.risk', icon: ShieldCheck },
  { to: '/metrics', labelKey: 'nav.metrics', icon: Gauge },
  { to: '/chat', labelKey: 'nav.chat', icon: MessageSquare },
  { to: '/market', labelKey: 'nav.market', icon: TrendingUp },
  { to: '/scheduler', labelKey: 'nav.scheduler', icon: CalendarClock },
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
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
              AI · v2.4
            </span>
          </div>
        ) : null}
      </div>
      <nav className="flex-1 space-y-0.5 overflow-y-auto px-2 py-3">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
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
      </nav>
      {!collapsed ? <SidebarFooter /> : null}
    </aside>
  );
};

export const SidebarFooterSlot = ({ children }: { children: ReactNode }) => (
  <div className="border-t border-border px-3 py-2">{children}</div>
);
