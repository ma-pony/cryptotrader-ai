import { ArrowLeft } from 'lucide-react';
import { type ReactNode } from 'react';

import { cn } from '@/lib/cn';

import { Button } from './button';

export interface PageHeaderProps {
  /** Required main title. Renders as `<h1 text-2xl font-semibold tracking-tight>`. */
  title: ReactNode;
  /** Secondary line below the title (e.g. ``BTC/USDT @ $103,000``). */
  subtitle?: ReactNode;
  /** Small uppercase label above the title (e.g. ``辩论可视化 · 决策 abcd1234``). */
  eyebrow?: ReactNode;
  /** Right-aligned actions (selectors, buttons). */
  actions?: ReactNode;
  /** When provided, renders a back button to the left of the title block. */
  onBack?: () => void;
  className?: string;
}

/**
 * Single source of truth for page-level headers. All pages should use this
 * instead of writing ``<h1 className="text-2xl ...">`` inline so title size,
 * eyebrow style, and action layout stay synchronised.
 */
export const PageHeader = ({
  title,
  subtitle,
  eyebrow,
  actions,
  onBack,
  className,
}: PageHeaderProps) => (
  <div className={cn('flex items-start justify-between gap-4', className)}>
    <div className="flex min-w-0 flex-1 items-start gap-3">
      {onBack ? (
        <Button
          variant="ghost"
          size="icon"
          onClick={onBack}
          className="-ml-1 mt-0.5 h-8 w-8 shrink-0"
          aria-label="back"
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>
      ) : null}
      <div className="min-w-0 flex-1">
        {eyebrow ? (
          <div className="mb-1 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            {eyebrow}
          </div>
        ) : null}
        <h1 className="truncate text-2xl font-semibold tracking-tight text-foreground">
          {title}
        </h1>
        {subtitle ? (
          <div className="mt-1 text-sm text-muted-foreground">{subtitle}</div>
        ) : null}
      </div>
    </div>
    {actions ? (
      <div className="flex shrink-0 items-center gap-2">{actions}</div>
    ) : null}
  </div>
);
