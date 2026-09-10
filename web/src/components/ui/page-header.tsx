import { ArrowLeft } from 'lucide-react';
import { type ReactNode } from 'react';

import { cn } from '@/lib/cn';

import { Button } from './button';

export interface PageHeaderProps {
  id?: string;
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
export const PageHeader = ({ id, title, subtitle, eyebrow, actions, onBack, className }: PageHeaderProps) => (
  <header id={id} className={cn('page-header flex flex-col items-start justify-between gap-4 sm:flex-row', className)}>
    <div className="flex min-w-0 flex-1 items-start gap-3">
      {onBack ? (
        <Button variant="ghost" size="icon" onClick={onBack} className="-ml-1 shrink-0" aria-label="返回">
          <ArrowLeft className="h-4 w-4" />
        </Button>
      ) : null}
      <div className="min-w-0 flex-1">
        {eyebrow ? <div className="mb-1 text-sm font-medium text-muted-foreground">{eyebrow}</div> : null}
        <h1 className="break-words text-2xl font-semibold tracking-tight text-foreground">{title}</h1>
        {subtitle ? (
          <div className="mt-2 max-w-prose text-sm leading-relaxed text-muted-foreground">{subtitle}</div>
        ) : null}
      </div>
    </div>
    {actions ? <div className="flex max-w-full flex-wrap items-center gap-2 sm:justify-end">{actions}</div> : null}
  </header>
);
