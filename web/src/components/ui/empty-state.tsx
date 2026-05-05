import { type ReactNode } from 'react';

import { cn } from '@/lib/cn';

export interface EmptyStateProps {
  /** Icon node — caller passes ``<Icon className="h-5 w-5" />`` already sized. */
  icon?: ReactNode;
  /** Required headline. */
  title: ReactNode;
  /** Optional supporting line. */
  description?: ReactNode;
  /** Optional CTA — typically a Button. */
  action?: ReactNode;
  className?: string;
  /**
   * - ``compact``: zero-chrome inline placeholder for use inside an existing
   *   card or table cell (replaces the old ``text-xs text-muted-foreground``
   *   single-line pattern).
   * - ``default``: standalone block with dashed border + padding.
   */
  size?: 'compact' | 'default';
}

/**
 * Unified empty state. Replaces the per-page ``<div className="text-xs
 * text-muted-foreground">暂无数据</div>`` pattern (which looked like a bug).
 */
export const EmptyState = ({
  icon,
  title,
  description,
  action,
  className,
  size = 'default',
}: EmptyStateProps) => {
  const compact = size === 'compact';
  return (
    <div
      role="status"
      className={cn(
        'flex flex-col items-center justify-center gap-2 text-center text-muted-foreground',
        compact
          ? 'py-4'
          : 'rounded-lg border border-dashed border-border bg-muted/20 p-8',
        className,
      )}
    >
      {icon ? (
        <div className={cn('text-muted-foreground/70', compact && 'opacity-80')}>
          {icon}
        </div>
      ) : null}
      <div className="space-y-1">
        <p
          className={cn(
            'font-medium',
            compact ? 'text-xs' : 'text-sm text-foreground',
          )}
        >
          {title}
        </p>
        {description ? <p className="text-xs">{description}</p> : null}
      </div>
      {action ? <div className="mt-1">{action}</div> : null}
    </div>
  );
};
