import { AlertTriangle, RefreshCw } from 'lucide-react';
import { type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { cn } from '@/lib/cn';

import { Button } from './button';

export interface ErrorStateProps {
  title?: ReactNode;
  description?: ReactNode;
  onRetry?: () => void;
  className?: string;
  icon?: ReactNode;
}

export const ErrorState = ({ title, description, onRetry, className, icon }: ErrorStateProps) => {
  const { t } = useTranslation();
  return (
    <div
      role="alert"
      className={cn('flex flex-col items-center justify-center gap-3 rounded-md border border-dashed border-border p-6 text-center text-muted-foreground', className)}
    >
      <div className="text-warning" aria-hidden="true">
        {icon ?? <AlertTriangle className="h-6 w-6" />}
      </div>
      <div className="space-y-1">
        <p className="text-sm font-medium text-foreground">{title ?? t('errors.generic')}</p>
        {description ? <p className="text-xs">{description}</p> : null}
      </div>
      {onRetry ? (
        <Button variant="outline" size="sm" onClick={onRetry} aria-label={t('actions.retry')}>
          <RefreshCw className="mr-2 h-3.5 w-3.5" />
          {t('actions.retry')}
        </Button>
      ) : null}
    </div>
  );
};
