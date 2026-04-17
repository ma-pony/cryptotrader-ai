import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Component, type ErrorInfo, type ReactNode } from 'react';
import { withTranslation, type WithTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { ApiError } from '@/lib/api-client';

interface Props extends WithTranslation {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  error: Error | null;
  traceId: string | null;
}

class ErrorBoundaryInner extends Component<Props, State> {
  override state: State = { error: null, traceId: null };

  static getDerivedStateFromError(error: Error): State {
    const traceId = error instanceof ApiError && error.traceId ? error.traceId : null;
    return { error, traceId };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  private reset = () => {
    this.setState({ error: null, traceId: null });
  };

  override render() {
    const { error, traceId } = this.state;
    const { t, children, fallback } = this.props;
    if (!error) return children;
    if (fallback) return fallback;
    return (
      <div role="alert" className="flex min-h-[60vh] flex-col items-center justify-center gap-4 p-6 text-center">
        <AlertTriangle className="h-10 w-10 text-warning" aria-hidden="true" />
        <div className="space-y-1">
          <p className="text-lg font-semibold text-foreground">{t('errors.generic')}</p>
          <p className="max-w-md text-sm text-muted-foreground">{error.message}</p>
          {traceId ? (
            <p className="text-xs text-muted-foreground">
              {t('errors.trace_id')}: <code className="font-mono">{traceId}</code>
            </p>
          ) : null}
        </div>
        <Button variant="outline" size="sm" onClick={this.reset}>
          <RefreshCw className="mr-2 h-3.5 w-3.5" />
          {t('actions.retry')}
        </Button>
      </div>
    );
  }
}

export const ErrorBoundary = withTranslation()(ErrorBoundaryInner);
