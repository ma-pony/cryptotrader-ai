import { Suspense, type ReactNode } from 'react';

import { ErrorBoundary } from '@/components/error-boundary';
import { ErrorState } from '@/components/ui/error-state';
import { Skeleton } from '@/components/ui/skeleton';

export interface PageBoundaryProps {
  children: ReactNode;
  /** Show skeleton when true. Mutually exclusive with ``isError``. */
  loading?: boolean;
  /** Show ErrorState when true. */
  isError?: boolean;
  /** Retry handler wired into ErrorState button. */
  onRetry?: () => void;
  /** Override the default page-level skeleton. */
  loadingFallback?: ReactNode;
  /** Custom error title. Defaults to the i18n ``errors.generic`` string. */
  errorTitle?: ReactNode;
  /** Custom error description. */
  errorDescription?: ReactNode;
}

/**
 * Page-level container that consolidates three concerns previously copy-pasted
 * into every page:
 *
 *   1. Runtime ErrorBoundary (catches uncaught exceptions in children).
 *   2. Suspense fallback (lets lazy children stream).
 *   3. Known data states (``loading`` / ``isError`` from React Query).
 *
 * Replaces the pattern::
 *
 *     <ErrorBoundary>
 *       <Suspense fallback={<Skeleton className="h-96 w-full" />}>
 *         {isLoading ? <Skeleton .../>
 *           : isError ? <ErrorState .../>
 *           : <Content />}
 *       </Suspense>
 *     </ErrorBoundary>
 *
 * with::
 *
 *     <PageBoundary loading={isLoading} isError={isError} onRetry={refetch}>
 *       <Content />
 *     </PageBoundary>
 *
 * Skeleton heights, error styling, and breakpoint behaviour are now defined
 * here and stay consistent across pages.
 */
export const PageBoundary = ({
  children,
  loading = false,
  isError = false,
  onRetry,
  loadingFallback,
  errorTitle,
  errorDescription,
}: PageBoundaryProps) => {
  const fallback = loadingFallback ?? <DefaultPageSkeleton />;

  let body: ReactNode;
  if (loading) body = fallback;
  else if (isError)
    body = (
      <ErrorState
        {...(errorTitle !== undefined ? { title: errorTitle } : {})}
        {...(errorDescription !== undefined ? { description: errorDescription } : {})}
        {...(onRetry ? { onRetry } : {})}
      />
    );
  else body = children;

  return (
    <ErrorBoundary>
      <Suspense fallback={fallback}>{body}</Suspense>
    </ErrorBoundary>
  );
};

/**
 * Default skeleton mimicking the typical page shell: a small title row, a
 * dense metrics row, and a tall main content block. More page-shaped than
 * the bare ``<Skeleton className="h-96 w-full" />`` previously copy-pasted
 * everywhere.
 */
const DefaultPageSkeleton = () => (
  <div className="space-y-6" aria-hidden>
    <Skeleton className="h-8 w-56" />
    <Skeleton className="h-24 w-full" />
    <Skeleton className="h-72 w-full" />
  </div>
);
