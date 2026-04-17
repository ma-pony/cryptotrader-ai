import { Skeleton } from '@/components/ui/skeleton';

export const RouteSkeleton = () => (
  <div className="flex flex-col gap-4">
    <Skeleton className="h-8 w-48" />
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Skeleton className="h-24" />
      <Skeleton className="h-24" />
      <Skeleton className="h-24" />
      <Skeleton className="h-24" />
    </div>
    <Skeleton className="h-72" />
  </div>
);
