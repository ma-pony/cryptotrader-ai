import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router';
import { ErrorBoundary } from '@/components/error-boundary';
import { AppShell } from '@/components/layout/app-shell';
import { RouteSkeleton } from '@/components/route-skeleton';
import { useRuntimeConfig } from '@/hooks/use-runtime-config';
const DashboardPage = lazy(() => import('@/pages/dashboard'));
const DecisionsPage = lazy(() => import('@/pages/decisions'));
const DebatePage = lazy(() => import('@/pages/debate'));
const BacktestPage = lazy(() => import('@/pages/backtest'));
const RiskPage = lazy(() => import('@/pages/risk'));
const MetricsPage = lazy(() => import('@/pages/metrics'));
const ChatPage = lazy(() => import('@/pages/chat'));
const MarketPage = lazy(() => import('@/pages/market'));
const SchedulerPage = lazy(() => import('@/pages/scheduler'));
const StrategyPage = lazy(() => import('@/pages/strategy'));
const MemoryPage = lazy(() => import('@/pages/memory/MemoryPage'));
const VenuesPage = lazy(() => import('@/pages/settings/venues'));
const BooksPage = lazy(() => import('@/pages/settings/execution-books'));
const SetupPage = lazy(() => import('@/pages/setup'));
const NotFoundPage = lazy(() => import('@/pages/not-found'));
const GuardedRoutes = () => {
  const runtime = useRuntimeConfig();
  const location = useLocation();
  if (runtime.isLoading) return <RouteSkeleton />;
  if (runtime.isError)
    return (
      <main className="grid min-h-screen place-items-center">
        <button onClick={() => void runtime.reload()}>配置加载失败，重试</button>
      </main>
    );
  if (runtime.setupRequired)
    return (
      <Routes>
        <Route path="*" element={<SetupPage />} />
      </Routes>
    );
  if (location.pathname === '/setup') return <Navigate to="/" replace />;
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<DashboardPage />} />
        <Route path="strategy" element={<StrategyPage />} />
        <Route path="settings/venues" element={<VenuesPage />} />
        <Route path="settings/execution-books" element={<BooksPage />} />
        <Route path="decisions" element={<DecisionsPage />} />
        <Route path="decisions/:cycleId" element={<DecisionsPage />} />
        <Route path="debate" element={<DebatePage />} />
        <Route path="debate/:cycleId" element={<DebatePage />} />
        <Route path="backtest" element={<BacktestPage />} />
        <Route path="risk" element={<RiskPage />} />
        <Route path="metrics" element={<MetricsPage />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="chat/:sessionId" element={<ChatPage />} />
        <Route path="market" element={<MarketPage />} />
        <Route path="scheduler" element={<SchedulerPage />} />
        <Route path="memory" element={<MemoryPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
};
export const App = () => (
  <ErrorBoundary>
    <Suspense fallback={<RouteSkeleton />}>
      <GuardedRoutes />
    </Suspense>
  </ErrorBoundary>
);
