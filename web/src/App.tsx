import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router';
import { ErrorBoundary } from '@/components/error-boundary';
import { AppShell } from '@/components/layout/app-shell';
import { RouteSkeleton } from '@/components/route-skeleton';
import { useRuntimeConfig } from '@/hooks/use-runtime-config';
import { ConfigurationProvider } from '@/pages/settings/configuration-context';
import SettingsPage, { ConfigurationLayout } from '@/pages/settings';
import { ConfigurationAccess } from '@/pages/settings/configuration-access';
const DashboardPage = lazy(() => import('@/pages/dashboard'));
const DecisionsPage = lazy(() => import('@/pages/decisions'));
const CyclesPage = lazy(() => import('@/pages/cycles'));
const CycleDetailPage = lazy(() => import('@/pages/cycles/cycle-detail'));
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
  if (!runtime.document) return <ConfigurationAccess runtime={runtime} />;
  return (
    <ConfigurationProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={runtime.setupRequired ? <Navigate to="/setup" replace /> : <DashboardPage />} />
          <Route element={<ConfigurationLayout />}>
            <Route path="setup" element={<SetupPage />} />
            <Route path="settings" element={<Navigate to="/settings/models" replace />} />
            <Route path="settings/models" element={<SettingsPage section="models" />} />
            <Route path="settings/market" element={<SettingsPage section="market" />} />
            <Route path="settings/risk" element={<SettingsPage section="risk" />} />
            <Route path="settings/scheduler" element={<SettingsPage section="scheduler" />} />
            <Route path="settings/system" element={<SettingsPage section="system" />} />
            <Route path="strategy" element={<StrategyPage />} />
            <Route path="settings/venues" element={<VenuesPage />} />
            <Route path="settings/execution-books" element={<BooksPage />} />
          </Route>
          <Route path="decisions" element={<DecisionsPage />} />
          <Route path="decisions/:cycleId" element={<DecisionsPage />} />
          <Route path="cycles" element={<CyclesPage />} />
          <Route path="cycles/:cycleId" element={<CycleDetailPage />} />
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
    </ConfigurationProvider>
  );
};
export const App = () => (
  <ErrorBoundary>
    <Suspense fallback={<RouteSkeleton />}>
      <GuardedRoutes />
    </Suspense>
  </ErrorBoundary>
);
