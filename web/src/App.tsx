import { lazy, Suspense } from 'react';
import { Route, Routes } from 'react-router';

import { ErrorBoundary } from '@/components/error-boundary';
import { AppShell } from '@/components/layout/app-shell';
import { RouteSkeleton } from '@/components/route-skeleton';

const DashboardPage = lazy(() => import('@/pages/dashboard'));
const DecisionsPage = lazy(() => import('@/pages/decisions'));
const DebatePage = lazy(() => import('@/pages/debate'));
const BacktestPage = lazy(() => import('@/pages/backtest'));
const RiskPage = lazy(() => import('@/pages/risk'));
const MetricsPage = lazy(() => import('@/pages/metrics'));
const ChatPage = lazy(() => import('@/pages/chat'));
const MarketPage = lazy(() => import('@/pages/market'));
const SchedulerPage = lazy(() => import('@/pages/scheduler'));
const MemoryPage = lazy(() => import('@/pages/memory/MemoryPage'));
const RulesDetailPage = lazy(() => import('@/pages/memory/RulesDetailPage'));
const PatternDetailPage = lazy(() => import('@/pages/memory/PatternDetailPage'));
const NotFoundPage = lazy(() => import('@/pages/not-found'));

export const App = () => (
  <ErrorBoundary>
    <Suspense fallback={<RouteSkeleton />}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<DashboardPage />} />
          <Route path="decisions" element={<DecisionsPage />} />
          <Route path="decisions/:commitId" element={<DecisionsPage />} />
          <Route path="debate" element={<DebatePage />} />
          <Route path="debate/:commitId" element={<DebatePage />} />
          <Route path="backtest" element={<BacktestPage />} />
          <Route path="risk" element={<RiskPage />} />
          <Route path="metrics" element={<MetricsPage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="chat/:sessionId" element={<ChatPage />} />
          <Route path="market" element={<MarketPage />} />
          <Route path="scheduler" element={<SchedulerPage />} />
          <Route path="memory" element={<MemoryPage />} />
          <Route path="memory/rules" element={<RulesDetailPage />} />
          <Route path="memory/patterns/:agent/:name" element={<PatternDetailPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </Suspense>
  </ErrorBoundary>
);
