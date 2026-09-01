import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router';
import { ErrorBoundary } from '@/components/error-boundary';
import { AppShell } from '@/components/layout/app-shell';
import { RouteSkeleton } from '@/components/route-skeleton';
import { ConfigurationEditorGate, ConfigurationScope } from '@/pages/settings/configuration-context';
import SettingsPage, { ConfigurationLayout } from '@/pages/settings';
const WorkbenchPage = lazy(() => import('@/pages/workbench'));
const DecisionsPage = lazy(() => import('@/pages/decisions'));
const DecisionDetailPage = lazy(() => import('@/pages/decisions/detail'));
const ResearchPage = lazy(() => import('@/pages/research'));
const BacktestDetail = lazy(() => import('@/pages/research/backtest-detail'));
const BacktestCompare = lazy(() => import('@/pages/research/backtest-compare'));
const ResearchAnalysis = lazy(() => import('@/pages/research/analysis'));
const MetricsPage = lazy(() => import('@/pages/settings/metrics'));
const MarketPage = lazy(() => import('@/pages/research/market'));
const EnginePage = lazy(() => import('@/pages/engine'));
const ComponentDetailPage = lazy(() => import('@/pages/engine/component-detail'));
const MemoryPage = lazy(() => import('@/pages/settings/agent-profiles'));
const ConnectionsPage = lazy(() => import('@/pages/accounts/connections-page'));
const AccountsPage = lazy(() => import('@/pages/accounts'));
const ConnectionDetail = lazy(() => import('@/pages/accounts/connection-detail'));
const BookDetail = lazy(() => import('@/pages/accounts/book-detail'));
const BooksPage = lazy(() => import('@/pages/accounts/books-page'));
const NotificationSettings = lazy(() => import('@/pages/settings/notifications'));
const NotFoundPage = lazy(() => import('@/pages/not-found'));
const ApplicationRoutes = () => (
  <Routes>
    <Route element={<AppShell />}>
      <Route index element={<WorkbenchPage />} />
      <Route path="decisions" element={<DecisionsPage />} />
      <Route path="decisions/:decisionId" element={<DecisionDetailPage />} />
      <Route path="research" element={<ResearchPage />} />
      <Route path="research/market" element={<MarketPage />} />
      <Route path="research/analysis" element={<ResearchAnalysis />} />
      <Route path="research/backtests/:runId" element={<BacktestDetail />} />
      <Route path="research/compare" element={<BacktestCompare />} />
      <Route element={<ConfigurationScope />}>
        <Route path="engine" element={<EnginePage />} />
        <Route path="engine/components/:componentId" element={<ComponentDetailPage />} />
        <Route path="accounts" element={<AccountsPage />} />
        <Route element={<ConfigurationEditorGate />}>
          <Route path="accounts/connections" element={<ConnectionsPage />} />
          <Route path="accounts/books" element={<BooksPage />} />
        </Route>
        <Route path="accounts/connections/:connectionId" element={<ConnectionDetail />} />
        <Route path="accounts/books/:bookId" element={<BookDetail />} />
        <Route element={<ConfigurationLayout />}>
          <Route path="settings" element={<Navigate to="/settings/models" replace />} />
          <Route element={<ConfigurationEditorGate />}>
            <Route path="settings/models" element={<SettingsPage section="models" />} />
            <Route path="settings/notifications" element={<NotificationSettings />} />
            <Route path="settings/security" element={<SettingsPage section="system" />} />
          </Route>
          <Route path="settings/metrics" element={<MetricsPage />} />
          <Route path="settings/agent-profiles" element={<MemoryPage />} />
        </Route>
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Route>
  </Routes>
);
export const App = () => (
  <ErrorBoundary>
    <Suspense fallback={<RouteSkeleton />}>
      <ApplicationRoutes />
    </Suspense>
  </ErrorBoundary>
);
