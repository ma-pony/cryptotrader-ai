import { QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router';

import { App } from './App';
import { ErrorBoundary } from './components/error-boundary';
import { I18nProvider } from './components/providers/i18n-provider';
import { ThemeProvider } from './components/providers/theme-provider';
import { Toaster } from './components/ui/toaster';
import './lib/i18n';
import { queryClient } from './lib/query-client';
import { initWebVitals } from './lib/web-vitals';
import './styles/globals.css';

initWebVitals();

const rootEl = document.getElementById('root');
if (!rootEl) throw new Error('Root element #root not found');

ReactDOM.createRoot(rootEl).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ThemeProvider>
          <I18nProvider>
            <ErrorBoundary>
              <App />
            </ErrorBoundary>
            <Toaster />
          </I18nProvider>
        </ThemeProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
