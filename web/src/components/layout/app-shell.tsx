import { Outlet } from 'react-router';

import { DesktopOnlyBanner } from './desktop-only-banner';
import { Sidebar } from './sidebar';
import { TopBar } from './top-bar';

export const AppShell = () => (
  <div className="flex min-h-screen bg-background text-foreground">
    <Sidebar />
    <div className="flex min-w-0 flex-1 flex-col">
      <DesktopOnlyBanner />
      <TopBar />
      <main className="flex-1 overflow-auto px-6 py-6">
        <Outlet />
      </main>
    </div>
  </div>
);
