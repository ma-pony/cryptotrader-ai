import { Outlet } from 'react-router';

import { Sheet, SheetContent } from '@/components/ui/sheet';
import { useUIStore } from '@/stores/use-ui-store';

import { Sidebar, SidebarDrawerBody } from './sidebar';
import { TopBar } from './top-bar';

/**
 * Mobile (<md) shows the sidebar inside a Sheet drawer triggered from
 * TopBar's menu button. Desktop (md+) keeps the sidebar permanently
 * mounted alongside the main content.
 *
 * DesktopOnlyBanner has been retired — the layout now actually adapts
 * to <1024px viewports instead of asking the user to switch to desktop.
 */
export const AppShell = () => {
  const mobileNavOpen = useUIStore((s) => s.mobileNavOpen);
  const setMobileNavOpen = useUIStore((s) => s.setMobileNavOpen);

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <a href="#page-content" className="skip-link">
        跳到主要内容
      </a>
      <Sidebar />

      {/* Mobile drawer — triggered by TopBar's menu button */}
      <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
        <SheetContent side="left" hideClose className="md:hidden">
          <SidebarDrawerBody onNavigate={() => setMobileNavOpen(false)} />
        </SheetContent>
      </Sheet>

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        {/* Tighter padding on small screens so content gets more room */}
        <main id="page-content" tabIndex={-1} className="app-content flex-1 px-4 py-5 md:px-6 md:py-8">
          <div className="page-container">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};
