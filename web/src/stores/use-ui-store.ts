import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export type Theme = 'dark' | 'light' | 'system';
export type Locale = 'zh-CN' | 'en-US';

interface UIState {
  theme: Theme;
  locale: Locale;
  sidebarCollapsed: boolean;
  /** Mobile-only drawer open state. Not persisted — always starts closed. */
  mobileNavOpen: boolean;
  setTheme: (theme: Theme) => void;
  setLocale: (locale: Locale) => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (v: boolean) => void;
  setMobileNavOpen: (v: boolean) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      theme: 'system',
      locale: 'zh-CN',
      sidebarCollapsed: false,
      mobileNavOpen: false,
      setTheme: (theme) => set({ theme }),
      setLocale: (locale) => set({ locale }),
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setSidebarCollapsed: (v) => set({ sidebarCollapsed: v }),
      setMobileNavOpen: (v) => set({ mobileNavOpen: v }),
    }),
    {
      name: 'cryptotrader-ui',
      storage: createJSONStorage(() => localStorage),
      // mobileNavOpen intentionally excluded — drawer state should not persist
      // across page reloads.
      partialize: (s) => ({ theme: s.theme, locale: s.locale, sidebarCollapsed: s.sidebarCollapsed }),
    },
  ),
);
