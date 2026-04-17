import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export type Theme = 'dark' | 'light' | 'system';
export type Locale = 'zh-CN' | 'en-US';

interface UIState {
  theme: Theme;
  locale: Locale;
  sidebarCollapsed: boolean;
  setTheme: (theme: Theme) => void;
  setLocale: (locale: Locale) => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (v: boolean) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      theme: 'system',
      locale: 'zh-CN',
      sidebarCollapsed: false,
      setTheme: (theme) => set({ theme }),
      setLocale: (locale) => set({ locale }),
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setSidebarCollapsed: (v) => set({ sidebarCollapsed: v }),
    }),
    {
      name: 'cryptotrader-ui',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ theme: s.theme, locale: s.locale, sidebarCollapsed: s.sidebarCollapsed }),
    },
  ),
);
