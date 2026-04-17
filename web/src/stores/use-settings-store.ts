import { create } from 'zustand';

// IMPORTANT: settings such as apiKey are kept ONLY in memory (NFR-S-005).
// They are never persisted to localStorage to avoid leaking via XSS or extensions.
interface SettingsState {
  apiKey: string;
  otlpEndpoint: string;
  setApiKey: (key: string) => void;
  setOtlpEndpoint: (endpoint: string) => void;
  reset: () => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  apiKey: '',
  otlpEndpoint: '',
  setApiKey: (apiKey) => set({ apiKey }),
  setOtlpEndpoint: (otlpEndpoint) => set({ otlpEndpoint }),
  reset: () => set({ apiKey: '', otlpEndpoint: '' }),
}));
