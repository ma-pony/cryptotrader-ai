import { create } from 'zustand';

import { env, isDev } from '@/lib/env';

// IMPORTANT: settings such as apiKey are kept ONLY in memory (NFR-S-005).
// They are never persisted to localStorage to avoid leaking via XSS or extensions.
//
// SEC-I3: VITE_API_KEY is consumed ONLY as a dev-mode boot hydration. The Vite
// plugin `forbid-baked-api-key` rejects production builds that set it, so prod
// bundles never carry a key. Runtime callers (api-client, stream-fetch, hooks)
// read exclusively from this store — they MUST NOT fall back to env.VITE_API_KEY.
const initialApiKey = isDev ? env.VITE_API_KEY : '';

interface SettingsState {
  apiKey: string;
  otlpEndpoint: string;
  setApiKey: (key: string) => void;
  setOtlpEndpoint: (endpoint: string) => void;
  reset: () => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  apiKey: initialApiKey,
  otlpEndpoint: env.VITE_OTLP_UI_ENDPOINT,
  setApiKey: (apiKey) => set({ apiKey }),
  setOtlpEndpoint: (otlpEndpoint) => set({ otlpEndpoint }),
  reset: () => set({ apiKey: '', otlpEndpoint: '' }),
}));
