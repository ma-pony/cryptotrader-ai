import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// SEC-I3: Reject `VITE_API_KEY` at production build time.
// Vite inlines `VITE_*` env vars into the JS bundle, so any value set during
// `pnpm build` would be world-readable in the deployed bundle. The runtime
// `useSettingsStore` (in-memory only) is the sole intended source of API keys.
const forbidBakedApiKey = {
  name: 'forbid-baked-api-key',
  config(_config: unknown, { command, mode }: { command: string; mode: string }) {
    if (command === 'build' && mode === 'production' && process.env.VITE_API_KEY) {
      throw new Error(
        'VITE_API_KEY is set during a production build. ' +
          'API keys must be entered at runtime via the Settings UI to avoid bundle exposure. ' +
          'Unset VITE_API_KEY before running `pnpm build`.',
      );
    }
  },
};

export default defineConfig({
  plugins: [react(), forbidBakedApiKey],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8003',
        changeOrigin: true,
      },
    },
  },
  build: {
    target: 'es2022',
    // SEC-M3: do not ship sourcemaps to production — they expose the full
    // TypeScript source (incl. comments) to anyone who downloads the bundle.
    // Use 'hidden' so we still emit .map files for upload to a private error
    // tracker but they are not referenced from the JS files served publicly.
    sourcemap: 'hidden',
    rollupOptions: {
      output: {
        // FE-m5: split vendor further to avoid a 187 KB gzipped monochunk on first
        // load. Pages that don't use i18next / lucide / react-query can skip those
        // chunks, reducing critical-path parse cost.
        manualChunks: (id) => {
          if (id.includes('lightweight-charts')) return 'charts';
          if (id.includes('react-markdown') || id.includes('rehype-sanitize')) return 'markdown';
          if (id.includes('@radix-ui')) return 'radix';
          if (id.includes('lucide-react')) return 'icons';
          if (id.includes('i18next') || id.includes('react-i18next')) return 'i18n';
          if (id.includes('@tanstack')) return 'query';
          if (id.includes('zustand')) return 'state';
          if (id.includes('zod')) return 'zod';
          if (id.includes('node_modules')) return 'vendor';
          return undefined;
        },
      },
    },
  },
});
