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
    // Switch to polling mode. We tried scope-limiting chokidar fsevents
    // first (ignored: ['../src/**', '../.venv/**', ...]), which reduced the
    // fsevents-overflow blast radius but didn't fix the underlying issue —
    // after a long-running dev session (~75 min, 10+ commits, restored
    // working-tree files, etc.) Vite's in-memory module resolver gradually
    // degrades and `@/stores/use-*` aliases start failing on healthy files.
    //
    // Polling sidesteps the fsevents path entirely:
    //   - chokidar stats files at fixed intervals instead of waiting for
    //     OS notifications, so the resolver state doesn't drift on weird
    //     event bursts (git checkout, pytest cache writes, etc.).
    //   - ~1s HMR latency vs ~100ms for fsevents — fine for a single dev.
    //   - CPU cost ~1-2% with `interval: 1500`, scoped to web/src.
    watch: {
      usePolling: true,
      interval: 1500,
      binaryInterval: 3000,
      ignored: [
        '**/node_modules/**',
        '**/.git/**',
        '**/dist/**',
        '**/coverage/**',
        '**/.pytest_cache/**',
        // Everything outside web/ — Python source, venv, agent memory, logs.
        path.resolve(__dirname, '..', 'src') + '/**',
        path.resolve(__dirname, '..', 'tests') + '/**',
        path.resolve(__dirname, '..', '.venv') + '/**',
        path.resolve(__dirname, '..', 'agent_memory') + '/**',
        path.resolve(__dirname, '..', 'agent_skills') + '/**',
        path.resolve(__dirname, '..', 'specs') + '/**',
        path.resolve(__dirname, '..', 'brainstorm') + '/**',
        path.resolve(__dirname, '..', '.specify') + '/**',
        path.resolve(__dirname, '..', '.claude') + '/**',
      ],
    },
    fs: {
      // Confine module resolution to the web subtree so a stray import that
      // escapes via `@/` won't crawl the entire monorepo.
      allow: [path.resolve(__dirname)],
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
