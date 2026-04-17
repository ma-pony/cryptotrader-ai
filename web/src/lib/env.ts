import { z } from 'zod';

const EnvSchema = z.object({
  VITE_API_BASE_URL: z.string().url().default('http://localhost:8003'),
  VITE_API_KEY: z.string().optional().default(''),
  VITE_OTLP_UI_ENDPOINT: z.string().optional().default(''),
  DEV: z.boolean(),
  PROD: z.boolean(),
  MODE: z.string(),
});

const parsed = EnvSchema.safeParse(import.meta.env);
if (!parsed.success) {
  // Fail-fast on startup so misconfigurations surface immediately.
  throw new Error(`Invalid environment variables: ${JSON.stringify(parsed.error.flatten(), null, 2)}`);
}

export const env = parsed.data;
export const isDev = env.DEV;
export const isProd = env.PROD;
