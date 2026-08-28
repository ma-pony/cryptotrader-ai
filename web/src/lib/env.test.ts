import { describe, expect, it } from 'vitest';

import { parseRuntimeEnv } from './env';

describe('runtime environment', () => {
  it('defaults the browser API base to same-origin', () => {
    const parsed = parseRuntimeEnv({
      DEV: false,
      PROD: true,
      MODE: 'production',
    });

    expect(parsed.VITE_API_BASE_URL).toBe('');
  });

  it('accepts an absolute API base for separate deployments', () => {
    const parsed = parseRuntimeEnv({
      VITE_API_BASE_URL: 'https://api.example.com',
      DEV: false,
      PROD: true,
      MODE: 'production',
    });

    expect(parsed.VITE_API_BASE_URL).toBe('https://api.example.com');
  });
});
