import { afterEach, expect, it, vi } from 'vitest';
import { z } from 'zod';
import { ApiError, apiClient } from './api-client';

afterEach(() => vi.unstubAllGlobals());

it.each([
  [401, { detail: { code: 'authentication_failed' } }, 'authentication_failed', 'authentication_failed'],
  [503, { detail: { code: 'credentials_missing' } }, 'credentials_missing', 'credentials_missing'],
  [
    409,
    { detail: 'Runtime configuration changed; reload and retry' },
    'HTTP_409',
    'Runtime configuration changed; reload and retry',
  ],
])('normalizes the real FastAPI %i error envelope', async (status, payload, code, message) => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(payload), { status })));
  await expect(apiClient.get('/api/config', z.unknown())).rejects.toMatchObject({
    name: 'ApiError',
    status,
    code,
    message,
  });
});

it('retains validation paths without reflecting input values or serialized error objects', async () => {
  vi.stubGlobal(
    'fetch',
    vi
      .fn()
      .mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: [
              {
                loc: ['body', 'document', 'llm', 'timeout'],
                type: 'int_parsing',
                msg: 'Input should be a valid integer',
                input: 'private-marker',
              },
            ],
          }),
          { status: 422 },
        ),
      ),
  );
  const failure = await apiClient.get('/api/config', z.unknown()).catch((error: unknown) => error);
  expect(failure).toBeInstanceOf(ApiError);
  expect(failure).toMatchObject({
    code: 'validation_error',
    details: { fieldErrors: { 'llm.timeout': 'Invalid value' } },
  });
  expect(JSON.stringify(failure)).not.toContain('private-marker');
});
