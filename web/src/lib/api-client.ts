import type { z } from 'zod';

import { env } from './env';
import { useSettingsStore } from '@/stores/use-settings-store';
import { ApiErrorSchema, type ApiError as ApiErrorShape } from '@/types/api.schema';

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly traceId?: string;
  readonly details?: Record<string, unknown>;

  constructor(status: number, payload: ApiErrorShape) {
    super(payload.message);
    this.name = 'ApiError';
    this.status = status;
    this.code = payload.code;
    if (payload.trace_id) this.traceId = payload.trace_id;
    if (payload.details) this.details = payload.details;
  }
}

interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: unknown;
  signal?: AbortSignal;
  skipApiKey?: boolean;
}

const buildUrl = (path: string): string => {
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  const base = env.VITE_API_BASE_URL.replace(/\/$/, '');
  const suffix = path.startsWith('/') ? path : `/${path}`;
  return `${base}${suffix}`;
};

const resolveApiKey = (): string => {
  const runtimeKey = useSettingsStore.getState().apiKey;
  return runtimeKey ?? env.VITE_API_KEY ?? '';
};

async function parseError(res: Response): Promise<ApiError> {
  let payload: ApiErrorShape = { code: `HTTP_${res.status}`, message: res.statusText || 'Request failed' };
  try {
    const json: unknown = await res.json();
    const parsed = ApiErrorSchema.safeParse(json);
    if (parsed.success) payload = parsed.data;
  } catch {
    // Body not JSON; keep default payload.
  }
  return new ApiError(res.status, payload);
}

async function request<S extends z.ZodTypeAny>(path: string, schema: S, options: RequestOptions = {}): Promise<z.output<S>> {
  const headers = new Headers(options.headers);
  if (!headers.has('Accept')) headers.set('Accept', 'application/json');
  if (options.body !== undefined && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');

  if (!options.skipApiKey) {
    const apiKey = resolveApiKey();
    if (apiKey) headers.set('X-API-Key', apiKey);
  }

  const { body, signal, skipApiKey: _skip, ...rest } = options;
  void _skip;
  const init: RequestInit = {
    ...rest,
    headers,
  };
  if (body !== undefined) init.body = JSON.stringify(body);
  if (signal) init.signal = signal;

  const res = await fetch(buildUrl(path), init);
  if (!res.ok) throw await parseError(res);

  if (res.status === 204) return undefined as z.output<S>;
  const json: unknown = await res.json();
  return schema.parse(json) as z.output<S>;
}

export const apiClient = {
  get: <S extends z.ZodTypeAny>(path: string, schema: S, options?: Omit<RequestOptions, 'body'>) =>
    request(path, schema, { ...options, method: 'GET' }),
  post: <S extends z.ZodTypeAny>(path: string, body: unknown, schema: S, options?: RequestOptions) =>
    request(path, schema, { ...options, method: 'POST', body }),
  put: <S extends z.ZodTypeAny>(path: string, body: unknown, schema: S, options?: RequestOptions) =>
    request(path, schema, { ...options, method: 'PUT', body }),
  delete: <S extends z.ZodTypeAny>(path: string, schema: S, options?: Omit<RequestOptions, 'body'>) =>
    request(path, schema, { ...options, method: 'DELETE' }),
  patch: <S extends z.ZodTypeAny>(path: string, body: unknown, schema: S, options?: RequestOptions) =>
    request(path, schema, { ...options, method: 'PATCH', body }),
};

export type { RequestOptions };
