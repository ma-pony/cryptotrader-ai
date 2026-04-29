import { env } from './env';
import { useSettingsStore } from '@/stores/use-settings-store';

// SSE event envelope per contracts/sse-events.md
export interface SSEEvent<T = unknown> {
  event: string;
  data: T;
  id?: string;
}

export interface StreamFetchOptions {
  body?: unknown;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  onEvent?: (event: SSEEvent) => void;
  onError?: (error: Error) => void;
  debug?: boolean;
}

/**
 * SSE error: status-aware so consumers can distinguish transient vs fatal.
 */
export class SSEError extends Error {
  readonly status: number;
  readonly retryable: boolean;
  constructor(status: number, message: string, retryable: boolean) {
    super(message);
    this.name = 'SSEError';
    this.status = status;
    this.retryable = retryable;
  }
}

const RETRYABLE_STATUSES = new Set([429, 502, 503, 504]);

const buildUrl = (path: string): string => {
  if (path.startsWith('http://') || path.startsWith('https://')) return path;
  const base = env.VITE_API_BASE_URL.replace(/\/$/, '');
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
};

const resolveApiKey = (): string => useSettingsStore.getState().apiKey;

// Patch JSON.parse to tolerate NaN/Infinity values that some agents emit.
const safeJsonParse = (raw: string): unknown => {
  if (raw === '') return null;
  const sanitized = raw.replace(/\bNaN\b/g, 'null').replace(/\bInfinity\b/g, 'null').replace(/\b-Infinity\b/g, 'null');
  try {
    return JSON.parse(sanitized);
  } catch {
    return raw;
  }
};

const parseEvent = (frame: string): SSEEvent | null => {
  // Each SSE frame: lines like "event: foo\ndata: ...\n\n". data may span multiple lines.
  const lines = frame.split(/\r?\n/);
  let event = 'message';
  let id: string | undefined;
  const dataChunks: string[] = [];
  for (const line of lines) {
    if (!line || line.startsWith(':')) continue;
    const sep = line.indexOf(':');
    const field = sep === -1 ? line : line.slice(0, sep);
    const value = sep === -1 ? '' : line.slice(sep + 1).replace(/^ /, '');
    if (field === 'event') event = value;
    else if (field === 'data') dataChunks.push(value);
    else if (field === 'id') id = value;
  }
  if (dataChunks.length === 0) return null;
  const dataRaw = dataChunks.join('\n');
  const result: SSEEvent = { event, data: safeJsonParse(dataRaw) };
  if (id !== undefined) result.id = id;
  return result;
};

/**
 * Streamed POST against an SSE endpoint. Replaces EventSource because we need
 * to inject `X-API-Key` as a header (NFR-S-001) which EventSource cannot do.
 */
export async function streamFetch(path: string, options: StreamFetchOptions): Promise<void> {
  const headers: Record<string, string> = {
    Accept: 'text/event-stream',
    'Cache-Control': 'no-cache',
    ...(options.headers ?? {}),
  };
  if (options.body !== undefined && !('Content-Type' in headers)) headers['Content-Type'] = 'application/json';
  const apiKey = resolveApiKey();
  if (apiKey) headers['X-API-Key'] = apiKey;

  const init: RequestInit = {
    method: options.body !== undefined ? 'POST' : 'GET',
    headers,
  };
  if (options.body !== undefined) init.body = JSON.stringify(options.body);
  if (options.signal) init.signal = options.signal;

  const res = await fetch(buildUrl(path), init);
  if (!res.ok) {
    const retryable = RETRYABLE_STATUSES.has(res.status);
    const text = await res.text().catch(() => '');
    throw new SSEError(res.status, text || res.statusText || `Stream request failed (${String(res.status)})`, retryable);
  }
  if (!res.body) {
    throw new SSEError(0, 'Response body missing for SSE stream', false);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  const debug = options.debug ?? env.DEV;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += value;
      const parts = buffer.split(/\r?\n\r?\n/);
      buffer = parts.pop() ?? '';
      for (const frame of parts) {
        const event = parseEvent(frame);
        if (!event) continue;
        if (debug) {
          console.debug('[SSE]', event.event, event.data);
        }
        options.onEvent?.(event);
      }
    }
    if (buffer.trim() !== '') {
      const tail = parseEvent(buffer);
      if (tail) options.onEvent?.(tail);
    }
  } catch (err) {
    if ((err as Error).name === 'AbortError') return;
    options.onError?.(err as Error);
    throw err;
  } finally {
    reader.releaseLock();
  }
}
