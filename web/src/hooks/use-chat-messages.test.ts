import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useChatStore } from '@/stores/use-chat-store';
import { useSettingsStore } from '@/stores/use-settings-store';
import type { StreamFetchOptions } from '@/lib/stream-fetch';

import { useChatMessages } from './use-chat-messages';

const BEFORE_SEND_VALUE = 'before-send-value';
const CURRENT_VALUE = 'current-value';

const { streamFetchMock } = vi.hoisted(() => ({
  streamFetchMock: vi.fn((_path: string, _options: StreamFetchOptions) => new Promise<void>(() => undefined)),
}));

vi.mock('@/lib/stream-fetch', () => ({
  streamFetch: streamFetchMock,
}));

describe('useChatMessages request lifecycle', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useChatStore.setState({ sessions: [], activeSessionId: null, pendingMessage: null });
    useSettingsStore.setState({ apiKey: BEFORE_SEND_VALUE });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('generates one concrete session ID before streaming and interrupts that same session before aborting', async () => {
    const generatedIds = [
      '00000000-0000-4000-8000-000000000001',
      '00000000-0000-4000-8000-000000000002',
    ];
    vi.stubGlobal('crypto', {
      randomUUID: vi.fn(() => generatedIds.shift() ?? '00000000-0000-4000-8000-000000000099'),
    });

    const effects: string[] = [];
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => {
      effects.push('interrupt');
      return Promise.resolve(new Response(null, { status: 200 }));
    });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useChatMessages(null));
    act(() => result.current.sendMessage('BTC/USDT'));

    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledOnce());
    const streamOptions = streamFetchMock.mock.calls[0]?.[1];
    expect(streamOptions?.body).toEqual(expect.objectContaining({
      session_id: '00000000-0000-4000-8000-000000000002',
      message: 'BTC/USDT',
    }));
    streamOptions?.signal?.addEventListener('abort', () => effects.push('abort'));

    useSettingsStore.setState({ apiKey: CURRENT_VALUE });
    act(() => result.current.stopStream());

    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/chat/interrupt/00000000-0000-4000-8000-000000000002',
      expect.objectContaining({
        method: 'POST',
        headers: { 'X-API-Key': CURRENT_VALUE },
      }),
    );
    expect(effects).toEqual(['interrupt', 'abort']);
    expect(streamOptions?.signal?.aborted).toBe(true);
    expect(result.current.status).toBe('idle');
    expect(useChatStore.getState().pendingMessage).toBeNull();
  });

  it('reuses a selected session ID for both streaming and interrupt', async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => Promise.resolve(new Response(null, { status: 200 })));
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useChatMessages('existing/session'));
    act(() => result.current.sendMessage('continue'));
    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledOnce());

    expect(streamFetchMock.mock.calls[0]?.[1].body).toEqual(expect.objectContaining({
      session_id: 'existing/session',
    }));

    act(() => result.current.stopStream());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/chat/interrupt/existing%2Fsession');
  });

  it('does not interrupt during cleanup when no request is active', () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const { unmount } = renderHook(() => useChatMessages('idle-session'));
    unmount();

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('interrupts an active request when browser navigation changes the selected session', async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => Promise.resolve(new Response(null, { status: 200 })));
    vi.stubGlobal('fetch', fetchMock);

    const { result, rerender } = renderHook(
      ({ selectedSessionId }) => useChatMessages(selectedSessionId),
      { initialProps: { selectedSessionId: 'session-1' as string | null } },
    );
    act(() => result.current.sendMessage('analyze'));
    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledOnce());

    rerender({ selectedSessionId: 'session-2' });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/chat/interrupt/session-1');
    expect(streamFetchMock.mock.calls[0]?.[1].signal?.aborted).toBe(true);
    expect(result.current.status).toBe('idle');
  });

  it('does not interrupt when the generated active session becomes the selected session', async () => {
    const generatedIds = [
      '00000000-0000-4000-8000-000000000011',
      '00000000-0000-4000-8000-000000000012',
    ];
    vi.stubGlobal('crypto', {
      randomUUID: vi.fn(() => generatedIds.shift() ?? '00000000-0000-4000-8000-000000000099'),
    });
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => Promise.resolve(new Response(null, { status: 200 })));
    vi.stubGlobal('fetch', fetchMock);

    const { result, rerender } = renderHook(
      ({ selectedSessionId }) => useChatMessages(selectedSessionId),
      { initialProps: { selectedSessionId: null as string | null } },
    );
    act(() => result.current.sendMessage('analyze'));
    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledOnce());
    const generatedSessionId = (streamFetchMock.mock.calls[0]?.[1].body as { session_id: string }).session_id;

    rerender({ selectedSessionId: generatedSessionId });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(streamFetchMock.mock.calls[0]?.[1].signal?.aborted).toBe(false);

    act(() => result.current.stopStream());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
  });

});
