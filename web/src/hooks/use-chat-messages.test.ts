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

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};

const terminalEvent = (type: 'stream_done' | 'stream_error', eventId: number) => ({
  event: type,
  data: {
    event_id: eventId,
    type,
    ts: '2026-08-28T00:00:00Z',
    session_id: 'session-1',
    data: type === 'stream_error' ? { error: 'failed' } : { status: 'completed' },
  },
});

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
    act(() => {
      result.current.sendMessage('BTC/USDT');
    });

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
    act(() => {
      result.current.sendMessage('continue');
    });
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
    act(() => {
      result.current.sendMessage('analyze');
    });
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
    act(() => {
      result.current.sendMessage('analyze');
    });
    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledOnce());
    const generatedSessionId = (streamFetchMock.mock.calls[0]?.[1].body as { session_id: string }).session_id;

    rerender({ selectedSessionId: generatedSessionId });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(streamFetchMock.mock.calls[0]?.[1].signal?.aborted).toBe(false);

    act(() => result.current.stopStream());
    await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
  });

  it.each(['stream_done', 'stream_error'] as const)(
    'releases only the matching run on %s before the HTTP body reaches EOF',
    async (terminalType) => {
      const firstStream = deferred<void>();
      const secondStream = deferred<void>();
      streamFetchMock
        .mockImplementationOnce(() => firstStream.promise)
        .mockImplementationOnce(() => secondStream.promise);
      const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => Promise.resolve(new Response(null, { status: 200 })));
      vi.stubGlobal('fetch', fetchMock);

      const { result } = renderHook(() => useChatMessages('session-1'));
      act(() => {
        result.current.sendMessage('first');
      });
      await waitFor(() => expect(streamFetchMock).toHaveBeenCalledOnce());
      const firstOptions = streamFetchMock.mock.calls[0]?.[1];

      act(() => firstOptions?.onEvent?.(terminalEvent(terminalType, 1)));
      let secondAccepted: boolean | undefined;
      act(() => {
        secondAccepted = result.current.sendMessage('second');
      });

      expect(secondAccepted).toBe(true);
      await waitFor(() => expect(streamFetchMock).toHaveBeenCalledTimes(2));
      expect(result.current.status).toBe('connecting');
      expect(result.current.messages.map((message) => message.content_md)).toEqual(['first', 'second']);
      expect(fetchMock).not.toHaveBeenCalled();

      act(() => firstOptions?.onEvent?.(terminalEvent(terminalType, 2)));
      firstStream.resolve();
      await act(async () => Promise.resolve());

      expect(result.current.status).toBe('connecting');
      expect(useChatStore.getState().pendingMessage?.content_md).toBe('second');

      const secondOptions = streamFetchMock.mock.calls[1]?.[1];
      act(() => secondOptions?.onEvent?.(terminalEvent('stream_done', 3)));
      secondStream.resolve();
    },
  );

  it('waits for a delayed interrupt before reusing the same session', async () => {
    const interrupt = deferred<Response>();
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => interrupt.promise);
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useChatMessages('session-1'));
    act(() => {
      result.current.sendMessage('first');
    });
    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledOnce());
    act(() => result.current.stopStream());

    let accepted: boolean | undefined;
    act(() => {
      accepted = result.current.sendMessage('second');
    });

    expect(accepted).toBe(true);
    expect(result.current.status).toBe('connecting');
    expect(streamFetchMock).toHaveBeenCalledOnce();

    interrupt.resolve(new Response(null, { status: 200 }));
    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledTimes(2));
    act(() => streamFetchMock.mock.calls[1]?.[1].onEvent?.(terminalEvent('stream_done', 2)));
  });

  it('keeps the same-session interrupt barrier across unmount and remount', async () => {
    const interrupt = deferred<Response>();
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => interrupt.promise);
    vi.stubGlobal('fetch', fetchMock);

    const firstHook = renderHook(() => useChatMessages('session-1'));
    act(() => {
      firstHook.result.current.sendMessage('first');
    });
    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledOnce());
    act(() => firstHook.result.current.stopStream());
    firstHook.unmount();

    const secondHook = renderHook(() => useChatMessages('session-1'));
    act(() => {
      secondHook.result.current.sendMessage('second');
    });
    expect(secondHook.result.current.status).toBe('connecting');
    expect(streamFetchMock).toHaveBeenCalledOnce();

    interrupt.resolve(new Response(null, { status: 200 }));
    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledTimes(2));
    act(() => streamFetchMock.mock.calls[1]?.[1].onEvent?.(terminalEvent('stream_done', 2)));
  });

  it('does not block a different session behind another session interrupt', async () => {
    const interrupt = deferred<Response>();
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => interrupt.promise);
    vi.stubGlobal('fetch', fetchMock);

    const { result, rerender } = renderHook(
      ({ selectedSessionId }) => useChatMessages(selectedSessionId),
      { initialProps: { selectedSessionId: 'session-1' } },
    );
    act(() => {
      result.current.sendMessage('first');
    });
    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledOnce());
    act(() => result.current.stopStream());
    rerender({ selectedSessionId: 'session-2' });
    act(() => {
      result.current.sendMessage('second');
    });

    await waitFor(() => expect(streamFetchMock).toHaveBeenCalledTimes(2));
    interrupt.resolve(new Response(null, { status: 200 }));
    act(() => streamFetchMock.mock.calls[1]?.[1].onEvent?.(terminalEvent('stream_done', 2)));
  });

});
