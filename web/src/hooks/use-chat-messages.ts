import { useCallback, useEffect, useRef, useState } from 'react';

import { interruptChatSession } from '@/lib/chat-control';
import { streamFetch, type SSEEvent } from '@/lib/stream-fetch';
import { useChatStore } from '@/stores/use-chat-store';
import type { ChatMessage } from '@/types/api';
import type { AdditionalContext } from '@/types/chart-analysis';
import type { SSEEnvelope } from '@/types/analysis-events';

export type StreamStatus = 'idle' | 'connecting' | 'streaming' | 'error';

export interface UseChatMessagesReturn {
  messages: ChatMessage[];
  status: StreamStatus;
  error: string | null;
  sendMessage: (text: string, additionalContext?: AdditionalContext) => void;
  stopStream: () => void;
  clearMessages: () => void;
}

export function useChatMessages(
  sessionId: string | null,
  onCycleEvent?: (event: SSEEvent) => void,
): UseChatMessagesReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<StreamStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const activeRequestRef = useRef<{ sessionId: string; controller: AbortController } | null>(null);
  const previousSessionIdRef = useRef(sessionId);
  const { upsertSession, setPendingMessage } = useChatStore();

  const interruptAndAbortActiveRequest = useCallback(() => {
    const activeRequest = activeRequestRef.current;
    if (!activeRequest) return false;

    activeRequestRef.current = null;
    void interruptChatSession(activeRequest.sessionId).catch(() => undefined);
    activeRequest.controller.abort();
    return true;
  }, []);

  const stopStream = useCallback(() => {
    if (!interruptAndAbortActiveRequest()) return;
    setStatus('idle');
    setPendingMessage(null);
  }, [interruptAndAbortActiveRequest, setPendingMessage]);

  useEffect(() => () => {
    interruptAndAbortActiveRequest();
  }, [interruptAndAbortActiveRequest]);

  useEffect(() => {
    const previousSessionId = previousSessionIdRef.current;
    previousSessionIdRef.current = sessionId;
    if (previousSessionId === sessionId) return;

    const activeRequest = activeRequestRef.current;
    if (activeRequest && activeRequest.sessionId !== sessionId) {
      stopStream();
    }
  }, [sessionId, stopStream]);

  const handleEvent = useCallback((event: SSEEvent) => {
    onCycleEvent?.(event);
    const envelope = event.data as SSEEnvelope;

    switch (event.event) {
      case 'session_start': {
        const rawPair = envelope.data.pair;
        const pair = typeof rawPair === 'string' ? rawPair : '';
        upsertSession({
          id: envelope.session_id,
          title: pair ? `${pair} 分析` : `Session ${envelope.session_id.slice(0, 8)}`,
          created_at: envelope.ts,
          updated_at: envelope.ts,
        });
        setStatus('streaming');
        break;
      }
      case 'stream_resume':
        setMessages((current) => [...current, {
          id: crypto.randomUUID(),
          role: 'system',
          ts: envelope.ts,
          content_md: '已从断点恢复本轮分析。',
        }]);
        break;
      case 'cycle_cancelled':
        setMessages((current) => [...current, {
          id: crypto.randomUUID(),
          role: 'system',
          ts: envelope.ts,
          content_md: '本轮分析已取消。',
        }]);
        break;
      case 'stream_done':
        setStatus('idle');
        setPendingMessage(null);
        break;
      case 'stream_error': {
        const rawError = envelope.data.error;
        const message = typeof rawError === 'string' ? rawError : 'Analysis error';
        setError(message);
        setStatus('error');
        setPendingMessage(null);
        break;
      }
    }
  }, [onCycleEvent, setPendingMessage, upsertSession]);

  const sendMessage = useCallback((text: string, additionalContext?: AdditionalContext) => {
    if (activeRequestRef.current) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      ts: new Date().toISOString(),
      content_md: text,
    };
    setMessages((current) => [...current, userMessage]);
    setStatus('connecting');
    setError(null);
    setPendingMessage(userMessage);

    const requestSessionId = sessionId ?? crypto.randomUUID();
    const controller = new AbortController();
    activeRequestRef.current = { sessionId: requestSessionId, controller };
    void streamFetch('/api/chat/stream', {
      body: {
        session_id: requestSessionId,
        message: text,
        ...(additionalContext ? { additional_context: additionalContext } : {}),
      },
      signal: controller.signal,
      onEvent: handleEvent,
      onError: (streamError) => {
        setError(streamError.message);
        setStatus('error');
        setPendingMessage(null);
      },
    }).catch((streamError: unknown) => {
      if ((streamError as Error).name === 'AbortError') return;
      setError((streamError as Error).message);
      setStatus('error');
      setPendingMessage(null);
    }).finally(() => {
      if (activeRequestRef.current?.controller === controller) {
        activeRequestRef.current = null;
      }
    });
  }, [handleEvent, sessionId, setPendingMessage]);

  const clearMessages = useCallback(() => {
    stopStream();
    setMessages([]);
    setError(null);
  }, [stopStream]);

  return { messages, status, error, sendMessage, stopStream, clearMessages };
}
