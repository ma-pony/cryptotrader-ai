/**
 * useChatMessages — SSE streaming hook for multi-agent chat (FR-600~619).
 * Hard limit: ≤ 500 lines (NFR-M-007).
 */
import { useCallback, useEffect, useRef, useState } from 'react';

import type { ChatMessage } from '@/types/api';
import { streamFetch, type SSEEvent } from '@/lib/stream-fetch';
import { useChatStore } from '@/stores/use-chat-store';

// ── Types ──

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface ToolResult {
  tool_call_id: string;
  output_md: string;
}

export interface InlineWidget {
  widget_id: string;
  html: string;
  height_px?: number;
}

interface MessageDelta {
  id: string;
  delta: string;
}

interface MessageStart {
  id: string;
  role: 'assistant';
  ts: string;
}

interface SessionEvent {
  session_id: string;
}

export type StreamStatus = 'idle' | 'connecting' | 'streaming' | 'error';

export interface UseChatMessagesReturn {
  messages: ChatMessage[];
  status: StreamStatus;
  error: string | null;
  sendMessage: (text: string) => void;
  stopStream: () => void;
  clearMessages: () => void;
}

// ── Hook ──

export function useChatMessages(sessionId: string | null): UseChatMessagesReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<StreamStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;

  const { upsertSession, setPendingMessage } = useChatStore();

  const stopStream = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus('idle');
    setPendingMessage(null);
  }, [setPendingMessage]);

  useEffect(() => () => stopStream(), [stopStream]);

  const handleEvent = useCallback(
    (event: SSEEvent) => {
      switch (event.event) {
        case 'session': {
          const { session_id } = event.data as SessionEvent;
          upsertSession({
            id: session_id,
            title: `Session ${session_id.slice(0, 8)}`,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
          break;
        }

        case 'message_start': {
          const start = event.data as MessageStart;
          const newMsg: ChatMessage = {
            id: start.id,
            role: start.role,
            ts: start.ts,
            content_md: '',
          };
          setMessages((prev) => [...prev, newMsg]);
          setStatus('streaming');
          break;
        }

        case 'content_delta': {
          const delta = event.data as MessageDelta;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === delta.id ? { ...m, content_md: (m.content_md ?? '') + delta.delta } : m,
            ),
          );
          break;
        }

        case 'tool_call': {
          const tc = event.data as ToolCall & { id: string };
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (!last) return prev;
            const existing = last.tool_calls ?? [];
            return [
              ...prev.slice(0, -1),
              { ...last, tool_calls: [...existing, { id: tc.id, name: tc.name, args: tc.args }] },
            ];
          });
          break;
        }

        case 'tool_result': {
          const tr = event.data as ToolResult;
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (!last) return prev;
            const existing = last.tool_results ?? [];
            return [
              ...prev.slice(0, -1),
              {
                ...last,
                tool_results: [...existing, { tool_call_id: tr.tool_call_id, output_md: tr.output_md }],
              },
            ];
          });
          break;
        }

        case 'inline_widget': {
          const w = event.data as InlineWidget;
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (!last) return prev;
            const existing = last.inline_widgets ?? [];
            return [
              ...prev.slice(0, -1),
              { ...last, inline_widgets: [...existing, w] },
            ];
          });
          break;
        }

        case 'message_end': {
          setPendingMessage(null);
          break;
        }

        case 'done': {
          setStatus('idle');
          break;
        }
      }
    },
    [upsertSession, setPendingMessage],
  );

  const sendMessage = useCallback(
    (text: string) => {
      if (status === 'streaming' || status === 'connecting') return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        ts: new Date().toISOString(),
        content_md: text,
      };
      setMessages((prev) => [...prev, userMsg]);
      setStatus('connecting');
      setError(null);
      setPendingMessage(userMsg);

      const controller = new AbortController();
      abortRef.current = controller;

      void streamFetch('/api/chat/stream', {
        body: {
          session_id: sessionId ?? '',
          message: text,
        },
        signal: controller.signal,
        onEvent: handleEvent,
        onError: (err) => {
          setError(err.message);
          setStatus('error');
          setPendingMessage(null);
        },
      }).catch((err: unknown) => {
        if ((err as Error).name === 'AbortError') return;
        setError((err as Error).message);
        setStatus('error');
        setPendingMessage(null);
      });
    },
    [sessionId, status, handleEvent, setPendingMessage],
  );

  const clearMessages = useCallback(() => {
    stopStream();
    setMessages([]);
    setError(null);
  }, [stopStream]);

  return { messages, status, error, sendMessage, stopStream, clearMessages };
}
