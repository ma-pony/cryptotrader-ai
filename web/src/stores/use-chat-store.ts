import { create } from 'zustand';

import type { ChatMessage } from '@/types/api';

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;
  pendingMessage: ChatMessage | null;
  setActiveSession: (id: string | null) => void;
  upsertSession: (session: ChatSession) => void;
  removeSession: (id: string) => void;
  setPendingMessage: (msg: ChatMessage | null) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  sessions: [],
  activeSessionId: null,
  pendingMessage: null,
  setActiveSession: (activeSessionId) => set({ activeSessionId }),
  upsertSession: (session) =>
    set((s) => {
      const next = s.sessions.filter((x) => x.id !== session.id);
      next.unshift(session);
      return { sessions: next, activeSessionId: session.id };
    }),
  removeSession: (id) =>
    set((s) => ({
      sessions: s.sessions.filter((x) => x.id !== id),
      activeSessionId: s.activeSessionId === id ? null : s.activeSessionId,
    })),
  setPendingMessage: (pendingMessage) => set({ pendingMessage }),
}));
