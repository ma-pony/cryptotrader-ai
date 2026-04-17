import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams, useNavigate } from 'react-router';

import { useChatMessages } from '@/hooks/use-chat-messages';
import { useChatStore } from '@/stores/use-chat-store';
import { Card } from '@/components/ui/card';

import { ChatInput } from './components/chat-input';
import { MessageStream } from './components/message-stream';
import { SessionList } from './components/session-list';

const ChatPage = () => {
  const { t } = useTranslation('chat');
  const { sessionId } = useParams<{ sessionId?: string }>();
  const navigate = useNavigate();

  const { sessions, activeSessionId, setActiveSession, removeSession } = useChatStore();
  const currentSessionId = sessionId ?? activeSessionId;
  const { messages, status, error, sendMessage, stopStream, clearMessages } = useChatMessages(currentSessionId);

  const handleSelectSession = useCallback(
    (id: string) => {
      setActiveSession(id);
      void navigate(`/chat/${id}`);
    },
    [setActiveSession, navigate],
  );

  const handleNewSession = useCallback(() => {
    setActiveSession(null);
    clearMessages();
    void navigate('/chat');
  }, [setActiveSession, clearMessages, navigate]);

  const handleRemoveSession = useCallback(
    (id: string) => {
      removeSession(id);
      if (currentSessionId === id) {
        clearMessages();
        void navigate('/chat');
      }
    },
    [removeSession, currentSessionId, clearMessages, navigate],
  );

  return (
    <div className="flex h-[calc(100vh-4rem)] gap-4">
      {/* Sidebar — session list */}
      <Card className="hidden w-64 shrink-0 overflow-hidden lg:block">
        <SessionList
          sessions={sessions}
          activeId={currentSessionId ?? null}
          onSelect={handleSelectSession}
          onNew={handleNewSession}
          onRemove={handleRemoveSession}
        />
      </Card>

      {/* Main chat area */}
      <Card className="flex flex-1 flex-col overflow-hidden">
        <div className="border-b border-border px-4 py-3">
          <h1 className="text-lg font-semibold text-foreground">{t('title')}</h1>
          {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
        </div>

        <div className="flex-1 overflow-hidden">
          <MessageStream messages={messages} status={status} />
        </div>

        <ChatInput onSend={sendMessage} onStop={stopStream} status={status} />
      </Card>
    </div>
  );
};

export default ChatPage;
