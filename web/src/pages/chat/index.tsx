import { useCallback, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useParams, useNavigate, useLocation } from 'react-router';

import type { AdditionalContext } from '@/types/chart-analysis';
import { AnalysisProgressPanel } from '@/components/analysis/analysis-progress-panel';
import { useAnalysisProgress } from '@/hooks/use-analysis-progress';
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
  const location = useLocation();

  const { sessions, activeSessionId, setActiveSession, removeSession } = useChatStore();
  const currentSessionId = sessionId ?? activeSessionId;
  const { messages, status, error, sendMessage, stopStream, clearMessages } = useChatMessages(currentSessionId);
  const { progress, sendInterrupt, sendSteer } = useAnalysisProgress();

  const initialContextRef = useRef(false);
  useEffect(() => {
    if (initialContextRef.current) return;
    const state = location.state as { additionalContext?: AdditionalContext } | null;
    if (!state?.additionalContext) return;
    initialContextRef.current = true;
    const timer = setTimeout(() => {
      sendMessage('请分析当前图表', state.additionalContext);
    }, 100);
    return () => clearTimeout(timer);
  }, [location.state, sendMessage]);

  const chartContext = (location.state as { additionalContext?: AdditionalContext } | null)?.additionalContext;

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
    // h-full fills the <main> flex-child height (viewport minus topbar minus
    // main padding). Previous h-[calc(100vh-4rem)] hardcoded the topbar at
    // 4rem but the actual topbar is h-14 (3.5rem) and ignored main's py-6,
    // leaving an 8px gap at the bottom and an awkward overflow on tall screens.
    <div className="flex h-full gap-4">
      <Card className="hidden w-64 shrink-0 overflow-hidden lg:block">
        <SessionList
          sessions={sessions}
          activeId={currentSessionId ?? null}
          onSelect={handleSelectSession}
          onNew={handleNewSession}
          onRemove={handleRemoveSession}
        />
      </Card>

      <Card className="flex flex-1 flex-col overflow-hidden">
        <div className="border-b border-border px-4 py-3">
          <h1 className="text-lg font-semibold text-foreground">{t('title')}</h1>
          {error && <p className="mt-1 text-xs text-destructive">{error}</p>}
          {chartContext && (
            <div className="mt-1 flex items-center gap-2 text-xs text-muted-foreground">
              <span className="inline-block rounded bg-blue-500/20 px-1.5 py-0.5 text-blue-400">
                {t('chart_context_badge', '已附加图表上下文')}
              </span>
              <span>{chartContext.payloads[0]?.symbol} / {chartContext.payloads[0]?.timeframe}</span>
            </div>
          )}
        </div>

        <AnalysisProgressPanel
          progress={progress}
          sessionId={currentSessionId ?? null}
          onSteer={currentSessionId
            ? (target, instruction) => void sendSteer(currentSessionId, target, instruction)
            : undefined}
          onInterrupt={currentSessionId
            ? () => void sendInterrupt(currentSessionId)
            : undefined}
        />

        <div className="flex-1 overflow-hidden">
          <MessageStream messages={messages} status={status} />
        </div>

        <ChatInput onSend={sendMessage} onStop={stopStream} status={status} />
      </Card>
    </div>
  );
};

export default ChatPage;
