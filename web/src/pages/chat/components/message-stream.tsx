import { Loader2 } from 'lucide-react';
import { memo, useEffect, useRef, type FC } from 'react';
import { useTranslation } from 'react-i18next';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';

import { ScrollArea } from '@/components/ui/scroll-area';
import type { StreamStatus } from '@/hooks/use-chat-messages';
import type { ChatMessage } from '@/types/api';

interface MessageStreamProps {
  messages: ChatMessage[];
  status: StreamStatus;
}

const UserBubble = memo(({ message }: { message: ChatMessage }) => (
  <div className="flex justify-end">
    <div className="max-w-[75%] rounded-[14px_14px_2px_14px] bg-amber-500 px-3.5 py-2.5 text-sm font-medium text-primary-foreground shadow-sm">
      {message.content_md ? <div className="prose prose-sm prose-invert max-w-none"><ReactMarkdown rehypePlugins={[rehypeSanitize]}>{message.content_md}</ReactMarkdown></div> : null}
    </div>
  </div>
));

const SystemLine = ({ message }: { message: ChatMessage }) => (
  <div className="flex justify-center">
    <div className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/35 px-3 py-1 text-[11px] text-muted-foreground">
      <span className="h-1 w-1 rounded-full bg-muted-foreground" />
      {message.content_md}
    </div>
  </div>
);

export const MessageStream: FC<MessageStreamProps> = ({ messages, status }) => {
  const { t } = useTranslation('chat');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return <div className="flex h-full items-center justify-center text-muted-foreground"><p>{t('empty')}</p></div>;
  }

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto flex w-full max-w-[860px] flex-col gap-3.5 px-5 py-5">
        {messages.map((message) => message.role === 'user'
          ? <UserBubble key={message.id} message={message} />
          : <SystemLine key={message.id} message={message} />)}
        {status === 'connecting' || status === 'streaming' ? (
          <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>{status === 'connecting' ? '正在连接…' : '周期运行中…'}</span>
          </div>
        ) : null}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
};
