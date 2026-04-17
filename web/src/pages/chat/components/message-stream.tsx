import { Bot, Loader2, User, Wrench } from 'lucide-react';
import { useEffect, useRef, type FC } from 'react';
import { useTranslation } from 'react-i18next';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';

import { InlineWidget } from '@/components/inline-widget/inline-widget';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { ChatMessage } from '@/types/api';
import type { StreamStatus } from '@/hooks/use-chat-messages';
import { cn } from '@/lib/cn';

interface MessageStreamProps {
  messages: ChatMessage[];
  status: StreamStatus;
}

const RoleIcon: FC<{ role: string }> = ({ role }) =>
  role === 'user' ? (
    <User className="h-5 w-5 shrink-0 text-primary" />
  ) : (
    <Bot className="h-5 w-5 shrink-0 text-muted-foreground" />
  );

const MessageBubble: FC<{ msg: ChatMessage }> = ({ msg }) => {
  const { t } = useTranslation('chat');
  const isUser = msg.role === 'user';

  return (
    <div className={cn('flex gap-3 px-4 py-3', isUser && 'flex-row-reverse')}>
      <RoleIcon role={msg.role} />
      <div className={cn('max-w-[80%] space-y-2', isUser && 'text-right')}>
        {msg.content_md && (
          <div className="prose prose-sm prose-invert max-w-none">
            <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{msg.content_md}</ReactMarkdown>
          </div>
        )}

        {msg.tool_calls && msg.tool_calls.length > 0 && (
          <div className="space-y-1">
            {msg.tool_calls.map((tc) => (
              <div key={tc.id} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Wrench className="h-3 w-3" />
                <span>
                  {t('tool_call')}: {tc.name}
                </span>
              </div>
            ))}
          </div>
        )}

        {msg.tool_results && msg.tool_results.length > 0 && (
          <div className="space-y-1">
            {msg.tool_results.map((tr) => (
              <div key={tr.tool_call_id} className="rounded bg-muted/40 p-2 text-xs">
                <Badge variant="outline" className="mb-1">
                  {t('tool_result')}
                </Badge>
                <div className="prose prose-sm prose-invert max-w-none">
                  <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{tr.output_md}</ReactMarkdown>
                </div>
              </div>
            ))}
          </div>
        )}

        {msg.inline_widgets && msg.inline_widgets.length > 0 && (
          <div className="space-y-2">
            {msg.inline_widgets.map((w) => (
              <InlineWidget key={w.widget_id} html={w.html} heightPx={w.height_px} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export const MessageStream: FC<MessageStreamProps> = ({ messages, status }) => {
  const { t } = useTranslation('chat');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground">
        <p>{t('empty')}</p>
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="divide-y divide-border">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}
        {(status === 'connecting' || status === 'streaming') && (
          <div className="flex items-center gap-2 px-4 py-3 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>{status === 'connecting' ? 'Connecting…' : 'Streaming…'}</span>
          </div>
        )}
      </div>
      <div ref={bottomRef} />
    </ScrollArea>
  );
};
