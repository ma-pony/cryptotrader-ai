import { Bot, Loader2, Wrench, Zap } from 'lucide-react';
import { memo, useEffect, useRef, type FC } from 'react';
import { useTranslation } from 'react-i18next';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';

import { InlineWidget } from '@/components/inline-widget/inline-widget';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/cn';
import type { ChatMessage } from '@/types/api';
import type { StreamStatus } from '@/hooks/use-chat-messages';

interface MessageStreamProps {
  messages: ChatMessage[];
  status: StreamStatus;
}

/**
 * Detect "verdict"-style assistant messages. Only the explicit markdown heading
 * form counts — FE-I13 removed the broader ``\bverdict\b`` fallback that used to
 * fire on any assistant message mentioning "verdict" in its first 120 chars
 * (e.g. "the verdict node returned..." or code blocks containing the word).
 *
 * A real verdict message starts with a heading line like ``## Verdict`` /
 * ``## AI 裁决`` — those are the only ones that get the amber glow card.
 */
const isVerdict = (msg: ChatMessage): boolean => {
  if (msg.role !== 'assistant') return false;
  const md = msg.content_md ?? '';
  return /^\s*#+\s*(verdict|裁决|ai 裁决)/i.test(md);
};

// FE-I8: memoize bubbles so previously-rendered messages don't re-parse markdown
// on every streaming token. ChatMessage objects are stable references (new tokens
// append to the array; existing objects are not mutated), so reference equality
// on ``msg`` is sufficient to short-circuit rerenders.
const UserBubbleInner = ({ msg }: { msg: ChatMessage }) => (
  <div className="flex justify-end">
    <div
      className="max-w-[75%] rounded-[14px_14px_2px_14px] px-3.5 py-2.5 text-sm font-medium shadow-sm"
      style={{ background: 'var(--amber-500)', color: 'hsl(var(--primary-foreground))' }}
    >
      {msg.content_md ? (
        <div className="prose prose-sm prose-invert max-w-none">
          <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{msg.content_md}</ReactMarkdown>
        </div>
      ) : null}
    </div>
  </div>
);

const SystemLine = ({ msg }: { msg: ChatMessage }) => (
  <div className="flex justify-center">
    <div className="inline-flex items-center gap-1.5 text-[11px] italic text-muted-foreground">
      <span className="h-1 w-1 rounded-full bg-muted-foreground animate-ct-pulse" />
      {msg.content_md}
    </div>
  </div>
);

const UserBubble = memo(UserBubbleInner);

const AssistantBubbleInner = ({ msg }: { msg: ChatMessage }) => {
  const { t } = useTranslation('chat');
  const verdict = isVerdict(msg);
  const accent = verdict ? 'var(--amber-500)' : 'var(--violet-500)';

  return (
    <div className="flex items-start gap-2.5">
      <span
        className="inline-flex h-7 w-7 items-center justify-center rounded-md border"
        style={{
          color: accent,
          background: `color-mix(in oklch, ${accent} 18%, transparent)`,
          borderColor: `color-mix(in oklch, ${accent} 40%, transparent)`,
        }}
      >
        {verdict ? <Zap size={15} strokeWidth={1.8} /> : <Bot size={15} strokeWidth={1.8} />}
      </span>
      <div className="flex-1 max-w-[720px] space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold" style={{ color: accent }}>
            {verdict ? 'AI 首席决策者' : t('assistant', { defaultValue: 'Assistant' })}
          </span>
        </div>
        <div
          className={cn(
            'rounded-[2px_14px_14px_14px] border p-3.5 text-sm leading-relaxed',
            verdict && 'shadow-glow-amber font-medium',
          )}
          style={{
            background: verdict
              ? 'linear-gradient(135deg, color-mix(in oklch, var(--amber-500) 12%, transparent), hsl(var(--card)))'
              : 'hsl(var(--card))',
            borderColor: verdict
              ? 'color-mix(in oklch, var(--amber-500) 35%, transparent)'
              : `color-mix(in oklch, ${accent} 20%, hsl(var(--border)))`,
          }}
        >
          {msg.content_md ? (
            <div className="prose prose-sm prose-invert max-w-none">
              <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{msg.content_md}</ReactMarkdown>
            </div>
          ) : null}

          {msg.tool_calls && msg.tool_calls.length > 0 ? (
            <div className="mt-2 space-y-1 border-t border-border/50 pt-2">
              {msg.tool_calls.map((tc) => (
                <div key={tc.id} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Wrench className="h-3 w-3" />
                  <span>
                    {t('tool_call', { defaultValue: 'tool call' })}: {tc.name}
                  </span>
                </div>
              ))}
            </div>
          ) : null}

          {msg.tool_results && msg.tool_results.length > 0 ? (
            <div className="mt-2 space-y-1">
              {msg.tool_results.map((tr) => (
                <div key={tr.tool_call_id} className="rounded bg-muted/40 p-2 text-xs">
                  <Badge variant="secondary" className="mb-1">
                    {t('tool_result', { defaultValue: 'tool result' })}
                  </Badge>
                  <div className="prose prose-sm prose-invert max-w-none">
                    <ReactMarkdown rehypePlugins={[rehypeSanitize]}>{tr.output_md}</ReactMarkdown>
                  </div>
                </div>
              ))}
            </div>
          ) : null}

          {msg.inline_widgets && msg.inline_widgets.length > 0 ? (
            <div className="mt-2 space-y-2">
              {msg.inline_widgets.map((w) => (
                <InlineWidget key={w.widget_id} html={w.html} heightPx={w.height_px ?? 320} />
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};

const AssistantBubble = memo(AssistantBubbleInner);

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
      <div className="mx-auto flex w-full max-w-[860px] flex-col gap-3.5 px-5 py-5">
        {messages.map((msg) => {
          if (msg.role === 'user') return <UserBubble key={msg.id} msg={msg} />;
          if (msg.role === 'system') return <SystemLine key={msg.id} msg={msg} />;
          return <AssistantBubble key={msg.id} msg={msg} />;
        })}
        {status === 'connecting' || status === 'streaming' ? (
          <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            <span>{status === 'connecting' ? 'Connecting…' : 'Streaming…'}</span>
          </div>
        ) : null}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
};
