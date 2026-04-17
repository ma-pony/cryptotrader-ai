import { Send, Square } from 'lucide-react';
import { useCallback, useRef, type FC, type KeyboardEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import type { StreamStatus } from '@/hooks/use-chat-messages';

interface ChatInputProps {
  onSend: (text: string) => void;
  onStop: () => void;
  status: StreamStatus;
}

export const ChatInput: FC<ChatInputProps> = ({ onSend, onStop, status }) => {
  const { t } = useTranslation('chat');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isStreaming = status === 'streaming' || status === 'connecting';

  const handleSend = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    const text = el.value.trim();
    if (!text) return;
    onSend(text);
    el.value = '';
    el.style.height = 'auto';
  }, [onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleInput = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${String(Math.min(el.scrollHeight, 160))}px`;
  }, []);

  return (
    <div className="border-t border-border bg-background p-3">
      <div className="flex items-end gap-2">
        <textarea
          ref={textareaRef}
          className="flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          placeholder={t('placeholder')}
          rows={1}
          disabled={isStreaming}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          aria-label={t('placeholder')}
        />
        {isStreaming ? (
          <Button variant="destructive" size="icon" onClick={onStop} aria-label={t('stop')}>
            <Square className="h-4 w-4" />
          </Button>
        ) : (
          <Button variant="primary" size="icon" onClick={handleSend} aria-label={t('send')}>
            <Send className="h-4 w-4" />
          </Button>
        )}
      </div>
    </div>
  );
};
