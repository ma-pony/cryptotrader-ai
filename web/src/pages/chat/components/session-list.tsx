import { MessageSquarePlus, Trash2 } from 'lucide-react';
import { type FC } from 'react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { ChatSession } from '@/stores/use-chat-store';
import { cn } from '@/lib/cn';

interface SessionListProps {
  sessions: ChatSession[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRemove: (id: string) => void;
}

export const SessionList: FC<SessionListProps> = ({ sessions, activeId, onSelect, onNew, onRemove }) => {
  const { t } = useTranslation('chat');

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-border p-3">
        <Button variant="outline" size="sm" className="w-full gap-1.5" onClick={onNew}>
          <MessageSquarePlus className="h-4 w-4" />
          {t('new_session')}
        </Button>
      </div>
      <ScrollArea className="flex-1">
        <div className="space-y-0.5 p-2">
          {sessions.map((s) => (
            <button
              key={s.id}
              type="button"
              className={cn(
                'flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm transition-colors',
                activeId === s.id ? 'bg-accent text-accent-foreground' : 'hover:bg-accent/50',
              )}
              onClick={() => onSelect(s.id)}
            >
              <span className="truncate">{s.title}</span>
              <button
                type="button"
                className="ml-2 shrink-0 rounded p-0.5 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-destructive/20"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove(s.id);
                }}
                aria-label={`Delete session ${s.title}`}
              >
                <Trash2 className="h-3.5 w-3.5 text-muted-foreground" />
              </button>
            </button>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
};
