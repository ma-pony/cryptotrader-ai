import { useId, type ReactNode } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './tabs';

export function SectionTabs({
  label,
  value,
  onValueChange,
  items,
}: {
  label: string;
  value: string;
  onValueChange: (value: string) => void;
  items: { id: string; label: string; content: ReactNode; dirty?: boolean }[];
}) {
  const id = useId();
  return (
    <Tabs value={value} onValueChange={onValueChange} className="section-tabs">
      {items.some((item) => item.dirty) ? (
        <span id={`${id}-dirty`} className="sr-only">
          有未保存修改
        </span>
      ) : null}
      <TabsList aria-label={label} className="section-tabs-list">
        {items.map((item) => (
          <TabsTrigger key={item.id} value={item.id} aria-describedby={item.dirty ? `${id}-dirty` : undefined}>
            {item.label}
            {item.dirty ? <span className="ml-2 h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" /> : null}
          </TabsTrigger>
        ))}
      </TabsList>
      {items.map((item) => (
        <TabsContent
          key={item.id}
          value={item.id}
          forceMount
          hidden={item.id !== value}
          className="section-tab-content"
        >
          {item.content}
        </TabsContent>
      ))}
    </Tabs>
  );
}
