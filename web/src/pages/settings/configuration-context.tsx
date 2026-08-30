import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { useConfigurationCatalog } from '@/hooks/use-configuration-catalog';
import { useConfigurationDraft } from '@/hooks/use-configuration-draft';
import type { Connection, ConnectionCheck } from '@/lib/configuration-readiness';
import type { ConfigurationDraft } from '@/types/api';

type BookRow = { key: string; persistedId?: string };

function useOwner() {
  const catalog = useConfigurationCatalog();
  const draft = useConfigurationDraft(catalog.data);
  const [checks, setChecks] = useState<Record<string, ConnectionCheck>>({});
  // Dirty rows retain their original persisted identity, never the text being edited.
  const [draftBookRows, setBookRows] = useState<BookRow[]>([]);
  const bookRows = draft.isDirty('books')
    ? draftBookRows
    : (draft.baseline?.execution.books.map((book) => ({ key: book.id, persistedId: book.id })) ?? []);
  const [venueDrafts, setVenueDrafts] = useState<Record<string, ConfigurationDraft<Connection>>>({});
  const dirty = draft.dirty || Object.keys(venueDrafts).length > 0;
  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = '';
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [dirty]);
  return {
    ...draft,
    catalog,
    checks,
    setChecks,
    bookRows,
    setBookRows,
    venueDrafts,
    setVenueDrafts,
    hasUnsaved: dirty,
  };
}
const ConfigurationContext = createContext<ReturnType<typeof useOwner> | null>(null);
export function ConfigurationProvider({ children }: { children: ReactNode }) {
  const owner = useOwner();
  return <ConfigurationContext.Provider value={owner}>{children}</ConfigurationContext.Provider>;
}
export function useConfiguration() {
  const context = useContext(ConfigurationContext);
  if (!context) throw new Error('ConfigurationProvider is required');
  return context;
}
