import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { useConfigurationCatalog } from '@/hooks/use-configuration-catalog';
import { useConfigurationDraft } from '@/hooks/use-configuration-draft';
import type { Connection, ConnectionCheck } from '@/lib/configuration-readiness';
import type { ConfigurationDraft } from '@/types/api';

function useOwner() {
  const catalog = useConfigurationCatalog();
  const draft = useConfigurationDraft(catalog.data);
  const [checks, setChecks] = useState<Record<string, ConnectionCheck>>({});
  // Presentation identity is independent from the editable persistent book ID.
  const [bookKeys, setBookKeys] = useState(() => draft.document?.execution.books.map(() => crypto.randomUUID()) ?? []);
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
    bookKeys,
    setBookKeys,
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
