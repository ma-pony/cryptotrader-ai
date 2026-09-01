import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { Outlet } from 'react-router';
import { useConfigurationCatalog, useVenueEnvironmentDefinitions } from '@/hooks/use-configuration-catalog';
import { useConfigurationDraft } from '@/hooks/use-configuration-draft';
import type { Connection } from '@/lib/configuration-readiness';
import { useConnectionChecks } from '@/hooks/use-connection-checks';
import { ConfigurationAccess } from './configuration-access';

type BookRow = { key: string; persistedId?: string };

function useOwner() {
  const catalog = useConfigurationCatalog();
  const draft = useConfigurationDraft(catalog.data);
  const connectionChecks = useConnectionChecks(draft.baseline?.execution.connections ?? [], draft.credentialStates);
  const venueDefinitions = useVenueEnvironmentDefinitions(draft.baseline?.execution.connections ?? []);
  // Dirty rows retain their original persisted identity, never the text being edited.
  const [draftBookRows, setBookRows] = useState<BookRow[]>([]);
  const bookRows = draft.isDirty('books')
    ? draftBookRows
    : (draft.baseline?.execution.books.map((book) => ({ key: book.id, persistedId: book.id })) ?? []);
  const [venueDrafts, setVenueDrafts] = useState<Record<string, Connection>>({});
  const priorConnections = useRef(draft.baseline?.execution.connections);
  useEffect(() => {
    const latest = draft.baseline?.execution.connections;
    const prior = priorConnections.current;
    priorConnections.current = latest;
    if (!latest || !prior) return;
    const changed = latest.filter((c) => prior.some((old) => old.id === c.id && old.enabled !== c.enabled));
    if (!changed.length) return;
    setVenueDrafts((current) => {
      const next = { ...current };
      for (const connection of changed)
        if (next[connection.id]) next[connection.id] = { ...next[connection.id]!, enabled: connection.enabled };
      return next;
    });
  }, [draft.baseline?.execution.connections]);
  const [newVenueDraft, setNewVenueDraft] = useState<Connection | null>(null);
  const dirty = draft.dirty || Object.keys(venueDrafts).length > 0 || newVenueDraft !== null;
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
    ...connectionChecks,
    venueDefinitions,
    bookRows,
    setBookRows,
    venueDrafts,
    setVenueDrafts,
    newVenueDraft,
    setNewVenueDraft,
    hasUnsaved: dirty,
  };
}
const ConfigurationContext = createContext<ReturnType<typeof useOwner> | null>(null);
export function ConfigurationProvider({ children }: { children: ReactNode }) {
  const owner = useOwner();
  return <ConfigurationContext.Provider value={owner}>{children}</ConfigurationContext.Provider>;
}

/**
 * Keeps ordinary drafts alive only while moving between configuration-owning
 * domains. Read-only workbench, decision and research routes never mount this
 * owner. Secrets remain in their local write-only controls.
 */
export function ConfigurationScope() {
  const owner = useOwner();
  return (
    <ConfigurationContext.Provider value={owner}>
      <Outlet />
    </ConfigurationContext.Provider>
  );
}

/** Restricts only configuration editors; read-only facts stay available. */
export function ConfigurationEditorGate({ children }: { children?: ReactNode }) {
  const owner = useConfiguration();
  if (!owner.document)
    return (
      <ConfigurationAccess
        embedded
        runtime={{
          ...owner,
          reload: async () => {
            const result = await owner.reload();
            if (result.isSuccess && !result.error) await owner.catalog.refetch();
            return result;
          },
        }}
      />
    );
  if (owner.catalog.isPending) return <p role="status">正在读取配置目录…</p>;
  if (owner.catalog.isError || !owner.catalog.data)
    return (
      <div role="alert">
        <p>配置目录读取失败。事实页面仍可读取；编辑前请重试。</p>
        <button className="configuration-button" onClick={() => void owner.catalog.refetch()}>
          重试
        </button>
      </div>
    );
  return children ?? <Outlet />;
}
export function useConfiguration() {
  const context = useContext(ConfigurationContext);
  if (!context) throw new Error('ConfigurationProvider is required');
  return context;
}
