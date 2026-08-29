import { useParams } from 'react-router';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useMultiVenueCycle } from '@/hooks/use-multi-venue-cycles';

export default function CycleDetailPage() {
  const { cycleId } = useParams<{ cycleId: string }>(); const cycle = useMultiVenueCycle(cycleId);
  if (cycle.isError) return <PageBoundary><p className="text-destructive">Unable to load cycle.</p></PageBoundary>;
  const value = cycle.data; if (!value) return <PageBoundary><p>Loading…</p></PageBoundary>;
  return <PageBoundary><div className="space-y-5"><PageHeader title={`Cycle ${value.cycle_id}`} />{value.requires_attention ? <p className="rounded border border-amber-500 bg-amber-500/10 p-3 font-medium">Requires attention</p> : null}<section className="rounded-lg border border-border p-4"><h2 className="font-semibold">Shared signal evidence</h2>{value.shared_signals.components.map((signal) => <div key={signal.component_id} className="mt-2 text-sm"><b>{signal.component_id}</b> · {signal.direction} · {signal.reasoning}</div>)}</section>{value.books.map((book) => <section key={book.book_id} className="rounded-lg border border-border p-4"><h2 className="font-semibold">{book.book_id} · {book.capital_scope} · {book.status}</h2>{book.execution?.requires_attention ? <p className="mt-2 text-amber-600">Requires attention</p> : null}{book.connections.map((connection) => <div key={connection.connection_id} className="mt-3 rounded border border-border p-3"><b>{connection.connection_id}</b>{connection.unavailable ? ' · unavailable' : ''}{connection.execution ? <div className="mt-1 text-xs">{connection.execution.status} · {connection.execution.error_operation || 'no execution error'} · {connection.execution.orders.length} orders</div> : null}</div>)}</section>)}</div></PageBoundary>;
}
