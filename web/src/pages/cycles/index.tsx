import { Link } from 'react-router';
import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';
import { useMultiVenueCycles } from '@/hooks/use-multi-venue-cycles';

export default function CyclesPage() {
  const cycles = useMultiVenueCycles();
  return <PageBoundary><div className="space-y-5"><PageHeader title="Cycles" />{cycles.isError ? <p className="text-destructive">Unable to load cycles.</p> : cycles.data?.items.length ? cycles.data.items.map((cycle) => <Link className="block rounded-lg border border-border p-4 hover:border-amber-500" key={cycle.cycle_id} to={`/cycles/${cycle.cycle_id}`}><div className="font-mono text-sm">{cycle.cycle_id}</div><div className="mt-1 text-xs text-muted-foreground">R{cycle.config_revision} · {cycle.cycle_status} · {cycle.books.length} books{cycle.requires_attention ? ' · requires attention' : ''}</div></Link>) : <p className="text-muted-foreground">No cycles recorded.</p>}</div></PageBoundary>;
}
