import { Link } from 'react-router';
import type { Readiness } from '@/types/api';

type Reason = Readiness['analysis']['reasons'][number];

const ownerPath = (path: string) => {
  if (path.startsWith('llm.')) return '/settings/models';
  if (
    path.startsWith('security.') ||
    path.startsWith('accounts.') ||
    path.startsWith('infrastructure.') ||
    path.startsWith('observability.')
  )
    return '/settings/security';
  if (path.startsWith('notifications.')) return '/settings/notifications';
  if (path.startsWith('execution.connections')) return '/accounts/connections';
  if (path.startsWith('execution.')) return '/accounts/books';
  return '/engine';
};

const ownerAnchor = (path: string) => {
  if (path.startsWith('llm.')) return 'models';
  if (path.startsWith('security.') || path.startsWith('accounts.') || path.startsWith('infrastructure.') || path.startsWith('observability.'))
    return 'system-settings';
  if (path.startsWith('notifications.')) return 'notifications';
  if (path.startsWith('execution.connections')) return 'execution.connections';
  if (path.startsWith('execution.')) return 'execution.books';
  return 'configuration';
};

export function readinessTarget(path: string) {
  const bookPrefix = 'execution.books.';
  if (path.startsWith(bookPrefix)) {
    const suffix = path.endsWith('.enabled') ? '.enabled' : path.endsWith('.allocations') ? '.allocations' : null;
    if (suffix) {
      const id = path.slice(bookPrefix.length, -suffix.length);
      if (id) return { pathname: '/accounts/books', fragment: `execution.books.${id}` };
    }
  }
  const connectionPrefix = 'execution.connections.';
  if (path.startsWith(connectionPrefix)) {
    const id = path.slice(connectionPrefix.length);
    if (id) return { pathname: '/accounts/connections', fragment: `execution.connections.${id}` };
  }
  const exact: Record<string, { pathname: string; fragment: string }> = {
    'execution.books': { pathname: '/accounts/books', fragment: 'execution.books' },
    'execution.connections': { pathname: '/accounts/connections', fragment: 'execution.connections' },
    'execution.pairs': { pathname: '/accounts/books', fragment: 'execution.pairs' },
    'execution.live_order_execution_enabled': {
      pathname: '/accounts/books',
      fragment: 'execution.live_order_execution_enabled',
    },
    'signals.components': { pathname: '/engine', fragment: 'signals' },
    market_data: { pathname: '/engine', fragment: 'market' },
    llm: { pathname: '/settings/models', fragment: 'models' },
    'infrastructure.redis_url': { pathname: '/settings/security', fragment: 'infrastructure.redis_url' },
    applied_revision: { pathname: '/engine', fragment: 'configuration' },
  };
  return exact[path] ?? { pathname: ownerPath(path), fragment: ownerAnchor(path) };
}

export function ReadinessNextStep({ reasons }: { reasons: Reason[] }) {
  const reason = reasons[0];
  if (!reason) return null;
  const target = readinessTarget(reason.path);
  return (
    <section className="operational-strip" aria-labelledby="next-step-title">
      <div className="min-w-0">
        <h2 id="next-step-title" className="font-semibold">
          下一步
        </h2>
        <p className="mt-1 text-muted-foreground">{reason.message}</p>
        <p className="mt-1 font-mono text-sm text-muted-foreground">字段：{reason.path}</p>
      </div>
      <Link
        className="configuration-button shrink-0 whitespace-nowrap"
        to={`${target.pathname}#${encodeURIComponent(target.fragment)}`}
      >
        前往设置
      </Link>
    </section>
  );
}
