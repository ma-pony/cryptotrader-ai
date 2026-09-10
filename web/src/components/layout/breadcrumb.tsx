import { ChevronRight } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Link, useLocation } from 'react-router';
import { SETTINGS_SECTIONS } from '@/pages/settings/navigation';

const roots = {
  decisions: 'nav.decisions',
  engine: 'nav.engine',
  accounts: 'nav.accounts',
  research: 'nav.research',
  settings: 'nav.settings',
} as const;
const details = {
  'accounts/connections': ['connections', 'connectionDetail'],
  'accounts/books': ['books', 'bookDetail'],
  'engine/components': ['componentDetail', 'componentDetail'],
  'research/backtests': ['backtestDetail', 'backtestDetail'],
  'research/market': ['market', 'market'],
  'research/analysis': ['analysis', 'analysis'],
  'research/compare': ['compare', 'compare'],
} as const;

export function Breadcrumb() {
  const { t } = useTranslation();
  const { pathname } = useLocation();
  const parts = pathname.split('/').filter(Boolean);
  const root = parts[0] as keyof typeof roots | undefined;
  const rootKey = root ? roots[root] : undefined;
  const section = SETTINGS_SECTIONS.find((item) => item.path === pathname);
  const detail = details[parts.slice(0, 2).join('/') as keyof typeof details];
  const rootLabel = rootKey ? t(rootKey) : t('nav.workbench');
  const current = section
    ? t(`configuration:${section.label}`)
    : detail
      ? t(`header.pages.${detail[parts.length > 2 ? 1 : 0]}`)
      : root === 'decisions' && parts.length > 1
        ? t('header.pages.decisionDetail')
        : rootKey || !parts.length
          ? rootLabel
          : t('errors.not_found');
  const parents = parts.length > 1 && rootKey ? [{ label: rootLabel, to: `/${root}` }] : [];
  if (root === 'accounts' && detail && parts.length > 2) {
    parents.push({ label: t(`header.pages.${detail[0]}`), to: `/${parts.slice(0, 2).join('/')}` });
  }
  return (
    <nav className="app-breadcrumb" aria-label={t('header.breadcrumb')}>
      <ol>
        {parents.map((parent) => (
          <li key={parent.to} className="breadcrumb-parent">
            <Link to={parent.to}>{parent.label}</Link>
            <ChevronRight className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          </li>
        ))}
        <li className="min-w-0">
          <span aria-current="page" className="block truncate font-medium">
            {current}
          </span>
        </li>
      </ol>
    </nav>
  );
}
