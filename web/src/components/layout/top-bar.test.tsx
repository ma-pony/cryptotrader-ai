import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { expect, it, vi } from 'vitest';
import '@/lib/i18n';
import { TopBar } from './top-bar';

vi.mock('@/hooks/use-market-data-ws', () => ({
  useMarketDataWS: () => ({ connectionStatus: 'disconnected', tickerData: undefined }),
}));

it.each([
  ['/accounts/connections/venue-123', '账户详情', '平台连接', '/accounts/connections'],
  ['/engine/components/custom-signal', '组件结果', '引擎', '/engine'],
  ['/research/backtests/run-123', '回测详情', '研究', '/research'],
])('names the destination and offers its parent on %s', (path, current, parent, href) => {
  render(
    <MemoryRouter initialEntries={[path]}>
      <TopBar />
    </MemoryRouter>,
  );
  const breadcrumb = within(screen.getByRole('navigation', { name: '导航路径' }));
  expect(breadcrumb.getByText(current)).toHaveAttribute('aria-current', 'page');
  expect(breadcrumb.getByRole('link', { name: parent })).toHaveAttribute('href', href);
  expect(breadcrumb.queryByText(/venue-123|custom-signal|run-123/)).not.toBeInTheDocument();
});
