import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';

import i18n from '@/lib/i18n';
import type { ApprovalRequest } from '@/types/api';

import { ApprovalItem } from './approval-item';

const approval = {
  approval_id: 'approval-1',
  cycle_id: 'cycle-1',
  pair: 'ETH/USDT',
  profile_revision: 8,
  trade_plan: {
    target: { side: 'short', size_ratio: 0.4 },
    stop_loss: 3_500,
    take_profit: 3_000,
    component_signals: [],
    fused_signal: { score: -0.55, reasoning: 'weighted', contributions: [] },
  },
  status: 'pending',
  decision_by: null,
  created_at: '2026-08-28T00:00:00Z',
  decided_at: null,
} as unknown as ApprovalRequest;

describe('target plan approval', () => {
  beforeEach(async () => {
    await i18n.changeLanguage('zh-CN');
  });

  it('renders the frozen target plan instead of a legacy verdict', () => {
    const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <ApprovalItem approval={approval} />
      </QueryClientProvider>,
    );

    expect(screen.getByText('目标空仓 40%')).toBeInTheDocument();
    expect(screen.getByText('Revision 8')).toBeInTheDocument();
  });
});
