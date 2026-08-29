import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { AllocationPreview } from './allocation-preview';
import { validateBooks } from './book-form';

describe('ExecutionBooksPage', () => {
  it('blocks mixed capital scopes and previews exact weighted notionals', () => {
    render(<AllocationPreview equity={100000} targetExposure={0.5} allocations={[{ connectionId: 'paper', label: 'Paper', weight: 40 }, { connectionId: 'demo', label: 'Demo', weight: 60 }]} />);
    expect(screen.getByText('20,000 USDT')).toBeInTheDocument();
    expect(screen.getByText('30,000 USDT')).toBeInTheDocument();
    expect(validateBooks([{ id: 'sim', label: '模拟资金池', capital_scope: 'simulated', enabled: true, hitl_required: true, allocations: [{ connection_id: 'live', enabled: true, weight: 1 }] }], [{ id: 'live', label: 'Live', adapter_id: 'okx', environment: 'live', enabled: true, leverage: 1, margin_mode: 'cross', parameters: {} }])).toContain('模拟资金池不能包含实盘连接或未启用连接');
  });
});
