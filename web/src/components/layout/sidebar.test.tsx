import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/hooks/use-runtime-status', () => ({ useRuntimeStatus: vi.fn() }));
vi.mock('@/hooks/use-scheduler-status', () => ({ useSchedulerStatus: vi.fn() }));

import { useRuntimeStatus } from '@/hooks/use-runtime-status';
import { useSchedulerStatus } from '@/hooks/use-scheduler-status';
import { SidebarDrawerBody } from './sidebar';
import '@/lib/i18n';

describe('shared sidebar footer', () => {
  beforeEach(() => vi.clearAllMocks());

  it('does not turn pending or absent facts into a paused state', () => {
    vi.mocked(useRuntimeStatus).mockReturnValue({ isPending: true, isError: false, data: undefined } as unknown as ReturnType<typeof useRuntimeStatus>);
    vi.mocked(useSchedulerStatus).mockReturnValue({ isPending: false, isError: false, data: undefined } as unknown as ReturnType<typeof useSchedulerStatus>);
    render(<MemoryRouter><SidebarDrawerBody /></MemoryRouter>);
    expect(screen.getByText('正在读取运行状态')).toBeVisible();
    expect(screen.getByText('正在读取定时来源状态')).toBeVisible();
    expect(screen.queryByText('自动运行已暂停')).not.toBeInTheDocument();
  });

  it('distinguishes read errors from successful source facts', () => {
    vi.mocked(useRuntimeStatus).mockReturnValue({ isPending: false, isError: true, data: undefined } as unknown as ReturnType<typeof useRuntimeStatus>);
    vi.mocked(useSchedulerStatus).mockReturnValue({ isPending: false, isError: true, data: undefined } as unknown as ReturnType<typeof useSchedulerStatus>);
    const view = render(<MemoryRouter><SidebarDrawerBody /></MemoryRouter>);
    expect(screen.getByText('运行状态未知')).toBeVisible();
    expect(screen.getByText('定时来源未知')).toBeVisible();

    vi.mocked(useRuntimeStatus).mockReturnValue({ isPending: false, isError: false, data: { automation_enabled: true } } as unknown as ReturnType<typeof useRuntimeStatus>);
    vi.mocked(useSchedulerStatus).mockReturnValue({ isPending: false, isError: false, data: { enabled: false } } as unknown as ReturnType<typeof useSchedulerStatus>);
    view.rerender(<MemoryRouter><SidebarDrawerBody /></MemoryRouter>);
    expect(screen.getByText('自动运行已开启')).toBeVisible();
    expect(screen.getByText('定时来源未启用')).toBeVisible();
  });
});
