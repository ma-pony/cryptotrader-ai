import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, it, vi } from 'vitest';
import { RuntimeConfigSchema } from '@/types/api.schema';
import { runtimeConfigFixture } from '@/test/runtime-config-fixture';

const update = vi.fn();
const save = vi.fn();
vi.mock('@/pages/settings/configuration-context', () => ({ useConfiguration: () => ({ document: { notifications: { enabled: true, webhook_url: 'https://example.test/hook', webhook_timeout: 5, events: ['daily_summary', 'execution_failed'] } }, errors: {}, update, save, isDirty: () => false, status: {}, failure: null, applyStatus: 'applied', conflict: null, isSaving: false, isReloading: false, discard: vi.fn(), reload: vi.fn() }) }));
vi.mock('@/hooks/use-alerts', () => ({ useAlertOverview: () => ({ data: { alerts: [], deliveries: [] }, retry: vi.fn(), isRetrying: false }) }));
import NotificationSettings from './notifications';

beforeEach(() => update.mockClear());
it('keeps event choices when notification enablement changes and explains empty delivery state', async () => {
  render(<NotificationSettings />);
  expect(screen.getByText(/站内事项始终保留/)).toBeInTheDocument();
  expect(screen.getByText(/尚无投递记录/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole('checkbox', { name: '启用 Webhook 投递' }));
  expect(update).toHaveBeenCalledWith('notifications', expect.objectContaining({ enabled: false, events: ['daily_summary', 'execution_failed'] }));
  await userEvent.click(screen.getByRole('checkbox', { name: '账户连接失败' }));
  expect(update).toHaveBeenLastCalledWith('notifications', expect.objectContaining({ events: ['daily_summary', 'execution_failed', 'connection_failed'] }));
});

it('rejects unknown notification events at the shared runtime DTO boundary', () => {
  const config = structuredClone(runtimeConfigFixture()) as unknown as {
    document: { notifications: { events: string[] } };
  };
  config.document.notifications.events = ['daily_summary', 'eighth_event'];
  expect(RuntimeConfigSchema.safeParse(config).success).toBe(false);
});
