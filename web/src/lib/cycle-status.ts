import type { TFunction } from 'i18next';

export type CycleStatusTone = 'neutral' | 'success' | 'warning' | 'danger';

const toneByStatus: Record<string, CycleStatusTone> = {
  completed: 'success',
  executed: 'success',
  approved: 'success',
  ready: 'success',
  partial: 'warning',
  pending: 'warning',
  awaiting_approval: 'warning',
  requires_attention: 'warning',
  failed: 'danger',
  rejected: 'danger',
  invalidated: 'danger',
  unavailable: 'danger',
};

export const cycleStatusTone = (status: string): CycleStatusTone => toneByStatus[status] ?? 'neutral';

export const formatCycleStatus = (t: TFunction, status: string) =>
  t(`status.${status}`, { ns: 'cycles', defaultValue: status.replaceAll('_', ' ') });
