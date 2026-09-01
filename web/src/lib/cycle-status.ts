import type { TFunction } from 'i18next';

export type CycleStatusTone = 'neutral' | 'success' | 'warning' | 'danger';

const toneByStatus: Record<string, CycleStatusTone> = {
  completed: 'success',
  executed: 'success',
  approved: 'success',
  filled: 'success',
  closed: 'success',
  ready: 'success',
  no_change: 'neutral',
  partial: 'warning',
  pending: 'warning',
  queued: 'warning',
  running: 'warning',
  interrupted: 'warning',
  skipped: 'neutral',
  open: 'warning',
  partially_filled: 'warning',
  awaiting_approval: 'warning',
  requires_attention: 'warning',
  failed: 'danger',
  rejected: 'danger',
  approval_rejected: 'danger',
  risk_rejected: 'danger',
  component_failed: 'danger',
  cycle_failed: 'danger',
  execution_failed: 'danger',
  cancelled: 'danger',
  canceled: 'danger',
  invalidated: 'danger',
  unavailable: 'danger',
};

export const cycleStatusTone = (status: string): CycleStatusTone => toneByStatus[status] ?? 'neutral';

const knownStatuses = new Set(Object.keys(toneByStatus));

export const formatCycleStatus = (t: TFunction, status: string) =>
  knownStatuses.has(status)
    ? t(`status.${status}`, { ns: 'cycles' })
    : t('status.unknown', { ns: 'cycles', status });
