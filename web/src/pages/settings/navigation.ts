export const SETTINGS_SECTIONS = [
  { id: 'models', path: '/settings/models', label: 'forms.modelsTitle' },
  { id: 'signals', path: '/strategy', label: 'forms.signalsTitle' },
  { id: 'market', path: '/settings/market', label: 'forms.marketTitle' },
  { id: 'venues', path: '/settings/venues', label: 'venues' },
  { id: 'books', path: '/settings/execution-books', label: 'books' },
  { id: 'risk', path: '/settings/risk', label: 'forms.riskTitle' },
  { id: 'scheduler', path: '/settings/scheduler', label: 'forms.schedulerTitle' },
  { id: 'system', path: '/settings/system', label: 'center.systemTitle' },
] as const;
