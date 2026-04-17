import { describe, expect, it } from 'vitest';

import { formatCurrency, formatNumber, formatPercent, pnlClass } from '@/lib/format';

describe('format helpers', () => {
  it('formats USDT currency', () => {
    expect(formatCurrency(1234.5)).toMatch(/1,234/);
  });

  it('signs percent', () => {
    expect(formatPercent(0.1234)).toMatch(/\+/);
    expect(formatPercent(-0.05)).toMatch(/-/);
  });

  it('returns class names by sign', () => {
    expect(pnlClass(10)).toContain('success');
    expect(pnlClass(-1)).toContain('destructive');
    expect(pnlClass(0)).toContain('muted');
  });

  it('formats fractional numbers', () => {
    expect(formatNumber(3.14159, 2)).toMatch(/3\.14/);
  });
});
