import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';

import { MoneyValues } from './money';

it('normalizes exponent-form Decimal zero without changing nonzero ledger precision', () => {
  render(
    <MoneyValues
      values={[
        { amount: '0E-19', currency: 'USDT', unavailable_reason: null },
        { amount: '1.2300', currency: 'BTC', unavailable_reason: null },
      ]}
    />,
  );

  expect(screen.getByText('0 USDT')).toBeInTheDocument();
  expect(screen.queryByText('0E-19 USDT')).not.toBeInTheDocument();
  expect(screen.getByText('1.2300 BTC')).toBeInTheDocument();
});
