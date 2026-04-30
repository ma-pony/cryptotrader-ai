/**
 * PairBadge — vitest coverage for spec 013 T038.
 */
import { render, screen } from '@testing-library/react';
import i18n from 'i18next';
import { I18nextProvider } from 'react-i18next';
import { describe, expect, it } from 'vitest';

import { PairBadge } from './PairBadge';

const i18nInstance = i18n.createInstance();
void i18nInstance.init({
  lng: 'zh-CN',
  defaultNS: 'common',
  resources: {
    'zh-CN': {
      common: {
        pair: {
          market_type: {
            spot: '现货',
            swap: '永续',
            future: '交割',
            option: '期权',
          },
        },
      },
    },
  },
});

function renderBadge(props: Parameters<typeof PairBadge>[0]) {
  return render(
    <I18nextProvider i18n={i18nInstance}>
      <PairBadge {...props} />
    </I18nextProvider>,
  );
}

describe('PairBadge', () => {
  it('renders spot pair with 现货 label', () => {
    renderBadge({ pair: 'BTC/USDT', marketType: 'spot' });
    expect(screen.getByText('BTC/USDT')).toBeInTheDocument();
    expect(screen.getByText('现货')).toBeInTheDocument();
  });

  it('strips ccxt suffix from perp symbol display', () => {
    renderBadge({ pair: 'BTC/USDT:USDT', marketType: 'swap' });
    expect(screen.getByText('BTC/USDT')).toBeInTheDocument();
    expect(screen.getByText('永续')).toBeInTheDocument();
  });

  it('renders inverse perp with 永续 label', () => {
    renderBadge({ pair: 'BTC/USD:BTC', marketType: 'swap' });
    expect(screen.getByText('BTC/USD')).toBeInTheDocument();
    expect(screen.getByText('永续')).toBeInTheDocument();
  });

  it('renders future market with 交割 label', () => {
    renderBadge({ pair: 'BTC/USDT:USDT-241227', marketType: 'future' });
    expect(screen.getByText('BTC/USDT')).toBeInTheDocument();
    expect(screen.getByText('交割')).toBeInTheDocument();
  });

  it('uses pair_display as tooltip when provided', () => {
    renderBadge({ pair: 'ETH/USDT:USDT', pairDisplay: 'ETH/USDT (perp)', marketType: 'swap' });
    const wrapper = screen.getByText('ETH/USDT').closest('span[title]');
    expect(wrapper).toHaveAttribute('title', 'ETH/USDT (perp)');
  });

  it('falls back to canonical pair as tooltip when pair_display absent', () => {
    renderBadge({ pair: 'BTC/USDT:USDT', marketType: 'swap' });
    const wrapper = screen.getByText('BTC/USDT').closest('span[title]');
    expect(wrapper).toHaveAttribute('title', 'BTC/USDT:USDT');
  });

  it('defaults to spot when marketType is omitted', () => {
    renderBadge({ pair: 'BTC/USDT' });
    expect(screen.getByText('现货')).toBeInTheDocument();
  });
});
