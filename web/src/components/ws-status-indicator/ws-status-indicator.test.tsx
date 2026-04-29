import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { I18nextProvider } from 'react-i18next';
import i18n from 'i18next';

import { WSStatusIndicator } from './ws-status-indicator';

const i18nInstance = i18n.createInstance();
void i18nInstance.init({
  lng: 'zh-CN',
  resources: {
    'zh-CN': {
      translation: {
        'ws.connecting': '连接中',
        'ws.reconnecting': '重连中…',
        'ws.degraded': '数据延迟中，轮询模式',
        'ws.degraded_interval': '数据延迟中，当前轮询间隔 {{interval}}',
        'ws.polling_mode': '轮询模式',
      },
    },
  },
});

function renderIndicator(props: Parameters<typeof WSStatusIndicator>[0]) {
  return render(
    <I18nextProvider i18n={i18nInstance}>
      <WSStatusIndicator {...props} />
    </I18nextProvider>,
  );
}

describe('WSStatusIndicator', () => {
  it('renders nothing when connected', () => {
    const { container } = renderIndicator({ status: 'connected' });
    expect(container.innerHTML).toBe('');
  });

  it('renders pulse dot when connecting', () => {
    renderIndicator({ status: 'connecting' });
    const el = screen.getByRole('status');
    expect(el).toBeInTheDocument();
    expect(el).toHaveAttribute('aria-label', '连接中');
  });

  it('renders reconnecting text', () => {
    renderIndicator({ status: 'reconnecting' });
    expect(screen.getByText('重连中…')).toBeInTheDocument();
  });

  it('renders polling mode with interval for degraded', () => {
    renderIndicator({ status: 'degraded', refetchInterval: 10_000 });
    expect(screen.getByText(/轮询模式/)).toBeInTheDocument();
    expect(screen.getByText(/10s/)).toBeInTheDocument();
  });

  it('renders polling mode with 60s interval', () => {
    renderIndicator({ status: 'degraded', refetchInterval: 60_000 });
    expect(screen.getByText(/60s/)).toBeInTheDocument();
  });

  it('renders polling mode without interval for disconnected', () => {
    renderIndicator({ status: 'disconnected' });
    expect(screen.getByText(/轮询模式/)).toBeInTheDocument();
  });

  it('has role=status and aria-live=polite for degraded', () => {
    renderIndicator({ status: 'degraded', refetchInterval: 10_000 });
    const el = screen.getByRole('status');
    expect(el).toHaveAttribute('aria-live', 'polite');
    expect(el).toHaveAttribute('aria-label');
  });

  it('has correct aria-label with interval', () => {
    renderIndicator({ status: 'degraded', refetchInterval: 10_000 });
    const el = screen.getByRole('status');
    expect(el.getAttribute('aria-label')).toContain('10s');
  });

  it('has correct aria-label without interval', () => {
    renderIndicator({ status: 'degraded' });
    const el = screen.getByRole('status');
    expect(el.getAttribute('aria-label')).toContain('轮询模式');
  });
});
