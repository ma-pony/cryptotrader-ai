import { cleanup, fireEvent, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { workflowApprovedChecks, workflowConfig, workflowHarness } from '@/test/configuration-workflow';

describe('configuration control plane in Chinese', () => {
  beforeEach(() => i18n.changeLanguage('zh-CN'));
  afterEach(cleanup);

  it('keeps sidebar navigation operable through Chinese accessible labels', async () => {
    workflowHarness('/accounts/connections');
    await screen.findByRole('heading', { name: '平台连接' });
    const toggle = screen.getByRole('button', { name: '切换侧边栏' });
    const sidebar = screen.getByRole('complementary', { name: '主要导航' });
    expect(screen.getByTitle('接口访问密钥')).toHaveTextContent('密钥未提供');
    fireEvent.click(toggle);
    expect(sidebar).toHaveClass('w-16');
    fireEvent.click(toggle);
    expect(sidebar).toHaveClass('w-60');
  });
});

describe('configuration control plane in English', () => {
  beforeEach(() => i18n.changeLanguage('en-US'));
  afterEach(async () => {
    cleanup();
    await i18n.changeLanguage('zh-CN');
  });

  it('keeps English configuration links reachable without a global activation gate', async () => {
    const config = workflowConfig();
    const h = workflowHarness('/settings/models', config, undefined, workflowApprovedChecks(config));
    const sidebar = await screen.findByRole('complementary', { name: 'primary' });
    expect(within(sidebar).getAllByRole('link')).toHaveLength(6);
    expect(await screen.findByRole('heading', { name: 'Models and gateway' })).toBeInTheDocument();
    expect(h.saved().document.scheduler.automation_enabled).toBe(false);
    expect(screen.queryByRole('button', { name: 'Next stage' })).not.toBeInTheDocument();
  });

  it('renders venue controls and fields in English', async () => {
    workflowHarness('/accounts/connections');
    expect(await screen.findByRole('heading', { name: 'Venue connections' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Add connection' }));
    const form = within(screen.getByRole('form', { name: 'Add venue connection' }));
    expect(await form.findByLabelText('Account code')).toHaveAttribute('type', 'password');
    expect(await form.findByLabelText('Access token')).toBeRequired();
    expect(await form.findByLabelText('Tenant PIN')).not.toBeRequired();
  });

  it('renders execution book controls in English', async () => {
    workflowHarness('/accounts/books');
    expect(await screen.findByRole('heading', { name: 'Execution books' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add execution book' })).toBeInTheDocument();
  });

  it('renders strategy controls in English', async () => {
    workflowHarness('/engine');
    expect(await screen.findByRole('heading', { name: 'Signals and weights' })).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'Save configuration' }).length).toBeGreaterThan(0);
  });
});
