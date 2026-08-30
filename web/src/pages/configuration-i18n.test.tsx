import { cleanup, fireEvent, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { workflowHarness } from '@/test/configuration-workflow';

describe('configuration control plane in English', () => {
  beforeEach(() => i18n.changeLanguage('en-US'));
  afterEach(async () => {
    cleanup();
    await i18n.changeLanguage('zh-CN');
  });

  it('renders the setup-required flow in English', async () => {
    workflowHarness('/');
    expect(await screen.findByRole('heading', { name: 'Commission trading system' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Activate trading system' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Next stage' })).not.toBeInTheDocument();
  });

  it('renders venue controls and fields in English', async () => {
    workflowHarness('/settings/venues');
    expect(await screen.findByRole('heading', { name: 'Venue connections' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Add connection' }));
    expect(
      within(screen.getByRole('form', { name: 'Add venue connection' })).getByLabelText(
        'Initial simulated equity (USDT)',
      ),
    ).toHaveValue(10000);
  });

  it('renders execution book controls in English', async () => {
    workflowHarness('/settings/execution-books');
    expect(await screen.findByRole('heading', { name: 'Execution books' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add execution book' })).toBeInTheDocument();
  });

  it('renders strategy controls in English', async () => {
    workflowHarness('/strategy');
    expect(await screen.findByRole('heading', { name: 'Signals and weights' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save configuration' })).toBeInTheDocument();
  });
});
