import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it } from 'vitest';
import i18n from '@/lib/i18n';
import { workflowHarness } from '@/test/configuration-workflow';
beforeEach(() => i18n.changeLanguage('zh-CN'));
it('keeps /strategy as typed signal settings and saves only signals', async () => {
  const h = workflowHarness('/strategy');
  fireEvent.change(await screen.findByLabelText('中性阈值'), { target: { value: '0.3' } });
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(h.writes).toHaveLength(1));
  expect(h.writes[0]).toMatchObject({
    document: {
      signals: { neutral_threshold: 0.3 },
      llm: { models: { analysis: 'analysis' } },
      execution: { live_order_execution_enabled: false },
      system: { active: false },
    },
  });
  expect(screen.queryByRole('textbox', { name: /JSON/ })).not.toBeInTheDocument();
});
it('preserves failed signal edits and reports actual FastAPI field errors', async () => {
  const h = workflowHarness('/strategy');
  h.fail(422, [{ loc: ['body', 'document', 'signals', 'neutral_threshold'], msg: 'invalid', type: 'value_error' }]);
  const field = await screen.findByLabelText('中性阈值');
  fireEvent.change(field, { target: { value: '0.3' } });
  fireEvent.click(screen.getByRole('button', { name: '保存配置' }));
  await waitFor(() => expect(field).toHaveAttribute('aria-invalid', 'true'));
  expect(field).toHaveValue(0.3);
  expect(field).toHaveFocus();
});
