import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { ResultBlocks } from './result-blocks';

describe('saved result blocks', () => {
  it('renders arbitrary component tables in a keyboard-scrollable region', () => {
    render(
      <ResultBlocks
        blocks={[
          {
            kind: 'table',
            title: '测试组件结果',
            columns: [{ key: 'window', label: '观察窗口' }],
            rows: [{ cells: [{ column_key: 'window', value: 37 }] }],
          },
        ]}
      />,
    );
    expect(screen.getByRole('columnheader', { name: '观察窗口' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: '37' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: '测试组件结果' })).toHaveAttribute('tabindex', '0');
    expect(screen.getByRole('region', { name: '测试组件结果' })).toHaveClass('overflow-x-auto');
  });
  it('marks the saved forecast boundary and expands complete timeline with the keyboard control', async () => {
    render(
      <ResultBlocks
        blocks={[
          {
            kind: 'series',
            title: '保存的曲线',
            forecast_start: '2026-01-01T01:00:00Z',
            evaluation_target: 'candle_close',
            series: [
              {
                name: '样本',
                unit: 'USDT',
                points: [
                  { time: '2026-01-01T00:00:00Z', value: '100' },
                  { time: '2026-01-01T01:00:00Z', value: '105' },
                ],
              },
            ],
          },
          {
            kind: 'timeline',
            title: '完整辩论',
            entries: [{ time: '2026-01-01T00:00:00Z', actor: '测试智能体', body: '保存的完整原文' }],
          },
        ]}
      />,
    );
    expect(screen.getByRole('img', { name: '保存的曲线' })).toBeInTheDocument();
    expect(screen.getByText(/预测区/)).toBeInTheDocument();
    const expand = screen.getByRole('button', { name: '展开完整辩论' });
    expand.focus();
    await userEvent.keyboard('{Enter}');
    expect(expand).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('保存的完整原文')).toBeVisible();
  });
  it('renders text literally and never creates executable markup', () => {
    const { container } = render(
      <ResultBlocks blocks={[{ kind: 'text', title: '原文', body: '<script>alert(1)</script>' }]} />,
    );
    expect(screen.getByText('<script>alert(1)</script>')).toBeInTheDocument();
    expect(container.querySelector('script')).toBeNull();
  });
});
