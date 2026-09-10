/**
 * spec 020a T028 — SkillsGrid triggers_keywords badges + inference_failed flag.
 */
import { fireEvent, render, screen } from '@testing-library/react';
import i18n from 'i18next';
import { I18nextProvider } from 'react-i18next';
import { describe, expect, it, vi } from 'vitest';
import memoryZh from '@/locales/zh-CN/memory.json';

import { SkillsGrid } from './skills-grid';

// Mock the queries module so we don't need a live API / React Query Provider
vi.mock('./queries', () => ({
  useSkills: vi.fn(),
}));

import { useSkills } from './queries';

const i18nInstance = i18n.createInstance();
void i18nInstance.init({
  lng: 'zh-CN',
  defaultNS: 'memory',
  resources: { 'zh-CN': { memory: memoryZh } },
});

function makeItem(overrides: Partial<Parameters<typeof useSkills>[0]> = {}) {
  return {
    name: 'test-skill',
    scope: 'global',
    version: '1',
    regime_tags: [],
    triggers_keywords: [],
    importance: 0.8,
    confidence: 0.9,
    access_count: 3,
    last_accessed_at: null,
    manually_edited: false,
    description: 'A test skill',
    inference_failed: false,
    ...overrides,
  };
}

function renderGrid() {
  return render(
    <I18nextProvider i18n={i18nInstance}>
      <SkillsGrid />
    </I18nextProvider>,
  );
}

it('shows a retryable read failure instead of an empty skills list', () => {
  const refetch = vi.fn();
  vi.mocked(useSkills).mockReturnValue({ data: undefined, isLoading: false, isError: true, refetch } as never);
  renderGrid();
  expect(screen.getByRole('alert')).toHaveTextContent('技能资料加载失败');
  expect(screen.queryByText(/暂无.*记录/)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button'));
  expect(refetch).toHaveBeenCalledOnce();
});

describe('SkillsGrid — triggers_keywords badges', () => {
  it('renders triggers_keywords badges when present', () => {
    vi.mocked(useSkills).mockReturnValue({
      data: { items: [makeItem({ triggers_keywords: ['BTC', 'breakout', 'momentum'] } as never)], total: 1 },
      isLoading: false,
    } as never);

    renderGrid();
    expect(screen.getByText('BTC')).toBeInTheDocument();
    expect(screen.getByText('breakout')).toBeInTheDocument();
    expect(screen.getByText('momentum')).toBeInTheDocument();
  });

  it('shows at most 5 badges and "+N more" for overflow', () => {
    vi.mocked(useSkills).mockReturnValue({
      data: {
        items: [
          makeItem({
            triggers_keywords: ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7'],
          } as never),
        ],
        total: 1,
      },
      isLoading: false,
    } as never);

    renderGrid();
    // first 5 visible
    expect(screen.getByText('k1')).toBeInTheDocument();
    expect(screen.getByText('k5')).toBeInTheDocument();
    // k6 and k7 not visible as badges
    expect(screen.queryByText('k6')).not.toBeInTheDocument();
    // overflow indicator present
    expect(screen.getByText('另有 2 项')).toBeInTheDocument();
  });

  it('renders no triggers_keywords container when list is empty', () => {
    vi.mocked(useSkills).mockReturnValue({
      data: { items: [makeItem({ triggers_keywords: [] } as never)], total: 1 },
      isLoading: false,
    } as never);

    renderGrid();
    expect(screen.queryByTestId('triggers-keywords')).not.toBeInTheDocument();
  });
});

describe('SkillsGrid — a11y aria-labels', () => {
  it('regime badge has aria-label="Regime: <tag>"', () => {
    vi.mocked(useSkills).mockReturnValue({
      data: { items: [makeItem({ regime_tags: ['bull', 'high_vol'] } as never)], total: 1 },
      isLoading: false,
    } as never);

    renderGrid();
    expect(screen.getByRole('generic', { name: '市场状态：bull' })).toBeInTheDocument();
    expect(screen.getByRole('generic', { name: '市场状态：high_vol' })).toBeInTheDocument();
  });

  it('triggers keyword badge has aria-label="Trigger keyword: <kw>"', () => {
    vi.mocked(useSkills).mockReturnValue({
      data: { items: [makeItem({ triggers_keywords: ['BTC', 'breakout'] } as never)], total: 1 },
      isLoading: false,
    } as never);

    renderGrid();
    expect(screen.getByRole('generic', { name: '触发关键词：BTC' })).toBeInTheDocument();
    expect(screen.getByRole('generic', { name: '触发关键词：breakout' })).toBeInTheDocument();
  });

  it('inference_failed badge has aria-label="Inference failed during proposal"', () => {
    vi.mocked(useSkills).mockReturnValue({
      data: { items: [makeItem({ inference_failed: true } as never)], total: 1 },
      isLoading: false,
    } as never);

    renderGrid();
    expect(screen.getByRole('generic', { name: '提案推断失败' })).toBeInTheDocument();
  });
});

describe('SkillsGrid — inference_failed flag', () => {
  it('shows inference-failed badge when inference_failed is true', () => {
    vi.mocked(useSkills).mockReturnValue({
      data: { items: [makeItem({ inference_failed: true } as never)], total: 1 },
      isLoading: false,
    } as never);

    renderGrid();
    expect(screen.getByTestId('inference-failed-badge')).toBeInTheDocument();
    expect(screen.getByText('提案推断失败')).toBeInTheDocument();
  });

  it('does not show inference-failed badge when inference_failed is false', () => {
    vi.mocked(useSkills).mockReturnValue({
      data: { items: [makeItem({ inference_failed: false } as never)], total: 1 },
      isLoading: false,
    } as never);

    renderGrid();
    expect(screen.queryByTestId('inference-failed-badge')).not.toBeInTheDocument();
  });

  it('does not show inference-failed badge when inference_failed is absent', () => {
    const item = makeItem() as never;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (item as any).inference_failed;
    vi.mocked(useSkills).mockReturnValue({
      data: { items: [item], total: 1 },
      isLoading: false,
    } as never);

    renderGrid();
    expect(screen.queryByTestId('inference-failed-badge')).not.toBeInTheDocument();
  });
});
