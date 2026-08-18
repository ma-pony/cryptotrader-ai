/**
 * spec 018 Memory page component tests — tests/unit/memory-page.test.tsx
 *
 * SC-Z13: >= 4 use cases PASS.
 * Tests: RulesGrid / CasesTimeline / ArchivedRules / RecentTransitions
 */

import { render, screen } from '@testing-library/react';
import i18n from 'i18next';
import { I18nextProvider } from 'react-i18next';
import { describe, expect, it, vi, beforeEach } from 'vitest';

// ── Mock @tanstack/react-query (must be before component imports) ───────────

vi.mock('@tanstack/react-query', () => ({
  useQuery: vi.fn(),
}));

// ── Mock @/lib/format to avoid jsdom issues ────────────────────────────────

vi.mock('@/lib/format', () => ({
  formatDateTime: (s: string) => s,
  formatPnl: (n: number) => String(n),
  formatNumber: (n: number) => String(n),
}));

// ── Component imports (after mocks) ───────────────────────────────────────

import { SkillsGrid } from '@/pages/memory/components/SkillsGrid';
import { useQuery } from '@tanstack/react-query';

const mockUseQuery = vi.mocked(useQuery);

// ── i18n stub ──────────────────────────────────────────────────────────────

const i18nInstance = i18n.createInstance();
void i18nInstance.init({
  lng: 'zh-CN',
  defaultNS: 'memory',
  resources: {
    'zh-CN': {
      memory: {
        title: '记忆演化',
        loading: '加载中…',
        rules_grid: { title: '规则状态矩阵', agent: 'Agent' },
        cases_timeline: { title: 'IVE 分类时间线 (24h)', empty: '最近 24h 无 case 记录' },
        archived_rules: { title: '已归档规则', empty: '暂无归档规则' },
        transitions: { title: 'FSM 状态转换', empty: '最近无状态转换' },
        maturity: {
          observed: '观察中',
          probationary: '试用期',
          active: '活跃',
          deprecated: '废弃',
          archived: '已归档',
        },
        skills: {
          title: 'Skills 列表',
          empty: '暂无 skill 记录',
          proposals: 'Skill 提案',
          agents: {
            tech: '技术分析',
            chain: '链上分析',
            news: '新闻情绪',
            macro: '宏观经济',
          },
        },
      },
    },
  },
});

function wrap(node: React.ReactNode) {
  return render(<I18nextProvider i18n={i18nInstance}>{node}</I18nextProvider>);
}

// ── Tests ──────────────────────────────────────────────────────────────────

beforeEach(() => {
  mockUseQuery.mockReset();
});

// ── spec 019 SkillsGrid tests (T041 a/b/c/d) ─────────────────────────────────

describe('SkillsGrid', () => {
  it('T041(a): renders skill names from data', () => {
    mockUseQuery.mockReturnValue({
      data: {
        items: [
          {
            name: 'tech-analysis',
            scope: 'agent:tech',
            version: '1.0',
            regime_tags: ['high_funding'],
            triggers_keywords: ['funding', 'trend'],
            importance: 0.8,
            confidence: 0.7,
            access_count: 5,
            last_accessed_at: '2026-05-09T12:00:00Z',
            manually_edited: false,
            description: 'Technical analysis skill for momentum detection.',
          },
          {
            name: 'chain-analysis',
            scope: 'agent:chain',
            version: '1.0',
            regime_tags: [],
            triggers_keywords: ['exchange_flow'],
            importance: 0.7,
            confidence: 0.7,
            access_count: 3,
            last_accessed_at: null,
            manually_edited: true,
            description: 'On-chain analysis skill.',
          },
        ],
        total: 2,
      },
      isLoading: false,
    } as ReturnType<typeof useQuery>);

    wrap(<SkillsGrid />);

    expect(screen.getByText('tech-analysis')).toBeInTheDocument();
    expect(screen.getByText('chain-analysis')).toBeInTheDocument();
  });

  it('T041(b): renders skill scope and importance values', () => {
    mockUseQuery.mockReturnValue({
      data: {
        items: [
          {
            name: 'macro-analysis',
            scope: 'agent:macro',
            version: '1.0',
            regime_tags: [],
            triggers_keywords: ['fed', 'rate'],
            importance: 0.9,
            confidence: 0.8,
            access_count: 12,
            last_accessed_at: '2026-05-09T08:00:00Z',
            manually_edited: false,
            description: 'Macro economic analysis with Fed policy focus.',
          },
        ],
        total: 1,
      },
      isLoading: false,
    } as ReturnType<typeof useQuery>);

    wrap(<SkillsGrid />);

    expect(screen.getByText('macro-analysis')).toBeInTheDocument();
    expect(screen.getByText('agent:macro')).toBeInTheDocument();
    // importance 0.90 rendered
    expect(screen.getByText(/0\.90/)).toBeInTheDocument();
  });

  it('T041(c): renders regime_tags as badges', () => {
    mockUseQuery.mockReturnValue({
      data: {
        items: [
          {
            name: 'funding-skill',
            scope: 'shared',
            version: '1.0',
            regime_tags: ['high_funding', 'extreme_greed'],
            triggers_keywords: ['funding'],
            importance: 0.75,
            confidence: 0.7,
            access_count: 2,
            last_accessed_at: null,
            manually_edited: false,
            description: 'Skill optimized for high funding rate regimes.',
          },
        ],
        total: 1,
      },
      isLoading: false,
    } as ReturnType<typeof useQuery>);

    wrap(<SkillsGrid />);

    expect(screen.getByText('high_funding')).toBeInTheDocument();
    expect(screen.getByText('extreme_greed')).toBeInTheDocument();
  });

  it('T041(d): shows empty state when no skills', () => {
    mockUseQuery.mockReturnValue({
      data: { items: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useQuery>);

    wrap(<SkillsGrid />);
    expect(screen.getByText('暂无 skill 记录')).toBeInTheDocument();
  });
});
