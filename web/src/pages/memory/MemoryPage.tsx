/**
 * spec 018 — MemoryPage
 * /memory 路由页面：4 sections（RulesGrid / CasesTimeline / ArchivedRules / RecentTransitions）
 */

import { useTranslation } from 'react-i18next';

import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';

import { ArchivedRules } from './components/ArchivedRules';
import { CasesTimeline } from './components/CasesTimeline';
import { RecentTransitions } from './components/RecentTransitions';
import { RulesGrid } from './components/RulesGrid';
import { SkillsGrid } from './components/SkillsGrid';

const MemoryPage = () => {
  const { t } = useTranslation('memory');

  return (
    <PageBoundary>
      <div className="space-y-6">
        <PageHeader
          title={t('title', { defaultValue: '记忆演化' })}
          subtitle={t('description', {
            defaultValue: '规则 FSM 状态、IVE 分类历史、归档记录',
          })}
        />

        {/* Section 1: Rules matrix */}
        <section aria-labelledby="memory-rules-heading">
          <RulesGrid />
        </section>

        {/* Section 2: Two-column layout for timeline + transitions */}
        <section
          aria-labelledby="memory-activity-heading"
          className="grid grid-cols-1 gap-6 lg:grid-cols-2"
        >
          <CasesTimeline />
          <RecentTransitions />
        </section>

        {/* Section 3: Archived rules */}
        <section aria-labelledby="memory-archived-heading">
          <ArchivedRules />
        </section>

        {/* Section 4: Skills Grid (spec 019) */}
        <section aria-labelledby="memory-skills-heading">
          <SkillsGrid />
        </section>
      </div>
    </PageBoundary>
  );
};

export default MemoryPage;
