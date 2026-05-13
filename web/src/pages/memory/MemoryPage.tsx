/**
 * MemoryPage — skill side only after 2026-05-13.
 * /memory 路由页面：仅展示 Skills Grid。
 */

import { useTranslation } from 'react-i18next';

import { PageBoundary } from '@/components/ui/page-boundary';
import { PageHeader } from '@/components/ui/page-header';

import { SkillsGrid } from './components/SkillsGrid';

const MemoryPage = () => {
  const { t } = useTranslation('memory');

  return (
    <PageBoundary>
      <div className="space-y-6">
        <PageHeader
          title={t('title', { defaultValue: '技能库' })}
          subtitle={t('description', {
            defaultValue: 'AI agent 加载的 SKILL.md 列表与 access 统计',
          })}
        />

        <section aria-labelledby="memory-skills-heading">
          <SkillsGrid />
        </section>
      </div>
    </PageBoundary>
  );
};

export default MemoryPage;
